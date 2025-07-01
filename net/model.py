import torch
import math
import torch.nn as nn
from progSIC.loss.distortion import Distortion
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.ops import quantize_ste as ste_round
from progSIC.layer.analysis_transform import AnalysisTransform
from progSIC.layer.synthesis_transform import SynthesisTransform
from progSIC.layer.contextModel import ContextModel_AR
from progSIC.layer.entropy_parameters import GroupEntropyParameters
from progSIC.loss.metrics._msssim import MultiscaleStructuralSimilarity
from lpips import lpips
from DISTS_pytorch import DISTS
import numpy as np


class ProgSIC(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config.dim
        self.semDim_interval = config.semDim_interval
        self.num_slices = self.dim // self.semDim_interval
        self.sampleNum_train = config.sampleNum_train
        self.sampleNum_test = config.sampleNum_test

        self.ga = AnalysisTransform(**config.ga_kwargs)
        self.gs = SynthesisTransform(**config.gs_kwargs)
        self.ha = nn.Sequential(
            nn.Conv2d(self.dim, self.dim * 3 // 4, 3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.dim * 3 // 4, self.dim * 3 // 4, 5, stride=2, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.dim * 3 // 4, self.dim * 3 // 4, 5, stride=2, padding=2),
        )
        self.hs = nn.Sequential(
            nn.ConvTranspose2d(self.dim * 3 // 4, self.dim, 5,
                               stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.dim, self.dim * 3 // 2, 5,
                               stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.dim * 3 // 2, self.dim * 2, 3, stride=1, padding=1)
        )
        self.entropy_bottleneck = EntropyBottleneck(self.dim * 3 // 4)
        self.gaussian_conditional = GaussianConditional(None)
        self.context_model = ContextModel_AR(self.dim)
        self.entropy_parameters = GroupEntropyParameters(self.dim * 2 * 2)
        self.distortion = Distortion(config)
        self.msssim = MultiscaleStructuralSimilarity(data_range=1.0, window_size=11)
        self.lpips = lpips.LPIPS(net='alex')
        self.dists = DISTS()
        self.H = self.W = 0

        self._init_conv_weights(self.ha)
        self._init_transConv_weights(self.hs)

    def _init_conv_weights(self, m):
        nn.init.kaiming_normal_(m[0].weight, mode="fan_out", nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(m[2].weight, mode="fan_out", nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(m[4].weight, mode="fan_out", nonlinearity='conv2d')

    def _init_transConv_weights(self, m):
        nn.init.kaiming_normal_(m[0].weight, mode="fan_out", nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(m[2].weight, mode="fan_out", nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(m[4].weight, mode="fan_out", nonlinearity='conv_transpose2d')

    def update_resolution(self, H, W):
        if H != self.H or W != self.W:
            self.ga.update_resolution(H, W)
            self.gs.update_resolution(H // 16, W // 16)
            self.H = H
            self.W = W

    def quantized_prob_based_guassian(self, feature, means, scales, training=True):
        if training is True:
            quantized = self.gaussian_conditional.quantize(feature, "noise")
        else:
            quantized = ste_round(feature)
        likelihoods = self.gaussian_conditional._likelihood(quantized, scales, means)
        likelihoods = torch.clamp(likelihoods, 1e-9, 1)
        return likelihoods


    def warming(self, input_image, training=True):
        B, C, H, W = input_image.shape
        self.update_resolution(H, W)

        # =============== step 1: forward process (Sequential Synonymous Mappings) ===============
        y = self.ga(input_image)
        z = self.ha(y)
        _, z_likelihoods = self.entropy_bottleneck(z)
        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset

        hyper_params = self.hs(z_hat)
        y_hat_syn = ste_round(y)

        context_params = self.context_model(y_hat_syn)
        guassian_params = self.entropy_parameters(torch.cat((hyper_params, context_params), dim=1))
        scales_hat, means_hat = guassian_params.chunk(2, 1)

        # ================ step 2: generative process (reconstruct the source) ===================
        # ******   step 2.1: obtain the syntactic sequence (y_hat_syn)
        y_likelihoods = self.quantized_prob_based_guassian(y, means_hat, scales_hat, training=training)
        y_split_likelihoods = y_likelihoods.chunk(self.num_slices, dim=1)

        # ******   step 2.1: reconstruct the image (x_hat_syn)
        x_hat_syn = self.gs(y_hat_syn).clamp(min=0, max=1)

        # ========================== calculate the loss ============================================
        mse_syntactic = self.distortion(input_image, x_hat_syn)
        lpips_syntactic = self.lpips(input_image, x_hat_syn, normalize=True).mean()

        bpp_y_splits = []
        for i in range(self.num_slices):
            bpp_y_split = torch.log(y_split_likelihoods[i]).sum() / (-math.log(2) * H * W) / B
            bpp_y_splits.append(bpp_y_split.reshape(1))
        bpp_y_splits_std = torch.std(torch.cat(bpp_y_splits))
        bpp_y_splits_min = torch.min(torch.cat(bpp_y_splits))

        bpp_z = torch.log(z_likelihoods).sum() / (-math.log(2) * H * W) / B  # bpp_z
        bpp_y = torch.log(y_likelihoods).sum() / (-math.log(2) * H * W) / B  # bpp_y

        return mse_syntactic, lpips_syntactic, bpp_y_splits_std, bpp_y_splits_min, bpp_y, bpp_z, x_hat_syn


    def forward(self, input_image, semDim, getReconFlag=False):
        B, C, H, W = input_image.shape
        self.update_resolution(H, W)

        # =============== step 1: forward process (Sequential Synonymous Mappings) ===============
        y = self.ga(input_image)
        z = self.ha(y)
        _, z_likelihoods = self.entropy_bottleneck(z)
        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset

        hyper_params = self.hs(z_hat)
        scales_hyper_hat, means_hyper_hat = hyper_params.chunk(2, 1)
        y_hat_syn = ste_round(y)

        context_params = self.context_model(y_hat_syn)
        guassian_params = self.entropy_parameters(torch.cat((hyper_params, context_params), dim=1))
        scales_hat, means_hat = guassian_params.chunk(2, 1)

        # ================ step 2: generative process (reconstruct the source) ===================
        # ******   step 2.1: obtain the syntactic sequence (y_hat_syn)
        y_likelihoods = self.quantized_prob_based_guassian(y, means_hat, scales_hat)

        quantized_y_likelihoods = self.quantized_prob_based_guassian(y, means_hat, scales_hat, training=False)
        y_split_likelihoods = quantized_y_likelihoods.chunk(self.num_slices, dim=1)

        # ******   step 2.2: obtain the sample in synonymous set (y_hat_semSample)
        scales_hat_samples, means_hat_samples = scales_hyper_hat[:, semDim:, :, :], means_hyper_hat[:, semDim:, :, :]
        y_hat_semDim = y_hat_syn[:, :semDim, :, :]
        y_sem_likelihoods, y_sample_likelihoods = y_likelihoods.split([semDim, self.dim - semDim], dim=1)
        y_sample_likelihoods.detach()

        y_hat_semDims = y_hat_semDim.repeat(self.sampleNum_train, 1, 1, 1)
        means_hat_samples = means_hat_samples.repeat(self.sampleNum_train, 1, 1, 1)
        scales_hat_samples = scales_hat_samples.repeat(self.sampleNum_train, 1, 1, 1)
        y_hat_samples = ste_round(means_hat_samples + 2 * torch.tensor(2 * np.random.rand(*scales_hat_samples.detach().cpu().numpy().shape) - 1, dtype=torch.float32).cuda())
        y_hat_semSamples = torch.cat([y_hat_semDims, y_hat_samples], dim=1)

        # ******  step 2.3: combaine y_hat_syn and y_hat_semSample to y_hat and reconstruct the source x_hat
        y_hat = torch.cat([y_hat_syn, y_hat_semSamples], dim=0)
        x_hat = self.gs(y_hat).clamp(min=0, max=1)

        # ================ step 3: reverse compress (verify the semantic synonymy and sample consistency)
        y_dis = self.ga(x_hat)

        # ========================== calculate the loss ============================================
        bpp_z = torch.log(z_likelihoods).sum() / (-math.log(2) * H * W) / B         # bpp_z
        bpp_y = torch.log(y_likelihoods).sum() / (-math.log(2) * H * W) / B         # bpp_y
        bpp_ys = torch.log(y_sem_likelihoods).sum() / (-math.log(2) * H * W) / B    # bpp_ys for semDim
        bpp_y_splits = []
        for i in range(self.num_slices):
            bpp_y_split = torch.log(y_split_likelihoods[i]).sum() / (-math.log(2) * H * W) / B
            bpp_y_splits.append(bpp_y_split.reshape(1))
        bpp_y_splits_std = torch.std(torch.cat(bpp_y_splits))
        bpp_y_splits_chunk = torch.log(y_split_likelihoods[semDim // self.semDim_interval -1]).sum() / (-math.log(2) * H * W) / B

        x_hat_syn, x_hat_semSamples = x_hat.split([B, x_hat.shape[0] - B], dim=0)
        mse_syntactic = self.distortion(input_image, x_hat_syn)
        lpips_syntactic = self.lpips(input_image, x_hat_syn, normalize=True).mean()
        dists_syntactic = self.dists(input_image, x_hat_syn).mean()

        mse_loss = []
        x_hat_semSamples = x_hat_semSamples.chunk(self.sampleNum_train, 0)
        for i in range(self.sampleNum_train):
            mse_loss.append(self.distortion(input_image, x_hat_semSamples[i]).mean().reshape(1))
        Emse_loss = torch.mean(torch.cat(mse_loss))

        lpips_semantic = []
        for i in range(self.sampleNum_train):
            lpips_semantic.append(self.lpips(input_image, x_hat_semSamples[i], normalize=True).mean().reshape(1))
        Elpips_semantic = torch.mean(torch.cat(lpips_semantic))

        dists_semantic = []
        for i in range(self.sampleNum_train):
            dists_semantic.append(self.dists(input_image, x_hat_semSamples[i]).mean().reshape(1))
        Edists_semantic = torch.mean(torch.cat(dists_semantic))

        y_dis_syn, y_dis_semSamples = y_dis.split([B, x_hat.shape[0] - B], dim=0)
        y_dis_semSamples = y_dis_semSamples.chunk(self.sampleNum_train, dim=0)
        y_hat_samples = y_hat_samples.chunk(self.sampleNum_train, dim=0)
        synonym_distance = []
        synonym_distance.append(self.distortion(y_hat_syn, y_dis_syn).mean().reshape(1))
        for i in range(self.sampleNum_train):
            synonym_distance.append(self.distortion(y_hat_semDim, y_dis_semSamples[i][:, :semDim, :, :]).mean().reshape(1))
        synonym_distance = torch.mean(torch.cat(synonym_distance))

        sample_distance = []
        sample_distance.append(torch.tensor([0], dtype=torch.float32).cuda().reshape(1))
        if semDim != self.dim:
            for i in range(self.sampleNum_train):
                sample_distance.append(self.distortion(y_hat_samples[i], y_dis_semSamples[i][:, semDim:, :, :]).mean().reshape(1))
        else:
            for i in range(self.sampleNum_train):
                sample_distance.append(torch.tensor([0], dtype=torch.float32).cuda().reshape(1))
        sample_distance = torch.mean(torch.cat(sample_distance))

        if not getReconFlag:
            return mse_syntactic, lpips_syntactic, dists_syntactic, Emse_loss, Elpips_semantic, Edists_semantic, synonym_distance, sample_distance, bpp_y, bpp_ys, bpp_z, bpp_y_splits_std, bpp_y_splits_chunk
        else:
            return mse_syntactic, lpips_syntactic, dists_syntactic, Emse_loss, Elpips_semantic, Edists_semantic, synonym_distance, sample_distance, bpp_y, bpp_ys, bpp_z, bpp_y_splits_std, bpp_y_splits_chunk, x_hat_syn, torch.cat(x_hat_semSamples, dim=0), y_hat_semDim, y_hat_semDims

    def test(self, input_image, semDim, quality_average=True, H_original=None, W_original=None, UniformSampling=True):
        B, C, H, W = input_image.shape
        self.update_resolution(H, W)
        if H_original == None or W_original == None:
            H_original = H
            W_original = W

        # =============== step 1: forward process (Sequential Synonymous Mappings) ===============
        y = self.ga(input_image)
        z = self.ha(y)
        _, z_likelihoods = self.entropy_bottleneck(z)
        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset

        hyper_params = self.hs(z_hat)
        scales_hyper_hat, means_hyper_hat = hyper_params.chunk(2, 1)
        y_hat_syn = ste_round(y)

        context_params = self.context_model(y_hat_syn)
        guassian_params = self.entropy_parameters(torch.cat((hyper_params, context_params), dim=1))
        scales_hat, means_hat = guassian_params.chunk(2, 1)

        # ================ step 2: generative process (reconstruct the source) ===================
        # ******   step 2.1: obtain the syntactic sequence (y_hat_syn)
        y_likelihoods = self.quantized_prob_based_guassian(y, means_hat, scales_hat, training=False)

        quantized_y_likelihoods = self.quantized_prob_based_guassian(y, means_hat, scales_hat, training=False)
        y_split_likelihoods = quantized_y_likelihoods.chunk(self.num_slices, dim=1)

        # ******   step 2.2: obtain the sample in synonymous set (y_hat_semSample)
        scales_hat_samples, means_hat_samples = scales_hyper_hat[:, semDim:, :, :], means_hyper_hat[:, semDim:, :, :]
        y_hat_semDim = y_hat_syn[:, :semDim, :, :]
        y_sem_likelihoods, y_sample_likelihoods = y_likelihoods.split([semDim, self.dim - semDim], dim=1)

        y_hat_semDims = y_hat_semDim.repeat(self.sampleNum_test, 1, 1, 1)
        means_hat_samples = means_hat_samples.repeat(self.sampleNum_test, 1, 1, 1)
        scales_hat_samples = scales_hat_samples.repeat(self.sampleNum_test, 1, 1, 1)
        if UniformSampling:
            y_hat_samples = ste_round(means_hat_samples + 2 * torch.tensor(2 * np.random.rand(*scales_hat_samples.detach().cpu().numpy().shape) - 1, dtype=torch.float32).to(scales_hat_samples.device))
        else:
            y_hat_samples = ste_round(means_hat_samples + scales_hat_samples * torch.tensor(np.random.randn(*scales_hat_samples.detach().cpu().numpy().shape), dtype=torch.float32).to(scales_hat_samples.device))
        y_hat_semSamples = torch.cat([y_hat_semDims, y_hat_samples], dim=1)

        # ******  step 2.3: combaine y_hat_syn and y_hat_semSample to y_hat and reconstruct the source x_hat
        y_hat = y_hat_semSamples
        x_hat = self.gs(y_hat).clamp(min=0, max=1)

        # ================ step 3: reverse compress (verify the semantic synonymy and sample consistency)
        y_dis = self.ga(x_hat)

        # ========================== calculate the loss ============================================
        bpp_z = torch.log(z_likelihoods).sum() / (-math.log(2) * H * W) / B           # bpp_z
        bpp_ys = torch.log(y_sem_likelihoods).sum() / (-math.log(2) * H * W) / B      # bpp_ys for semDim  # bpp_y for semDim
        bpp_y = torch.log(y_likelihoods).sum() / (-math.log(2) * H * W) / B           # bpp_y
        bpp_y_splits = []
        for i in range(self.num_slices):
            bpp_y_split = torch.log(y_split_likelihoods[i]).sum() / (-math.log(2) * H * W) / B
            bpp_y_splits.append(bpp_y_split.reshape(1))
        bpp_y_splits_std = torch.std(torch.cat(bpp_y_splits))
        bpp_y_splits_chunk = torch.log(y_split_likelihoods[semDim // self.semDim_interval - 1]).sum() / (-math.log(2) * H * W) / B

        x_hat_semSamples = x_hat

        mse_loss = []
        x_hat_semSamples = x_hat_semSamples.chunk(self.sampleNum_test, 0)
        for i in range(self.sampleNum_test):
            mse_loss.append(self.distortion(input_image[:, :, :H_original, :W_original], x_hat_semSamples[i][:, :, :H_original, :W_original]).mean().reshape(1))
        if quality_average:
            Emse_loss = torch.mean(torch.cat(mse_loss))
        else:
            Emse_loss = torch.cat(mse_loss)

        lpips_semantic = []
        for i in range(self.sampleNum_test):
            lpips_semantic.append(self.lpips(input_image[:, :, :H_original, :W_original], x_hat_semSamples[i][:, :, :H_original, :W_original], normalize=True).mean().reshape(1))
        if quality_average:
            Elpips_semantic = torch.mean(torch.cat(lpips_semantic))
        else:
            Elpips_semantic = torch.cat(lpips_semantic)

        dists_semantic = []
        for i in range(self.sampleNum_test):
            dists_semantic.append(self.dists(input_image[:, :, :H_original, :W_original], x_hat_semSamples[i][:, :, :H_original, :W_original]).mean().reshape(1))
        if quality_average:
            Edists_semantic = torch.mean(torch.cat(dists_semantic))
        else:
            Edists_semantic = torch.cat(dists_semantic)

        y_dis_semSamples = y_dis
        y_dis_semSamples = y_dis_semSamples.chunk(self.sampleNum_test, dim=0)
        y_hat_samples = y_hat_samples.chunk(self.sampleNum_test, dim=0)
        synonym_distance = []
        for i in range(self.sampleNum_test):
            synonym_distance.append(
                self.distortion(y_hat_semDim, y_dis_semSamples[i][:, :semDim, :, :]).mean().reshape(1))
        synonym_distance = torch.mean(torch.cat(synonym_distance))

        sample_distance = []
        if semDim != self.dim:
            for i in range(self.sampleNum_test):
                sample_distance.append(
                    self.distortion(y_hat_samples[i], y_dis_semSamples[i][:, semDim:, :, :]).mean().reshape(1))
        else:
            for i in range(self.sampleNum_test):
                sample_distance.append(torch.tensor([0], dtype=torch.float32).cuda().reshape(1))
        sample_distance = torch.mean(torch.cat(sample_distance))

        with torch.no_grad():
            msssim = []
            for i in range(self.sampleNum_test):
                msssim.append((1 - self.msssim(input_image[:, :, :H_original, :W_original], x_hat_semSamples[i][:, :, :H_original, :W_original])).mean().reshape(1))
            msssim = torch.mean(torch.cat(msssim))

        return Emse_loss, Elpips_semantic, Edists_semantic, synonym_distance, sample_distance, bpp_y, bpp_ys, bpp_z, bpp_y_splits_std, bpp_y_splits_chunk, msssim, torch.cat(x_hat_semSamples, dim=0).reshape(B, self.sampleNum_test, C, H, W)


    def aux_loss(self):
        """Return the aggregated loss over the auxiliary entropy bottleneck
        module(s).
        """
        aux_loss = sum(
            m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck)
        )
        return aux_loss