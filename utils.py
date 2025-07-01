import numpy as np
import torch
import torchvision
import os
import logging


def workdir_configuration(filename, phase):
    workdir = './history/{}'.format(filename)
    if phase == 'test':
        workdir += '_test'
    samples = workdir + '/samples'
    models = workdir + '/models'
    percep_models = workdir + '/perceps'
    makedirs(workdir)
    makedirs(samples)
    makedirs(models)
    makedirs(percep_models)
    return workdir


def workdir_testConfiguration(filename, phase):
    workdir = './history/{}_test'.format(filename)
    samples = workdir + "/samples"
    makedirs(workdir)
    makedirs(samples)
    return workdir


def logger_configuration(workdir, filename, save_log=True):
    logger = logging.getLogger("Progressive Synonymous Image Compression (Progressive SIC)")
    log = workdir + '/Log_{}.log'.format(filename)
    formatter = logging.Formatter("%(asctime)s;%(levelname)s;%(message)s",
                                  "%Y-%m-%d %H:%M:%S")
    stdhandler = logging.StreamHandler()
    stdhandler.setLevel(logging.INFO)
    stdhandler.setFormatter(formatter)
    logger.addHandler(stdhandler)
    if save_log:
        filehandler = logging.FileHandler(log)
        filehandler.setLevel(logging.INFO)
        filehandler.setFormatter(formatter)
        logger.addHandler(filehandler)
    logger.setLevel(logging.INFO)
    return logger


def single_plot(epoch, global_step, real, gen, config):
    images = [real, gen]
    filename = "{}/SICModel_{}_epoch{}_step{}.png".format(config.samples, config.trainset, epoch, global_step)
    torchvision.utils.save_image(images, filename)


def makedirs(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def bpp_snr_to_kdivn(bpp, SNR):
    snr = 10 ** (SNR / 10)
    kdivn = bpp / 3 / np.log2(1 + snr)
    return kdivn


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def clear(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0


class MinMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.min = 1e6
        self.count = 0

    def update(self, val, idx=0, n=1):
        if idx != 0:
            if idx == 16:
                self.clear()
            self.val = val
            self.min = min(self.min, val)
            self.count += n

    def clear(self):
        self.val = 0
        self.min = 1e6
        self.count = 0


def BLN2BCHW(x, H, W):
    B, L, N = x.shape
    return x.reshape(B, H, W, N).permute(0, 3, 1, 2)


def BCHW2BLN(x):
    return x.flatten(2).permute(0, 2, 1)


def CalcuPSNR_int(img1, img2, max_val=255.):
    float_type = 'float64'
    img1 = np.round(torch.clamp(img1, 0, 1).detach().cpu().numpy() * 255)
    img2 = np.round(torch.clamp(img2, 0, 1).detach().cpu().numpy() * 255)

    img1 = img1.astype(float_type)
    img2 = img2.astype(float_type)
    mse = np.mean(np.square(img1 - img2), axis=(1, 2, 3))
    psnr = 20 * np.log10(max_val) - 10 * np.log10(mse)
    return psnr


def load_weights(net, model_path):
    pretrained = torch.load(model_path, map_location='cpu')     # ['state_dict']
    result_dict = {}
    for key, weight in pretrained.items():
        if 'module' in key:
            result_key = key[7:]
        else:
            result_key = key
        if 'attn_mask' not in key and 'rate_adaption.mask' not in key:
            result_dict[result_key] = weight
    print(net.load_state_dict(result_dict, strict=False))
    del result_dict, pretrained