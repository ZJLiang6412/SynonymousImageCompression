import time
from datetime import datetime
import sys
import argparse
import csv
sys.path.append("../")

from progSIC.net.model import ProgSIC
import torch.optim as optim
from utils import *
from savePic import *
from progSIC.data.dataset import get_loader_mGPU
from config import config
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

import torch._dynamo
torch._dynamo.config.verbose = True
torch._dynamo.config.suppress_errors = True

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
con = config()

os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '29600'
os.environ['CUDA_VISIBLE_DEVICES'] = str('0, 1, 2, 3, 4, 5, 6, 7')  # utilized GPU indexes

def main():
    settings = ['--phase', 'train',     # 'train' or 'test'
                "--checkpoint", "/path/to/ckpt/or/your/model/"]
    process(settings)

def semDimRolling(batch_idx):
    dist.barrier()
    semDim = con.semDim_level - (batch_idx % con.semDim_level)
    semDim = semDim * con.semDim_interval
    dist.barrier()
    return semDim


def train(world_size, dist, epoch, net, train_loader, optimizers, device, con_local, logLists):
    elapsed, losses, psnr_syntactics, lpips_syntactics, dists_syntactics, Epsnrs, Elpipss, Edistss, distSyns, distSams, bppys, bpps, bppyss, bppzs, bppy_stds = [AverageMeter() for _ in range(15)]
    bppy_chunks = MinMeter()
    metrics = [elapsed, losses, psnr_syntactics, lpips_syntactics, dists_syntactics, Epsnrs, Elpipss, Edistss, distSyns, distSams, bppys, bpps, bppyss, bppzs, bppy_stds, bppy_chunks]

    [optimizer, aux_optimizer] = optimizers

    for batch_idx, input_image in enumerate(train_loader):
        net.train()

        start_time = time.time()
        input_image = input_image.to(device)

        # training - semDim
        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        semDim = semDimRolling(batch_idx)

        mse_syntactic, lpips_syntactic, dists_syntactic, Emse_loss, Elpips_loss, Edists_loss, synonym_distance, sample_distance, bpp_y, bpp_ys, bpp_z, bpp_y_splits_std, bpp_y_splits_chunk = net.module.forward(input_image, semDim=semDim)

        dist.barrier()
        dist.all_reduce(mse_syntactic, op=dist.ReduceOp.SUM)
        dist.all_reduce(lpips_syntactic, op=dist.ReduceOp.SUM)
        dist.all_reduce(dists_syntactic, op=dist.ReduceOp.SUM)
        dist.all_reduce(Emse_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(Elpips_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(Edists_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(synonym_distance, op=dist.ReduceOp.SUM)
        dist.all_reduce(sample_distance, op=dist.ReduceOp.SUM)
        dist.all_reduce(bpp_y, op=dist.ReduceOp.SUM)
        dist.all_reduce(bpp_ys, op=dist.ReduceOp.SUM)
        dist.all_reduce(bpp_z, op=dist.ReduceOp.SUM)
        dist.all_reduce(bpp_y_splits_std, op=dist.ReduceOp.SUM)
        dist.all_reduce(bpp_y_splits_chunk, op=dist.ReduceOp.SUM)
        dist.barrier()
        mse_syntactic = mse_syntactic / world_size
        lpips_syntactic = lpips_syntactic / world_size
        dists_syntactic = dists_syntactic / world_size
        Emse_loss = Emse_loss / world_size
        Elpips_loss = Elpips_loss / world_size
        Edists_loss = Edists_loss / world_size
        synonym_distance = synonym_distance / world_size
        sample_distance = sample_distance / world_size
        bpp_y = bpp_y / world_size
        bpp_ys = bpp_ys / world_size
        bpp_z = bpp_z / world_size
        bpp_y_splits_std = bpp_y_splits_std / world_size
        bpp_y_splits_chunk = bpp_y_splits_chunk / world_size

        loss_syntactic = con_local.Emse_lambda[-1] * mse_syntactic + con_local.lpips_gamma[-1] * lpips_syntactic + con_local.bpp_gamma[-1] * (bpp_y + bpp_z)
        loss_synonymous = con_local.Emse_lambda[semDim // con.semDim_interval - 1] * Emse_loss + con_local.lpips_gamma[semDim // con.semDim_interval - 1] * Elpips_loss + con_local.bpp_gamma[semDim // con.semDim_interval - 1] * (bpp_ys + bpp_z)
        if epoch < 2:
            loss_constraint = synonym_distance + sample_distance + 64 * bpp_y_splits_std - 4 * bpp_y_splits_chunk
        else:
            loss_constraint = synonym_distance + sample_distance
        loss = (0.5 * loss_syntactic + 0.5 * loss_synonymous) + loss_constraint

        loss.backward()
        for [name, param] in net.named_parameters():
            if param.grad != None:
                dist.all_reduce(param.grad, op=torch.distributed.ReduceOp.SUM)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
        optimizer.step()

        aux_loss = net.module.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        elapsed.update(time.time() - start_time)
        losses.update(loss.item())
        PSNR = 10 * (torch.log(1. * 1. / mse_syntactic) / np.log(10))
        psnr_syntactics.update(PSNR.item())
        lpips_syntactics.update(lpips_syntactic.item())
        dists_syntactics.update(dists_syntactic.item())
        Epsnr = 10 * (torch.log(1. * 1. / Emse_loss) / np.log(10))
        Epsnrs.update(Epsnr)
        Elpipss.update(Elpips_loss.item())
        Edistss.update(Edists_loss.item())
        bppys.update(bpp_y.item())
        bpps.update(bpp_ys.item() + bpp_z.item())
        bppyss.update(bpp_ys.item())
        bppzs.update(bpp_z.item())
        distSyns.update(synonym_distance.item())
        distSams.update(sample_distance.item())
        bppy_stds.update(bpp_y_splits_std.item())
        bppy_chunks.update(bpp_y_splits_chunk.item(), idx=semDim//con.semDim_interval)

        # statistics
        if (((batch_idx + 1) % (con_local.print_step)) == 0):
            process = ((batch_idx + 1) % train_loader.__len__()) / (train_loader.__len__()) * 100.0
            with open(logLists[0], 'a+', newline="") as file:
                writer = csv.writer(file)
                writer.writerow(
                    ['train', '{:0>3d}'.format(epoch), '{:.2f}%'.format(process), '{:.2f}'.format(elapsed.avg),
                     '{:.4f}'.format(losses.val), '{:.4f}'.format(losses.avg),
                     '{:.4f}'.format(psnr_syntactics.val), '{:.4f}'.format(psnr_syntactics.avg),
                     '{:.4f}'.format(lpips_syntactics.val), '{:.4f}'.format(lpips_syntactics.avg),
                     '{:.4f}'.format(dists_syntactics.val), '{:.4f}'.format(dists_syntactics.avg),
                     '{:.4f}'.format(Epsnrs.val), '{:.4f}'.format(Epsnrs.avg),
                     '{:.4f}'.format(Elpipss.val), '{:.4f}'.format(Elpipss.avg),
                     '{:.4f}'.format(Edistss.val), '{:.4f}'.format(Edistss.avg),
                     '{:.4f}'.format(bppys.val), '{:.4f}'.format(bppys.avg),
                     '{:.4f}'.format(bpps.val), '{:.4f}'.format(bpps.avg),
                     '{:.4f}'.format(bppyss.val), '{:.4f}'.format(bppyss.avg),
                     '{:.4f}'.format(bppzs.val), '{:.4f}'.format(bppzs.avg),
                     '{:.4f}'.format(distSyns.val), '{:.4f}'.format(distSyns.avg),
                     '{:.4f}'.format(distSams.val), '{:.4f}'.format(distSams.avg),
                     '{:.4f}'.format(bppy_stds.val), '{:.4f}'.format(bppy_stds.avg),
                     '{:.4f}'.format(bppy_chunks.val), '{:.4f}'.format(bppy_chunks.min)])
            log = (' | '.join([
                f'Epoch {epoch}',
                f'Step [{(batch_idx + 1) % train_loader.__len__()}/{train_loader.__len__()}={process:.2f}%]',
                f'Time {elapsed.avg:.2f}',
                f'Loss {losses.val:.3f} ({losses.avg:.3f})',
                f'PSNRsyn {psnr_syntactics.val:.2f} ({psnr_syntactics.avg:.2f})',
                f'LPIPSsyn {lpips_syntactics.val:.2f} ({lpips_syntactics.avg:.2f})',
                f'DISTSsyn {dists_syntactics.val:.2f} ({dists_syntactics.avg:.2f})',
                f'EPSNR {Epsnrs.val:.2f} ({Epsnrs.avg:.2f})',
                f'ELPIPS {Elpipss.val:.2f} ({Elpipss.avg:.2f})',
                f'EDISTS {Edistss.val:.2f} ({Edistss.avg:.2f})',
                f'Bpp_y {bppys.val:.4f} ({bppys.avg:.4f})',
                f'Bpp_ys {bppyss.val:.4f} ({bppyss.avg:.4f})',
                f'Bpp_z {bppzs.val:.4f} ({bppzs.avg:.4f})',
                f'DistSym {distSyns.val:.4f} ({distSyns.avg:.4f})',
                f'DistSam {distSams.val:.4f} ({distSams.avg:.4f})',
                f'Bppy_stds {bppy_stds.val:.4f} ({bppy_stds.avg:.4f})',
                f'Bppy_chunks {bppy_chunks.val:.4f} ({bppy_chunks.min:.4f})',
            ]))
            con_local.logger.info(log)

            for i in metrics:
                i.clear()

        dist.barrier()


def test(world_size, dist, net, test_loader, device, con_local, epoch, semDim, logCSV):
    with torch.no_grad():
        net.eval()
        elapsed, losses, Epsnrs, Elpipss, Edistss, distSyns, distSams, msssims, msssim_dbs, bppys, bpps, bppyss, bppzs, bppy_stds = [AverageMeter() for _ in range(14)]
        bppy_chunks = MinMeter()

        for batch_idx, input_image in enumerate(test_loader):
            start_time = time.time()
            input_image = input_image.to(device)
            B, C, H, W = input_image.shape
            if (H % 64 == 0) and (W % 64 == 0):
                input_image_resize = input_image
            elif (H % 64 == 0) and (W % 64 != 0):
                input_image_resize = torch.zeros(B, C, H, (W // 64 + 1) * 64).to(device)
            elif (H % 64 != 0) and (W % 64 == 0):
                input_image_resize = torch.zeros(B, C, (H // 64 + 1) * 64, W).to(device)
            else:
                input_image_resize = torch.zeros(B, C, (H // 64 + 1) * 64, (W // 64 + 1) * 64).to(device)
            input_image_resize[:, :, :H, :W] = input_image

            Emse_loss, Elpips_loss, Edists_loss, synonym_distance, sample_distance, bpp_y, bpp_ys, bpp_z, bpp_y_splits_std, bpp_y_splits_chunk, msssim, x_hat = net.module.test(input_image_resize, semDim=semDim)

            dist.barrier()
            dist.all_reduce(Emse_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(Elpips_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(Edists_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(synonym_distance, op=dist.ReduceOp.SUM)
            dist.all_reduce(sample_distance, op=dist.ReduceOp.SUM)
            dist.all_reduce(bpp_y, op=dist.ReduceOp.SUM)
            dist.all_reduce(bpp_ys, op=dist.ReduceOp.SUM)
            dist.all_reduce(bpp_z, op=dist.ReduceOp.SUM)
            dist.all_reduce(bpp_y_splits_std, op=dist.ReduceOp.SUM)
            dist.all_reduce(bpp_y_splits_chunk, op=dist.ReduceOp.SUM)
            dist.barrier()
            Emse_loss = Emse_loss / world_size
            Elpips_loss = Elpips_loss / world_size
            Edists_loss = Edists_loss / world_size
            synonym_distance = synonym_distance / world_size
            sample_distance = sample_distance / world_size
            bpp_y = bpp_y / world_size
            bpp_ys = bpp_ys / world_size
            bpp_z = bpp_z / world_size
            bpp_y_splits_std = bpp_y_splits_std / world_size
            bpp_y_splits_chunk = bpp_y_splits_chunk / world_size
            elapsed.update(time.time() - start_time)

            loss_synonymous = con_local.Emse_lambda[semDim // con.semDim_interval - 1] * Emse_loss + con_local.lpips_gamma[semDim // con.semDim_interval - 1] * Elpips_loss + con_local.bpp_gamma[semDim // con.semDim_interval - 1] * (bpp_ys + bpp_z)
            if epoch < 2:
                loss_constraint = synonym_distance + sample_distance + 64 * bpp_y_splits_std - 4 * bpp_y_splits_chunk
            else:
                loss_constraint = synonym_distance + sample_distance
            loss = loss_synonymous + loss_constraint

            losses.update(loss.item())
            Epsnr = 10 * (torch.log(1. * 1. / Emse_loss) / np.log(10))
            Epsnrs.update(Epsnr)
            msssims.update(msssim)
            msssim_db = 10 * (torch.log(1. / msssim) / np.log(10))
            msssim_dbs.update(msssim_db.item())
            Elpipss.update(Elpips_loss.item())
            Edistss.update(Edists_loss.item())
            bppys.update(bpp_y.item())
            bpps.update(bpp_ys.item() + bpp_z.item())
            bppyss.update(bpp_ys.item())
            bppzs.update(bpp_z.item())
            distSyns.update(synonym_distance.item())
            distSams.update(sample_distance.item())
            bppy_stds.update(bpp_y_splits_std.item())
            bppy_chunks.update(bpp_y_splits_chunk.item(), idx=semDim//con.semDim_interval)

            if batch_idx < 1:
                savePic_samples(input_image, x_hat[:, :, :, :H, :W],
                        con_local.workdir + "/samples/epoch%05d_semDim%03d_batch%02d_rank%d_" % (epoch + 1, semDim, batch_idx, dist.get_rank()))

            if ((batch_idx + 1) % test_loader.__len__()) == 0:
                process = ((batch_idx + 1) % test_loader.__len__()) / (test_loader.__len__()) * 100.0
                with open(logCSV, 'a+', newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(
                        ['test', '{:0>3d}'.format(epoch), '{:.2f}%'.format(process), '{:.2f}'.format(elapsed.avg),
                         '{:.4f}'.format(losses.val), '{:.4f}'.format(losses.avg),
                         '{:.4f}'.format(Epsnrs.val), '{:.4f}'.format(Epsnrs.avg),
                         '{:.4f}'.format(Elpipss.val), '{:.4f}'.format(Elpipss.avg),
                         '{:.4f}'.format(Edistss.val), '{:.4f}'.format(Edistss.avg),
                         '{:.4f}'.format(msssims.val), '{:.4f}'.format(msssims.avg),
                         '{:.4f}'.format(msssim_dbs.val), '{:.4f}'.format(msssim_dbs.avg),
                         '{:.4f}'.format(bppys.val), '{:.4f}'.format(bppys.avg),
                         '{:.4f}'.format(bpps.val), '{:.4f}'.format(bpps.avg),
                         '{:.4f}'.format(bppyss.val), '{:.4f}'.format(bppyss.avg),
                         '{:.4f}'.format(bppzs.val), '{:.4f}'.format(bppzs.avg),
                         '{:.4f}'.format(distSyns.val), '{:.4f}'.format(distSyns.avg),
                         '{:.4f}'.format(distSams.val), '{:.4f}'.format(distSams.avg),
                         '{:.4f}'.format(bppy_stds.val), '{:.4f}'.format(bppy_stds.avg),
                         '{:.4f}'.format(bppy_chunks.val), '{:.4f}'.format(bppy_chunks.min)])
                log = (' | '.join([
                    f'Epoch {epoch}',
                    f'Step [{(batch_idx + 1) % test_loader.__len__()}/{test_loader.__len__()}={process:.2f}%]',
                    f'Time {elapsed.avg:.2f}',
                    f'Loss1 {losses.val:.3f} ({losses.avg:.3f})',
                    f'EPSNR {Epsnrs.val:.2f} ({Epsnrs.avg:.2f})',
                    f'ELPIPS {Elpipss.val:.2f} ({Elpipss.avg:.2f})',
                    f'EDISTS {Edistss.val:.2f} ({Edistss.avg:.2f})',
                    f'MS_SSIM {msssims.val:.2f} ({msssims.avg:.2f})',
                    f'MS_SSIMdB {msssim_dbs.val:.2f} ({msssim_dbs.avg:.2f})',
                    f'Bpp_y {bppys.val:.4f} ({bppys.avg:.4f})',
                    f'Bpp_ys {bppyss.val:.4f} ({bppyss.avg:.4f})',
                    f'Bpp_z {bppzs.val:.4f} ({bppzs.avg:.4f})',
                    f'DistSym {distSyns.val:.4f} ({distSyns.avg:.4f})',
                    f'DistSym {distSams.val:.4f} ({distSams.avg:.4f})',
                    f'Bppy_stds {bppy_stds.val:.4f} ({bppy_stds.avg:.4f})',
                    f'Bppy_chunks {bppy_chunks.val:.4f} ({bppy_chunks.min:.4f})',
                ]))
                con_local.logger.info(log)

                dist.barrier()

    con_local.logger.info(f'Finish test! SemDim={semDim}, Average Loss = {losses.avg:.4f}, EPSNR={Epsnrs.avg:.4f}, ELPIPS={Elpipss.avg:.4f}, EDISTS={Edistss.avg:.4f}, MSSSIM={msssims.avg:.4f}, MSSSIMdB={msssim_dbs.avg:.4f}, bpp={bpps.avg:.4f}, DistSyms={distSyns.avg:.4f}, DistSams={distSams.avg:.4f}, Bppy_stds={bppy_stds.avg:.4f}, Bppy_chunks={bppy_chunks.min:.4f}')
    return losses.avg, Epsnrs.avg, Elpipss.avg, Edistss.avg, msssim_dbs.avg, bpps.avg, distSyns.avg, distSams.avg


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Progressive Synonymous Image Compression (Progressive SIC).")
    parser.add_argument(
        "-p",
        "--phase",
        default='test',  # train
        type=str,
        help="Train or Test",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=5000,
        type=int,
        help="Number of epochs (default: %(default)s)"
    )
    parser.add_argument("--cuda", default=True, action="store_true", help="Use cuda")
    parser.add_argument(
        "--gpu-id",
        type=str,
        default=0,
        help="GPU ids (default: %(default)s)",
    )
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument(
        "--seed", type=float, default=3407, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        '--name',
        default="ProgSIC_" + datetime.now().strftime('%Y-%m-%d_%H_%M_%S'),
        type=str,
        help='Result dir name',
    )
    parser.add_argument(
        '--save_log', action='store_true', default=True, help='Save log to disk'
    )
    parser.add_argument("--checkpoint",
                        default=None,
                        type=str, help="Path to a checkpoint")
    args = parser.parse_args(argv)
    return args


def mGPU_process(rank, world_size, args, con, workdir, semLog_lists, semLog_test):
    dist.init_process_group("nccl", init_method='env://', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        # random.seed((args.seed + dist.get_rank() * 1024) % 10000)
        # np.random.seed((args.seed + dist.get_rank() * 1024) % 10000)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger = logger_configuration(workdir, args.name, save_log=True)
    con.logger = logger
    logger.info(con.__dict__)

    net = ProgSIC(con)
    model_path = args.checkpoint
    if model_path != None:
        load_weights(net, model_path)
    device = torch.device(f'cuda:{rank}')
    model = net.to(device)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    params = set(p for n, p in net.named_parameters() if (not n.endswith(".quantiles") and ("lpips" not in n)))
    optimizer = optim.AdamW(params, lr=con.lr, weight_decay=5e-5)
    aux_params = set(p for n, p in net.named_parameters() if n.endswith(".quantiles"))
    aux_optimizer = optim.AdamW(aux_params, lr=con.aux_lr, weight_decay=5e-5)

    if args.phase == 'test':
        _, test_loader = get_loader_mGPU(config, rank, world_size)
        for semDim in range(con.semDim_interval, con.dim + 1, con.semDim_interval):
            test(world_size, dist, model, test_loader, device, con, epoch=-1, semDim=semDim, logCSV=semLog_test[semDim // con.semDim_interval - 1])
    elif args.phase == 'train':
        train_loader, test_loader = get_loader_mGPU(config, rank, world_size)

        best_loss = float("inf")

        steps_epoch = 160
        tot_epoch = 5000
        for epoch in range(steps_epoch, tot_epoch):
            logger.info('======Current epoch %s ======' % epoch)
            logger.info(f"Learning rate: {optimizer.param_groups[0]['lr']}")
            train(world_size, dist, epoch, model, train_loader, [optimizer, aux_optimizer], device, con, logLists=semLog_lists)

            test_losses, test_Epsnrs, test_Elpipss, test_Edistss, test_msssim_dbs, test_bpps, test_distSyns, test_distSams = [], [], [], [], [], [], [], []
            for semDim in range(con.semDim_interval, con.dim + 1, con.semDim_interval):
                test_loss, test_Epsnr, test_Elpips, test_Edists, test_msssim_db, test_bpp, test_distSyn, test_distSam = test(
                    world_size, dist, model, test_loader, device, con, epoch, semDim=semDim, logCSV=semLog_test[semDim // con.semDim_interval - 1])
                test_losses.append(test_loss)
                test_Epsnrs.append(test_Epsnr)
                test_msssim_dbs.append(test_msssim_db)
                test_Elpipss.append(test_Elpips)
                test_Edistss.append(test_Edists)
                test_bpps.append(test_bpp)
                test_distSyns.append(test_distSyn)
                test_distSams.append(test_distSam)
            test_losses = torch.mean(torch.tensor(test_losses)).item()
            test_Epsnrs = torch.mean(torch.tensor(test_Epsnrs)).item()
            test_msssim_dbs = torch.mean(torch.tensor(test_msssim_dbs)).item()
            test_Elpipss = torch.mean(torch.tensor(test_Elpipss)).item()
            test_Edistss = torch.mean(torch.tensor(test_Edistss)).item()
            test_bpps = torch.mean(torch.tensor(test_bpps)).item()
            test_distSyns = torch.mean(torch.tensor(test_distSyns)).item()
            test_distSams = torch.mean(torch.tensor(test_distSams)).item()

            if epoch >= 2:
                is_best = (test_losses < best_loss)
                best_loss = min(test_losses, best_loss)
            else:
                is_best = True

            if is_best:
                if dist.get_rank() == 0:
                    save_model(model,
                               save_path=workdir + '/models/EP{}_rank_{}_loss_{}_Epsnr_{}_Elpips_{}_Edists_{}_msssim_{}_bpp_{}_distSyns_{}_distSams_{}.model'.format(
                                   "%02d" % (epoch + 1), rank, "%.4f" % test_losses,
                                   "%.3f" % test_Epsnrs, "%.3f" % test_Elpipss, "%.3f" % test_Edistss, "%.3f" % test_msssim_dbs,
                                   "%.3f" % test_bpps, "%.4f" % test_distSyns, "%.4f" % test_distSams))

            if (epoch + 1) % 100 == 0:
                if dist.get_rank() == 0:
                    save_model(model, save_path=workdir + '/models/EP{}.model'.format(epoch + 1))

            dist.barrier()

    dist.destroy_process_group()


def process(argv):
    args = parse_args(argv)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    con.device = device

    workdir = workdir_configuration(args.name, phase=args.phase)
    con.workdir = workdir

    semLog_lists = []
    logName = os.path.join(workdir, 'Log_semDimTrain.csv')
    semLog_lists.append(logName)
    with open(logName, "a+", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(['mode', 'epoch', 'percent', 'time', 'loss(item)', 'loss(average)', 'psnr_syn(item)', 'psnr_syn(average)', 'lpips_syn(item)', 'lpips_syn(average)', 'dists_syn(item)', 'dists_syn(average)',
                         'Epsnrs(item)', 'Epsnrs(average)', 'Elpips(item)', 'Elpips(average)', 'Edists(item)', 'Edists(average)',
                         'bppy_syn(item)', 'bppy_syn(average)', 'bpp(item)', 'bpp(average)', 'bppys(item)', 'bppys(average)', 'bppz(item)', 'bppz(average)',
                         'distSyn(item)', 'distSyn(average)', 'distSam(item)', 'distSam(average)', 'bppy_stds(item)', 'bppy_stds(average)', 'bppy_mins(item)', 'bppy_mins(average)'])
    semLog_test = []
    for i in range(1, con.semDim_level + 1):
        logName = os.path.join(workdir, 'Log_semDimTest_%03d.csv' % (i * con.semDim_interval))
        semLog_test.append(logName)
        with open(logName, "a+", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(['mode', 'epoch', 'percent', 'time', 'loss(item)', 'loss(average)',
                         'Epsnrs(item)', 'Epsnrs(average)', 'Elpips(item)', 'Elpips(average)', 'Edists(item)', 'Edists(average)',
                         'msssim(item)', 'msssim(average)', 'msssim_dB(item)', 'msssim_dB(average)',
                         'bppy_syn(item)', 'bppy_syn(average)', 'bpp(item)', 'bpp(average)', 'bppys(item)', 'bppys(average)', 'bppz(item)', 'bppz(average)',
                         'distSyn(item)', 'distSyn(average)', 'distSam(item)', 'distSam(average)', 'bppy_stds(item)', 'bppy_stds(average)', 'bppy_mins(item)', 'bppy_mins(average)'])


    world_size = torch.cuda.device_count()
    mp.spawn(mGPU_process, args=(world_size, args, con, workdir, semLog_lists, semLog_test), nprocs=world_size, join=True)




if __name__ == '__main__':
    main()


