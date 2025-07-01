import ssl
ssl._create_default_https_context = ssl._create_unverified_context

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
from progSIC.data.dataset_warming import get_loader_mGPU, get_test_loader
from config_warming import config
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
os.environ['CUDA_VISIBLE_DEVICES'] = str('0')


def main():
    settings = ['--phase', 'train'] #, "--checkpoint", "/media/D/liangzijian/MPSIC/PIMISIC_v512/pretrained/initialModel.model"]
    process(settings)


def train(world_size, dist, epoch, net, train_loader, optimizers, device, con_local, logLists):
    elapsed, losses, psnr_syntactics, lpips_syntactics, bppy_stds, bppy_mins, bppys, bppzs = [AverageMeter() for _ in range(8)]
    metrics = [elapsed, losses, psnr_syntactics, lpips_syntactics, bppy_stds, bppy_mins, bppys, bppzs]

    [optimizer, aux_optimizer] = optimizers

    for batch_idx, input_image in enumerate(train_loader):
        net.train()

        start_time = time.time()
        input_image = input_image.to(device)

        # training - semDim
        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        mse_syntactic, lpips_syntactic, bpp_y_splits_std, bpp_y_splits_min, bpp_y, bpp_z, _ = net.module.warming(input_image)

        loss = con_local.Emse_lambda[-1] * mse_syntactic + con_local.lpips_gamma[-1] * lpips_syntactic + (bpp_y_splits_std - bpp_y_splits_min) + (bpp_y + bpp_z)

        loss.backward()
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
        bppys.update(bpp_y.item())
        bppzs.update(bpp_z.item())
        bppy_stds.update(bpp_y_splits_std.item())
        bppy_mins.update(bpp_y_splits_min.item())

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
                     '{:.4f}'.format(bppy_stds.val), '{:.4f}'.format(bppy_stds.avg),
                     '{:.4f}'.format(bppy_mins.val), '{:.4f}'.format(bppy_mins.avg),
                     '{:.4f}'.format(bppys.val), '{:.4f}'.format(bppys.avg),
                     '{:.4f}'.format(bppzs.val), '{:.4f}'.format(bppzs.avg),
                     ])
            log = (' | '.join([
                f'Epoch {epoch}',
                f'Step [{(batch_idx + 1) % train_loader.__len__()}/{train_loader.__len__()}={process:.2f}%]',
                f'Time {elapsed.avg:.2f}',
                f'Loss {losses.val:.3f} ({losses.avg:.3f})',
                f'PSNRsyn {psnr_syntactics.val:.2f} ({psnr_syntactics.avg:.2f})',
                f'LPIPSsyn {lpips_syntactics.val:.2f} ({lpips_syntactics.avg:.2f})',
                f'bppy_stds {bppy_stds.val:.2f} ({bppy_stds.avg:.2f})',
                f'bppy_mins {bppy_mins.val:.2f} ({bppy_mins.avg:.2f})',
                f'Bpp_y {bppys.val:.4f} ({bppys.avg:.4f})',
                f'Bpp_z {bppzs.val:.4f} ({bppzs.avg:.4f})',
            ]))
            con_local.logger.info(log)

            for i in metrics:
                i.clear()

        dist.barrier()


def test(world_size, dist, net, test_loader, device, con_local, epoch, semDim, logCSV):
    with torch.no_grad():
        net.eval()
        elapsed, losses, psnr_syntactics, lpips_syntactics, bppy_stds, bppy_mins, bppys, bppzs = [AverageMeter() for _ in range(8)]
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

            mse_syntactic, lpips_syntactic, bpp_y_splits_std, bpp_y_splits_min, bpp_y, bpp_z, x_hat = net.module.warming(input_image_resize)

            loss = con_local.Emse_lambda[-1] * mse_syntactic + con_local.lpips_gamma[-1] * lpips_syntactic + (bpp_y_splits_std - bpp_y_splits_min)

            elapsed.update(time.time() - start_time)
            losses.update(loss.item())
            PSNR = 10 * (torch.log(1. * 1. / mse_syntactic) / np.log(10))
            psnr_syntactics.update(PSNR.item())
            lpips_syntactics.update(lpips_syntactic.item())
            bppys.update(bpp_y.item())
            bppzs.update(bpp_z.item())
            bppy_stds.update(bpp_y_splits_std.item())
            bppy_mins.update(bpp_y_splits_min.item())

            if batch_idx < 1:
                savePic(input_image, x_hat[:, :, :, :H, :W],
                                con_local.workdir + "/samples/epoch%05d_semDim%03d_batch%02d_rank%d_" % (
                                epoch + 1, semDim, batch_idx, dist.get_rank()))

            if ((batch_idx + 1) % test_loader.__len__()) == 0:
                process = ((batch_idx + 1) % test_loader.__len__()) / (test_loader.__len__()) * 100.0
                with open(logCSV, 'a+', newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(
                        ['test', '{:0>3d}'.format(epoch), '{:.2f}%'.format(process), '{:.2f}'.format(elapsed.avg),
                         '{:.4f}'.format(losses.val), '{:.4f}'.format(losses.avg),
                         '{:.4f}'.format(psnr_syntactics.val), '{:.4f}'.format(psnr_syntactics.avg),
                         '{:.4f}'.format(lpips_syntactics.val), '{:.4f}'.format(lpips_syntactics.avg),
                         '{:.4f}'.format(bppy_stds.val), '{:.4f}'.format(bppy_stds.avg),
                     '{:.4f}'.format(bppy_mins.val), '{:.4f}'.format(bppy_mins.avg),
                     '{:.4f}'.format(bppys.val), '{:.4f}'.format(bppys.avg),
                     '{:.4f}'.format(bppzs.val), '{:.4f}'.format(bppzs.avg)])
                log = (' | '.join([
                    f'Epoch {epoch}',
                    f'Step [{(batch_idx + 1) % test_loader.__len__()}/{test_loader.__len__()}={process:.2f}%]',
                    f'Time {elapsed.avg:.2f}',
                    f'Loss1 {losses.val:.3f} ({losses.avg:.3f})',
                    f'PSNRsyn {psnr_syntactics.val:.2f} ({psnr_syntactics.avg:.2f})',
                    f'LPIPSsyn {lpips_syntactics.val:.2f} ({lpips_syntactics.avg:.2f})',
                    f'PSNRsyn {psnr_syntactics.val:.2f} ({psnr_syntactics.avg:.2f})',
                    f'LPIPSsyn {lpips_syntactics.val:.2f} ({lpips_syntactics.avg:.2f})',
                    f'bppy_stds {bppy_stds.val:.2f} ({bppy_stds.avg:.2f})',
                    f'bppy_mins {bppy_mins.val:.2f} ({bppy_mins.avg:.2f})',
                    f'Bpp_y {bppys.val:.4f} ({bppys.avg:.4f})',
                ]))
                con_local.logger.info(log)

                dist.barrier()

    con_local.logger.info(
        f'Finish test! SemDim={semDim}, Average Loss = {losses.avg:.4f}, PSNRsyn={psnr_syntactics.avg:.4f}dB, LPIPSsyn={lpips_syntactics.avg:.4f}, bppy_std={bppy_stds.avg:.4f}, bppy_min={bppy_mins.avg:.4f}, bpp_y={bppys.avg:.4f}, bpp_z={bppzs.avg}')
    return losses.avg, psnr_syntactics.avg, lpips_syntactics.avg, bppy_stds.avg, bppy_mins.avg, bppys.avg, bppzs.avg


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Progressive Synonymous Image Compression (Progressive SIC) - warming.")
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
        default="warming_" + datetime.now().strftime('%Y-%m-%d_%H_%M_%S'),
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger = logger_configuration(workdir, args.name, save_log=True)
    con.logger = logger
    logger.info(con.__dict__)

    net = ProgSIC(con)
    model_path = args.checkpoint
    if model_path != None:
        load_weights(net, model_path)
    else:
        load_weights(net, workdir + '/models/initialize.model')
    device = torch.device(f'cuda:{rank}')
    model = net.to(device)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    params = set(p for n, p in net.named_parameters() if (not n.endswith(".quantiles") and ("lpips" not in n)))
    optimizer = optim.Adam(params, lr=con.lr)
    aux_params = set(p for n, p in net.named_parameters() if n.endswith(".quantiles"))
    aux_optimizer = optim.Adam(aux_params, lr=con.aux_lr)

    if args.phase == 'test':
        test_loader = get_test_loader(config)
        for semDim in range(con.dim, con.dim + 1, con.semDim_interval):
            test(world_size, dist, model, test_loader, device, con, epoch=-1, semDim=semDim,
                 logCSV=semLog_test[0])
    elif args.phase == 'train':
        train_loader, test_loader = get_loader_mGPU(config, rank, world_size)

        best_loss = float("inf")

        steps_epoch = 0
        tot_epoch = 1
        for epoch in range(steps_epoch, tot_epoch):
            logger.info('======Current epoch %s ======' % epoch)
            logger.info(f"Learning rate: {optimizer.param_groups[0]['lr']}")
            train(world_size, dist, epoch, model, train_loader, [optimizer, aux_optimizer], device, con,
                  logLists=semLog_lists)

            test_losses, test_psnr_syntactics, test_lpips_syntactics, test_bppy_stds, test_bppy_mins, test_bppys, test_bppzs = [], [], [], [], [], [], []
            for semDim in range(con.dim, con.dim + 1, con.semDim_interval):
                test_loss, test_psnr_syntactic, test_lpips_syntactic, test_bppy_std, test_bppy_min, test_bppy, test_bppz = test(
                    world_size, dist, model, test_loader, device, con, epoch, semDim=semDim,
                    logCSV=semLog_test[0])
                test_losses.append(test_loss)
                test_psnr_syntactics.append(test_psnr_syntactic)
                test_lpips_syntactics.append(test_lpips_syntactic)
                test_bppy_stds.append(test_bppy_std)
                test_bppy_mins.append(test_bppy_min)
                test_bppys.append(test_bppy)
                test_bppzs.append(test_bppz)
            test_losses = torch.mean(torch.tensor(test_losses)).item()
            test_psnr_syntactics = torch.mean(torch.tensor(test_psnr_syntactics)).item()
            test_lpips_syntactics = torch.mean(torch.tensor(test_lpips_syntactics)).item()
            test_bppy_stds = torch.mean(torch.tensor(test_bppy_stds)).item()
            test_bppy_mins = torch.mean(torch.tensor(test_bppy_mins)).item()
            test_bppys = torch.mean(torch.tensor(test_bppys)).item()
            test_bppzs = torch.mean(torch.tensor(test_bppzs)).item()
            is_best = (test_losses < best_loss)
            best_loss = min(test_losses, best_loss)

            if is_best:
                if dist.get_rank() == 0:
                    save_model(model,
                               save_path=workdir + '/models/warming_EP{}_rank_{}_loss_{}_PSNRsyn_{}_LPIPSsyn_{}_bppy_std_{}_bppy_min_{}_bppy_{}_bppz_{}.model'.format(
                                   "%02d" % (epoch + 1), rank, "%.4f" % test_losses, "%.3f" % test_psnr_syntactics,
                                   "%.3f" % test_lpips_syntactics, "%.4f" % test_bppy_stds,
                                   "%.4f" % test_bppy_mins, "%.4f" % test_bppys, "%.4f" % test_bppzs))

            if (epoch + 1) % 100 == 0:
                if dist.get_rank() == 0:
                    save_model(model, save_path=workdir + '/models/Pretrained_EP{}.model'.format(epoch + 1))

            dist.barrier()

    dist.destroy_process_group()


def process(argv):
    args = parse_args(argv)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    con.device = device

    workdir = workdir_configuration(args.name, phase=args.phase)
    con.workdir = workdir

    net = ProgSIC(con)
    save_model(net, save_path=workdir + '/models/initialize.model')

    semLog_lists = []
    logName = os.path.join(workdir, 'Log_semDimTrain.csv')
    semLog_lists.append(logName)
    with open(logName, "a+", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ['mode', 'epoch', 'percent', 'time', 'loss(item)', 'loss(average)', 'psnr_syn(item)', 'psnr_syn(average)',
             'lpips_syn(item)', 'lpips_syn(average)',
             'bppy_std(item)', 'bppy_std(average)', 'bppy_min(item)', 'bppy_min(average)', 'bppys(item)', 'bppys(average)',
             'bppz(item)', 'bppz(average)'])
    semLog_test = []
    for i in range(16, 17):
        logName = os.path.join(workdir, 'Log_semDimTest_%03d.csv' % (i * con.semDim_interval))
        semLog_test.append(logName)
        with open(logName, "a+", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(['mode', 'epoch', 'percent', 'time', 'loss(item)', 'loss(average)', 'psnr_syn(item)',
                             'psnr_syn(average)', 'lpips_syn(item)', 'lpips_syn(average)',
                             'bppy_std(item)', 'bppy_std(average)', 'bppy_min(item)', 'bppy_min(average)', 'bppys(item)', 'bppys(average)',
                             'bppz(item)', 'bppz(average)'])

    world_size = torch.cuda.device_count()
    mp.spawn(mGPU_process, args=(world_size, args, con, workdir, semLog_lists, semLog_test), nprocs=world_size,
             join=True)


if __name__ == '__main__':
    main()


