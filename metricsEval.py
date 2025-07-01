import os

os.environ['CUDA_VISIBLE_DEVICES'] = '7'
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import lpips
from torchmetrics.image.psnr import PeakSignalNoiseRatio
from loss.metrics._msssim import MultiscaleStructuralSimilarity
from DISTS_pytorch import DISTS
from tqdm import tqdm
import csv

from neuralcompression.metrics import (
    FrechetInceptionDistance,
    KernelInceptionDistance,
    update_patch_fid,
)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

mseLoss = torch.nn.MSELoss(reduction='none')

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0).to(device)

def get_image_paths(directory, semDimIdx, dataLength):
    return sorted(
        [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(('png', 'jpg', 'jpeg', 'bmp'))],
        key=lambda x: os.path.basename(x)  # sort the images
    )[((semDimIdx - 1) * dataLength):(semDimIdx * dataLength)]


def calculate_qualities(img_dir, semDimIdx, dataLength, path_idx):
    psnr_metric = PeakSignalNoiseRatio().to(device)
    msssim_metric = MultiscaleStructuralSimilarity(data_range=1.0, window_size=11).to(device)
    loss_fn_alex = lpips.LPIPS(net='alex').to(device)
    fid_metric = FrechetInceptionDistance().to(device)
    kid_metric = KernelInceptionDistance().to(device)
    D = DISTS().to(device)

    psnr_values = []
    msssim_values = []
    lpips_values = []
    dists_values = []

    image_paths = get_image_paths(img_dir, semDimIdx, dataLength)
    for img_path_idx in tqdm(range(len(image_paths))):
        img_path = image_paths[img_path_idx]
        imgs = load_image(img_path)
        B, C, H, W = imgs.shape
        W_ = (W - 20) // 2
        img1 = imgs[:, :, :, :W_]
        img2 = imgs[:, :, :, -W_:]
        psnr_values.append(psnr_metric(img1, img2).item())
        msssim_values.append(msssim_metric(img1, img2).item())
        lpips_values.append(loss_fn_alex(img1, img2, normalize=True).item())
        dists_values.append(D(img1, img2).item())

        if path_idx <= 3:
            img1 = img1.to(device)
            img2 = img2.to(device)
            update_patch_fid(
                img1, img2, fid_metric, kid_metric, patch_size=299
            )

    results = []
    results.append(np.mean(psnr_values))
    results.append(np.mean(msssim_values))
    results.append(np.mean(10 * np.log10(1 / (1 - np.array(msssim_values)))))
    results.append(np.mean(lpips_values))
    results.append(np.mean(dists_values))
    if path_idx <= 3:
        results.append(fid_metric.compute().item())
        results.append(kid_metric.compute()[0].item())
        results.append(kid_metric.compute()[1].item())
    return results


def evaluate_image_quality(img_dir, csvfileName, path_idx):
    dataLength_kodak = 24
    dataLength_div2k = 100
    dataLength_clic = 428

    if path_idx == 0 or path_idx == 1:
        dataLength = dataLength_clic
    elif path_idx == 2 or path_idx == 3:
        dataLength = dataLength_div2k
    else:
        dataLength = dataLength_kodak

    for semDimIdx in range(1, 17):
        results = calculate_qualities(img_dir, semDimIdx, dataLength, path_idx)

        print(f"\nResults for semDim {semDimIdx}:")
        print(f"Average PSNR: {results[0]:.4f}")
        print(f"Average MS-SSIM: {results[1]:.4f}({results[2]:.4f}dB)")
        print(f"Average LPIPS: {results[3]:.4f}")
        print(f"Average DISTS: {results[4]:.4f}")
        if path_idx <= 3:
            print(f"Average FID: {results[5]:.4f}")
            print(f"Average KID mean: {results[6]:.4f}")
            print(f"Average KID std: {results[7]:.4f}")

        with open(csvfileName, "a+", newline="") as file:
            writer = csv.writer(file)
            if path_idx <= 3:
                writer.writerow([semDimIdx * 32, results[0], results[1], results[2], results[3], results[4], results[5], results[6], results[7]])
            else:
                writer.writerow([semDimIdx * 32, results[0], results[1], results[2], results[3], results[4]])


if __name__ == '__main__':
    img_dir = "local/path/to/your/images"
    test_path = ["/root/path/to/your/project/"]

    path_idx = 0
    for path in test_path:
        word_dir = os.path.join(path, img_dir)
        csvfileName = os.path.join(path, "result.csv")
        with open(csvfileName, "a+", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["semDim", "PSNR", "MS-SSIM(value)", "MS-SSIM(dB)", "LPIPS", "DISTS", "FID", "KID mean", "KID std", "bpp"])

        evaluate_image_quality(word_dir, csvfileName, path_idx)

        path_idx = path_idx + 1
