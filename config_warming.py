import torch
import datetime
import torch.nn as nn
import math


class config:
    train_data_dir = ['/root/path/to/your/train/set/']
    test_data_dir = ['/root/path/to/your/test/set/']
    batch_size = 8
    num_workers = 8

    sampleNum = 1

    print_step = 48
    plot_step = 1000
    logger = None

    # training details
    image_dims = (3, 256, 256)
    lr = 1e-4
    aux_lr = 1e-3
    distortion_metric = 'MSE'  # 'MS-SSIM'

    dim = 512
    semDim_level = 16
    semDim_interval = dim // semDim_level

    # The following hyperparameters are empirically set based on $semDim_level=16$.
    mse_lambda = 1024
    Emse_lambda = [0, 128, 256, 384, 512, 640, 768, 896, 1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920, 2048]
    lpips_gamma = [math.pow(2, 5 - i * (2 / 16)) for i in range(0, 17)]
    bpp_gamma = [math.pow(2, 6 - i * (6 / 16)) for i in range(0, 17)]

    ga_kwargs = dict(
        img_size=(image_dims[1], image_dims[2]),
        embed_dims=[256, 384, 512, 512], depths=[1, 2, 2, 3], num_heads=[16, 16, 16, 16],
        window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
        norm_layer=nn.LayerNorm, patch_norm=True,
    )

    gs_kwargs = dict(
        img_size=(image_dims[1], image_dims[2]),
        embed_dims=[512, 512, 384, 256], depths=[3, 2, 2, 1], num_heads=[16, 16, 16, 16],
        window_size=8, mlp_ratio=4., norm_layer=nn.LayerNorm, patch_norm=True
    )
