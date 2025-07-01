import torch
import numpy as np
from PIL import Image

def savePic(originPic, reconPic, namepath):
    B, C, H, W = originPic.size()
    whiteColumn = torch.ones(B, C, H, 20).to(originPic.device)
    pic = torch.cat((originPic, whiteColumn, reconPic), dim=-1)
    for i in range(pic.shape[0]):
        frame = pic[i].detach().cpu().numpy().transpose(1, 2, 0) * 255
        frame = np.clip(np.rint(frame), 0, 255)
        frame = Image.fromarray(frame.astype('uint8'), 'RGB')
        frame.save(namepath + ("%02d.png" % i))

def savePic_samples(originPic, reconPic, namepath):
    B, times, C, H, W = reconPic.size()
    for t in range(times):
        whiteColumn = torch.ones(B, C, H, W // 10).to(originPic.device)
        pic = torch.cat((originPic, whiteColumn, reconPic[:, t, :, :, :]), dim=-1)
        for i in range(pic.shape[0]):
            frame = pic[i].detach().cpu().numpy().transpose(1, 2, 0) * 255
            frame = np.clip(np.rint(frame), 0, 255)
            frame = Image.fromarray(frame.astype('uint8'), 'RGB')
            frame.save(namepath + ("%02d_times%02d.png" % (i, t)))

def savePic_4pics(originPic_1, reconPic_1, originPic_2, reconPic_2, namepath):
    B, C, H, W = originPic_1.size()
    whiteColumn = torch.ones(B, C, H, W // 10).cuda()
    whiteRow = torch.ones(B, C, H // 10, 2*W + W // 10).cuda()
    pic_1 = torch.cat((originPic_1, whiteColumn, reconPic_1), dim=-1)
    pic_2 = torch.cat((originPic_2, whiteColumn, reconPic_2), dim=-1)
    pic = torch.cat((pic_1, whiteRow, pic_2), dim=-2)
    for i in range(pic.shape[0]):
        frame = pic[i].detach().cpu().numpy().transpose(1, 2, 0) * 255
        frame = np.clip(np.rint(frame), 0, 255)
        frame = Image.fromarray(frame.astype('uint8'), 'RGB')
        frame.save(namepath + ("%02d.png" % i))