import logging
import torch
import matplotlib.pyplot as plt


def A(x,Phi):
    temp = x*Phi
    y = torch.sum(temp,1) / Phi.shape[1]
    return y

def At(y,Phi):
    temp = torch.unsqueeze(y, 1).repeat(1,Phi.shape[1],1,1)
    x = temp*Phi / Phi.shape[1]
    return x

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter("[%(asctime)s] %(message)s")
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
 
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
 
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
 

def batch_psnr(imgs_batch, refs_batch):
    # n_batches, n_frames, img_h, img_w = refs_batch.size()
    ssqe_batch = (imgs_batch - refs_batch)**2
    mse_batch = torch.sum(ssqe_batch, dim=(2,3)) / (refs_batch.shape[2] * refs_batch.shape[3])
    all_psnr = -10 * torch.log10(mse_batch)
    PSNR = torch.sum(all_psnr, dim=(0,1)) / (refs_batch.shape[0] * refs_batch.shape[1])
    return PSNR