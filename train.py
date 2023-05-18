import logging
import os
import torch
import scipy
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from mydataset import DAVISDataset, lightDAVISDataset, miniDAVISDataset
from mymodel import GAPNet, GAPNet2

from utils import batch_psnr, get_logger

batch_size = 16
os.environ["CUDA_VISIBLE_DEVICES"] = '5, 7'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


dataset = miniDAVISDataset(folder_dir='/home/zhangshiyu/data/DAVIS/JPEGImages/480p/', mask_dir='/home/zhangshiyu/GAP/Mask_r0.5.mat')
data_loaded = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=4)
# for inputs, gts in data_loaded:
#     batch_size, CSr, img_h, img_w = gts.size()
#     print(inputs.shape)
#     print(gts.shape)
#     for i in range(0, batch_size):
#         plt.subplot(3,3,1)
#         plt.imshow(np.squeeze(inputs[i, :, :]), cmap='gray')
#         for j in range(0, CSr):    
#             plt.subplot(3,3,j+2)
#             plt.imshow(np.squeeze(gts[i, j, :, :]), cmap='gray')
#         plt.show()

mask = scipy.io.loadmat('/home/zhangshiyu/GAP/Mask_r0.5.mat')
mask = mask['mask']
mask = np.squeeze(mask[:,:,:,0])
CSr, img_h, img_w = mask.shape
mask = torch.from_numpy(mask).float()
mask = torch.unsqueeze(mask, 0).repeat(batch_size, 1, 1, 1)
##############################################
model = GAPNet().to(device)
# model = GAPNet2(n_stage=13).to(device)
##############################################

lr=1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
myloss = torch.nn.MSELoss()
loss_list=[]

n_epochs = 20
n_iter = len(data_loaded)

mask = mask.to(device)

HHt = torch.sum(mask, dim=1, keepdim=False) / (CSr**2)
HHt = HHt.to(device)
# tau = torch.tensor([0.01]).to(device)

get_logger('/home/zhangshiyu/GAP/train.log')
loss_list = []
psnr_list = []
epoch_loss_list = []
epoch_psnr_list = []
logging.info('start training!')
for epoch in range(n_epochs):
    batch_loss_list = []
    batch_psnr_list = []
    for i, (inputs, gts) in enumerate(data_loaded):
        inputs, gts = inputs.float(), gts.float()

        inputs = inputs.to(device)
        gts = gts.to(device)

        outputs = model(inputs, mask, HHt)

        loss = myloss(outputs, gts)
        psnr = -10 * torch.log10(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        logging.info(f'Epoch {epoch+1:4d}/{n_epochs:4d}, Step {i+1:5d}/{n_iter:5d}, Loss = {loss.item():.8f}, Batch_PSNR = {psnr:.8f}')

        batch_loss_list.append(loss.item())
        batch_psnr_list.append(psnr)
        loss_list.append(loss.item())
        psnr_list.append(psnr)

    logging.info("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated()/1024/1024/1024))
    logging.info("torch.cuda.memory_cached: %fGB"%(torch.cuda.memory_cached()/1024/1024/1024))

    AVG_epoch_loss = sum(batch_loss_list) / n_iter
    AVG_epoch_psnr = sum(batch_psnr_list) / n_iter
    epoch_loss_list.append(AVG_epoch_loss)
    epoch_psnr_list.append(AVG_epoch_psnr)
    logging.info(f'AVG Epoch Loss = {AVG_epoch_loss:.8f}, AVG Epoch PSNR = {AVG_epoch_psnr:.8f}')

    if (epoch+1) % 1 == 0:
        name = f'checkpoint_ep{epoch+1}.pth'
        dir = '/home/zhangshiyu/GAP/checkpoint/'
        checkpoint = {'epoch': epoch, 'model_state': model.state_dict(), 'optim_state': optimizer.state_dict()}
        torch.save(checkpoint, os.path.join(dir, name))
        logging.info(f'epoch{epoch+1} saved!')
logging.info('finish training!')

plt.plot([i for i in range(1, n_epochs*int(n_iter)+1)], loss_list,'s-',color='r',label='loss')
plt.xlabel('n_iter')
plt.ylabel('loss via iter')
plt.show()
plt.plot([i for i in range(1, n_epochs+1)], epoch_loss_list,'s-',color='r',label='loss')
plt.xlabel('n_epochs')
plt.ylabel('loss via epoch')
plt.show()

