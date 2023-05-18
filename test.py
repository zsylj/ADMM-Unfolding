import os
import torch
import scipy
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from mydataset import DAVISDataset, miniDAVISDataset
from mymodel import GAPNet, GAPNet2

batch_size = 16
os.environ["CUDA_VISIBLE_DEVICES"] = '5, 7'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


dataset = miniDAVISDataset(folder_dir='/home/zhangshiyu/data/DAVIS/JPEGImages/480p/', mask_dir='/home/zhangshiyu/GAP/Mask_r0.5.mat')
data_loaded = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=4)


mask = scipy.io.loadmat('/home/zhangshiyu/GAP/Mask_r0.5.mat')
mask = mask['mask']
mask = np.squeeze(mask[:,:,:,0])
CSr, img_h, img_w = mask.shape
mask = torch.from_numpy(mask).float()
mask = torch.unsqueeze(mask, 0).repeat(batch_size, 1, 1, 1)

#temp for visualization
mask_temp = mask.detach().numpy()
mask_temp = np.squeeze(mask_temp[0, :, :, :])
mask_sum = np.sum(mask_temp, axis=0)
mask_sumv = np.reshape(mask_sum, (img_h*img_w, 1))
zero_ind = np.where(mask_sumv==0)[0].tolist()
mask_sumv[zero_ind]=np.Inf
mask_sum = np.reshape(mask_sumv, (img_h, img_w))
mask_sum = np.expand_dims(mask_sum, axis=0)
mask_sum = np.repeat(mask_sum, batch_size, axis=0)
##################################################
model = GAPNet().to(device)
# model = GAPNet2(n_stage=13).to(device)
####################################################

optimizer = torch.optim.Adam(model.parameters(), lr=0)

epoch = torch.load('/home/zhangshiyu/GAP/checkpoint/checkpoint_ep20.pth')['epoch']
checkpoint = {'epoch': epoch, 'model_state': model.state_dict(), 'optim_state': optimizer.state_dict()}

model.load_state_dict(checkpoint['model_state'])
optimizer.load_state_dict(checkpoint['optim_state'])
model.eval()

mask = mask.to(device)
HHt = torch.sum(mask, dim=1, keepdim=False) / (CSr ** 2)
HHt = HHt.to(device)


for i, (inputs, gts) in enumerate(data_loaded):
    inputs, gts = inputs.float(), gts.float()

    inputs = inputs.to(device)
    gts = gts.to(device)

    outputs = model(inputs, mask, HHt)

    print(f'ground truth minimum:{torch.min(gts)}, maximum:{torch.max(gts)}')
    print(f'output minimum:{torch.min(outputs)}, maximum:{torch.max(outputs)}')

    outputs = outputs.to(torch.device('cpu'))
    outputs = outputs.detach().numpy()

    gts = gts.to(torch.device('cpu'))
    gts = gts.detach().numpy()
    
    inputs = inputs.to(torch.device('cpu'))
    inputs = inputs.detach().numpy()
    
    # temp for visualization
    HHt_inv_y = inputs / mask_sum

    # for j in range(0, batch_size):
    #     images = np.squeeze(outputs[j, :, :, :])
    #     frames = []
    #     for k in range(0, 8):
    #         frames.append(images[:, :, k])
    #     imageio.mimsave(f'/home/zhangshiyu/GAP/reconstruction/{i:2d}_{j:2d}_result.gif', frames, 'GIF', duration=0.5)

    for i in range(0, 1): #batch_size
        for j in range(0, CSr):
            plt.figure(figsize=(20.48, 20.48))
            plt.subplot(1,4,1)
            plt.imshow(np.squeeze(outputs[i, j, :, :]), cmap='gray')
            plt.title('reconstruction')
            plt.subplot(1,4,2)
            plt.imshow(np.squeeze(gts[i, j, :, :]), cmap='gray')
            plt.title('ground truth')
            plt.subplot(1,4,3)
            plt.imshow(np.squeeze(inputs[i, :, :]), cmap='gray')
            plt.title('measurement')
            plt.subplot(1,4,4)
            plt.imshow(np.squeeze(HHt_inv_y[i, :, :]), cmap='gray')
            plt.title('HHt_inv * y')
            plt.show()

    break
