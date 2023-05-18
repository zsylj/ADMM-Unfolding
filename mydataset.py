import os
import scipy
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision
from PIL import Image

#Overlapped, Augmented ######################################HAVENT AUG YET!!!!!
class DAVISDataset(Dataset):
    def __init__(self, folder_dir, mask_dir):
        # self.folder_dir = folder_dir
        self.mask = scipy.io.loadmat(mask_dir)
        self.mask = self.mask['mask']
        self.CSr, self.img_h, self.img_w, self.img_c = self.mask.shape
        self.mask = self.mask.transpose(0, 3, 1, 2)
        self.data_dir_list = []
        video_name_list = os.listdir(folder_dir)
        video_name_list.sort(key = lambda x:(x))
        for video_name in video_name_list:
            video_dir = os.path.join(folder_dir, video_name)
            # print(video_dir)
            img_name_list = os.listdir(video_dir)
            img_name_list.sort(key = lambda x:int(x[:-4]))
            for i in  range(0, len(img_name_list) - self.CSr):
                sub_data_path = img_name_list[i : i + self.CSr]
                sub_dir_path = []
                for item in sub_data_path:
                    item = os.path.join(video_dir, item)
                    sub_dir_path.append(item)
                    self.data_dir_list.append(sub_dir_path)
        print(f'Total number of samples = {len(self.data_dir_list)}')

    def __getitem__(self, index):
        imgs = np.zeros((self.CSr, self.img_h, self.img_w, self.img_c))
        for i, img_path in enumerate(self.data_dir_list[index]):
            img = Image.open(os.path.join(img_path))
            img = torchvision.transforms.functional.to_grayscale(img, num_output_channels=1)
            
            W, H = img.size
            left = int(np.floor((W - 480)/2.0))
            right = left + 480
            img = img.crop((left, 0, right, 480))
            img = img.resize((256, 256))

            img = np.array(img)
            img = img/255.0

            imgs[i, :, :, :] = img
        imgs = imgs.transpose(0, 3, 1, 2)
        # shape of gts (batch), CSr, img_c, img_h, img_w

        measurement = np.sum(np.multiply(imgs, self.mask), axis=0) / self.CSr
        # shape of measurement (batch), img_c, img_h, img_w

        return measurement, imgs
        
    def __len__(self,):
        return len(self.data_dir_list)

#Overlapped, None Augmentation
class lightDAVISDataset(Dataset):
    def __init__(self, folder_dir, mask_dir):
        # self.folder_dir = folder_dir
        self.mask = scipy.io.loadmat(mask_dir)
        self.mask = self.mask['mask']
        self.CSr, self.img_h, self.img_w, self.img_c = self.mask.shape
        self.mask = self.mask.transpose(0, 3, 1, 2)
        self.data_dir_list = []
        video_name_list = os.listdir(folder_dir)
        video_name_list.sort(key = lambda x:(x))
        for video_name in video_name_list:
            video_dir = os.path.join(folder_dir, video_name)
            # print(video_dir)
            img_name_list = os.listdir(video_dir)
            img_name_list.sort(key = lambda x:int(x[:-4]))

            n_segment = int(np.floor(len(img_name_list) / self.CSr))

            for i in range(0, n_segment):
                sub_data_path = img_name_list[i*n_segment : i*n_segment + self.CSr]
                sub_dir_path = []
                for item in sub_data_path:
                    item = os.path.join(video_dir, item)
                    sub_dir_path.append(item)
                    self.data_dir_list.append(sub_dir_path)
        print(f'Total number of samples = {len(self.data_dir_list)}')

    def __getitem__(self, index):
        imgs = np.zeros((self.CSr, self.img_h, self.img_w, self.img_c))
        for i, img_path in enumerate(self.data_dir_list[index]):
            img = Image.open(os.path.join(img_path))
            img = torchvision.transforms.functional.to_grayscale(img, num_output_channels=1)
            
            W, H = img.size
            left = int(np.floor((W - 480)/2.0))
            right = left + 480
            img = img.crop((left, 0, right, 480))
            img = img.resize((256, 256))

            img = np.array(img)
            img = img/255.0

            imgs[i, :, :, :] = img
        imgs = imgs.transpose(0, 3, 1, 2)
        # shape of gts (batch), CSr, img_c, img_h, img_w

        measurement = np.sum(np.multiply(imgs, self.mask), axis=0) / self.CSr
        # shape of measurement (batch), img_c, img_h, img_w

        return measurement, imgs
        
    def __len__(self,):
        return len(self.data_dir_list)


#None Overlapp, None Augmentation
class miniDAVISDataset(Dataset):
    def __init__(self, folder_dir, mask_dir):
        # self.folder_dir = folder_dir
        self.mask = scipy.io.loadmat(mask_dir)
        self.mask = self.mask['mask']
        self.CSr, self.img_h, self.img_w, self.img_c = self.mask.shape
        self.mask = self.mask.transpose(0, 3, 1, 2)
        self.data_dir_list = []
        video_name_list = os.listdir(folder_dir)
        video_name_list.sort(key = lambda x:(x))
        for video_name in video_name_list:
            video_dir = os.path.join(folder_dir, video_name)
            # print(video_dir)
            img_name_list = os.listdir(video_dir)
            img_name_list.sort(key = lambda x:int(x[:-4]))

            start_frame = int((len(img_name_list) - self.CSr)/2)
            sub_data_path = img_name_list[start_frame : start_frame + self.CSr]
            sub_dir_path = []
            for item in sub_data_path:
                item = os.path.join(video_dir, item)
                sub_dir_path.append(item)
                self.data_dir_list.append(sub_dir_path)
        print(f'Total number of samples = {len(self.data_dir_list)}')

    def __getitem__(self, index):
        imgs = np.zeros((self.CSr, self.img_h, self.img_w)) #grayscale
        # imgs = np.zeros((self.CSr, self.img_h, self.img_w, self.img_c)) #color
        for i, img_path in enumerate(self.data_dir_list[index]):
            img = Image.open(os.path.join(img_path))
            img = torchvision.transforms.functional.to_grayscale(img, num_output_channels=1) #grayscale
            # img = torchvision.transforms.functional.to_grayscale(img, num_output_channels=3) #color

            W, H = img.size
            left = int(np.floor((W - 480)/2.0))
            right = left + 480
            img = img.crop((left, 0, right, 480))
            img = img.resize((256, 256))

            img = np.array(img)
            img = img/255.0

            imgs[i, :, :] = img
            # imgs[i, :, :, :] = img

        imgs = imgs #grayscale
        # imgs = imgs.transpose(0, 3, 1, 2) #color

        measurement = np.sum(np.multiply(imgs, self.mask[:, 0, :, :]), axis=0) / self.CSr
        # measurement = np.sum(np.multiply(imgs, self.mask), axis=0) / self.CSr
        # shape of measurement (batch), img_c, img_h, img_w

        return measurement, imgs
        
    def __len__(self,):
        return len(self.data_dir_list)
