import os
import time
import scipy
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

#Inspect and print all Image information
#Read Images from folder and save them into .mat file
#Generate Mask and save them into .mat file
#Generate Measurement and save them into .mat file

dir='/home/zhangshiyu/data/DAVIS/JPEGImages/480p'

def Inspection(dir):
    # Inspection Data
    # Print order, dir, n_frame, img_size
    folder_list = os.listdir(dir)
    folder_list.sort(key = lambda x:(x)) #
    img_dir_list = []
    len_list = []
    shape_list = []
    ind = 0
    for folder in folder_list:
        sub_dir = os.path.join(dir, folder)
        if os.path.isfile(sub_dir)==True:
            continue
        else:
            img_name_list = os.listdir(sub_dir)
            img_name_list.sort(key = lambda x:int(x[:-4])) #

            img_name_list2 = img_name_list
            for i in range(0,len(img_name_list2)):
                img_name_list2[i] = sub_dir + '/' + img_name_list2[i]
            img_dir_list.append(img_name_list2)

            len_list.append(len(img_name_list))
            
            img = Image.open(os.path.join(sub_dir, img_name_list[0]))
            img = np.array(img)
            img = img/255.0

            shape_list.append(img.shape)
            print(f'No.{ind:2d} {sub_dir:65s} {len(img_name_list):3d} {img.shape}')
            ind = ind+1
    return folder_list, img_dir_list, len_list, shape_list

folder_list, img_dir_list, len_list, shape_list = Inspection(dir)


# 50% Masking Ratio
def MaskGenerate_original(CSr, img_h, img_w, img_c, mask_ratio):
    #Generate Mask with specific ratio
    N = img_h * img_w    
    n_zeros = int(np.floor(N * mask_ratio))
    temp_vect = np.ones(N)
    temp_vect[0:n_zeros]=0
    mask_mat = np.zeros((CSr, img_h, img_w))
    for i in range (0, CSr):
        np.random.seed(114514)
        np.random.shuffle(temp_vect)
        temp_mtx = np.reshape(temp_vect, (img_h, img_w))
        mask_mat[i, :, :] = temp_mtx
        
    mask_mat = np.expand_dims(mask_mat, axis=3)
    mask_mat = np.repeat(mask_mat, img_c, axis=3)

    mask_sum = np.sum(mask_mat, axis=0) / CSr
    plt.imshow(mask_sum)
    plt.show()

    for i in range (0, CSr):
        plt.imshow(np.squeeze(mask_mat[i, :, :, :]))
        plt.show()

    dic = {'mask': mask_mat}
    scipy.io.savemat('Mask_r0.875.mat', dic)

MaskGenerate_original(8, 256, 256, 3, 0.875)





# Bernoulli Mask
# def MaskGenerate_bernoulli(CSr, img_h, img_w, img_c, p):
#     #Generate Mask each pixel is bernoulli
#     N = img_h * img_w
#     mask_mat = np.zeros((CSr, img_h, img_w))
#     for i in range(0, CSr):
#         flag=[1]
#         while flag != []:
#             temp_vect = np.random.randn(N)
#             flag = np.where(temp_vect==0)[0].tolist()
#             ind0 = np.where(temp_vect<0)[0].tolist()
#             ind1 = np.where(temp_vect>0)[0].tolist()
#             temp_vect[ind0]=0
#             temp_vect[ind1]=1
#         mask_mat[i, :, :] = np.reshape(temp_vect, (img_h, img_w))
    
#     mask_mat = np.expand_dims(mask_mat, axis=3)
#     mask_mat = np.repeat(mask_mat, img_c, axis=3)

#     # for i in range (0, CSr):
#     #     plt.imshow(np.squeeze(mask_mat[i, :, :, :]))
#     #     plt.show()

#     dic = {'mask': mask_mat}
#     scipy.io.savemat('Mask_p0.5.mat', dic)

# MaskGenerate_bernoulli(8, 256, 256, 3, 0.5)





# Ortho-Complete Mask
# def MaskGenerate_orthocomplete(CSr, img_h, img_w, img_c):
#     N = img_h * img_w
#     n = int(N / CSr)
#     ind = np.arange(N)
#     np.random.seed(114514)
#     np.random.shuffle(ind)
#     ind_new = []
#     for i in range(0, CSr):
#         temp = ind[i * n : (i+1) * n]
#         ind_new.append(temp.tolist())
    
#     mask_mat = np.zeros((CSr, img_h, img_w))
#     for i in range(0, CSr):
#         temp = np.zeros(N)
#         temp[ind_new[i]]=1
#         temp = temp.reshape(img_h, img_w)
#         mask_mat[i, :, :] = temp
    
#     mask_mat = np.expand_dims(mask_mat, axis=3)
#     mask_mat = np.repeat(mask_mat, img_c, axis=3)

#     mask_sum = np.sum(mask_mat, axis=0)
#     plt.imshow(mask_sum)
#     plt.show()

#     for i in range (0, CSr):
#         plt.imshow(np.squeeze(mask_mat[i, :, :, :]))
#         plt.show()
    
#     dic = {'mask': mask_mat}
#     scipy.io.savemat('Mask_oc.mat', dic)
    
# MaskGenerate_orthocomplete(8, 256, 256, 3)

