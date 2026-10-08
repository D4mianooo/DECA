import os, sys
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import scipy
from skimage.io import imread, imsave
from skimage.transform import estimate_transform, warp, resize, rescale
from glob import glob
from torch.utils.data import Dataset, DataLoader, ConcatDataset

class VoxelDataset(Dataset):
    def __init__(self, K, image_size, scale, trans_scale = 0, dataname='vox2', n_train=100000, isTemporal=False, isEval=False, isSingle=False):
        self.K = K
        self.image_size = image_size
        if dataname == 'vox1':
            self.kpt_suffix = '.txt'
            self.imagefolder = '/ps/project/face2d3d/VoxCeleb/vox1/dev/images_cropped'
            self.kptfolder = '/ps/scratch/yfeng/Data/VoxCeleb/vox1/landmark_2d'

            self.face_dict = {}
            for person_id in sorted(os.listdir(self.kptfolder)):
                for video_id in os.listdir(os.path.join(self.kptfolder, person_id)):   
                    for face_id in os.listdir(os.path.join(self.kptfolder, person_id, video_id)):
                        if 'txt' in face_id:
                            continue
                        key = person_id + '/' + video_id + '/' + face_id
                        # if key not in self.face_dict.keys():
                        #     self.face_dict[key] = []
                        name_list = os.listdir(os.path.join(self.kptfolder, person_id, video_id, face_id))
                        name_list = [name.split['.'][0] for name in name_list]
                        if len(name_list)<self.K:
                            continue
                        self.face_dict[key] = sorted(name_list)

        elif dataname == 'vox2':
            # clean version: filter out images with bad lanmark labels, may lack extreme pose example
            self.kpt_suffix = '.npy'
            self.imagefolder = '/mnt/d/datasets/Vox2/train/'
            self.kptfolder = '/mnt/d/datasets/Vox2/train_annotated_fan'
            self.segfolder = '/mnt/d/datasets/Vox2/train_seg'

            cleanlist_path = '/mnt/d/datasets/Vox2/train_second_cleaning.npy'
            self.cleanlist = np.load(cleanlist_path).astype('str')


        self.isTemporal = isTemporal
        self.scale = scale #[scale_min, scale_max]
        self.trans_scale = trans_scale #[dx, dy]
        self.isSingle = isSingle
        if isSingle:
            self.K = 1

    def __len__(self):
        return len(self.cleanlist)
    
    def __getitem__(self, idx):
        images_list = []; kpt_list = []; fullname_list = []; mask_list = []

        for i in range(self.K):
            name = self.cleanlist[idx, i]
            image_path = (os.path.join(self.imagefolder, name + '.jpg'))
            kpt_path = (os.path.join(self.kptfolder, name + self.kpt_suffix))
            seg_path = (os.path.join(self.segfolder, name + '.npy'))
                                            
            image = imread(image_path)/255.
            kpt = np.load(kpt_path)[:,:2]
            mask = self.load_mask(seg_path, image.shape[0], image.shape[1])

            ### crop information
            tform = self.crop(image, kpt)
            ## crop 
            cropped_image = warp(image, tform.inverse, output_shape=(self.image_size, self.image_size))
            cropped_mask = warp(mask, tform.inverse, output_shape=(self.image_size, self.image_size))
            cropped_kpt = np.dot(tform.params, np.hstack([kpt, np.ones([kpt.shape[0],1])]).T).T # np.linalg.inv(tform.params)

            # normalized kpt
            cropped_kpt[:,:2] = cropped_kpt[:,:2]/self.image_size * 2  - 1

            images_list.append(cropped_image.transpose(2,0,1))
            kpt_list.append(cropped_kpt)
            mask_list.append(cropped_mask)

        ###
        images_array = torch.from_numpy(np.array(images_list)).type(dtype = torch.float32) #K,224,224,3
        kpt_array = torch.from_numpy(np.array(kpt_list)).type(dtype = torch.float32) #K,224,224,3
        mask_array = torch.from_numpy(np.array(mask_list)).type(dtype = torch.float32) #K,224,224,3

        if self.isSingle:
            images_array = images_array.squeeze()
            kpt_array = kpt_array.squeeze()
            mask_array = mask_array.squeeze()
                    
        data_dict = {
            'image': images_array,
            'landmark': kpt_array,
            'mask': mask_array
        }
        
        return data_dict

    def crop(self, image, kpt):
        left = np.min(kpt[:,0]); right = np.max(kpt[:,0]); 
        top = np.min(kpt[:,1]); bottom = np.max(kpt[:,1])

        h, w, _ = image.shape
        old_size = (right - left + bottom - top)/2
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 ])#+ old_size*0.1])
        # translate center
        trans_scale = (np.random.rand(2)*2 -1) * self.trans_scale
        center = center + trans_scale*old_size # 0.5

        scale = np.random.rand() * (self.scale[1] - self.scale[0]) + self.scale[0]
        size = int(old_size*scale)

        # crop image
        src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])
        DST_PTS = np.array([[0,0], [0,self.image_size - 1], [self.image_size - 1, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)
        
        # cropped_image = warp(image, tform.inverse, output_shape=(self.image_size, self.image_size))
        # # change kpt accordingly
        # cropped_kpt = np.dot(tform.params, np.hstack([kpt, np.ones([kpt.shape[0],1])]).T).T # np.linalg.inv(tform.params)
        return tform

    def load_mask(self, maskpath, h, w):
        # print(maskpath)
        if os.path.isfile(maskpath):
            vis_parsing_anno = np.load(maskpath)
            # atts = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
            #     'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']
            mask = np.zeros_like(vis_parsing_anno)
            # for i in range(1, 16):
            mask[vis_parsing_anno>0.5] = 1.
        else:
            mask = np.ones((h, w))
        return mask
