# -------------------------------------------------------------------
# Copyright (C) 2020 Harbin Institute of Technology, China
# Author: Xudong Lv (15B901019@hit.edu.cn)
# Released under Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# http://creativecommons.org/licenses/by-nc-sa/4.0/

import csv
import os
from math import radians
import cv2

import h5py
import mathutils
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TTF
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from utils import invert_pose, rotate_forward, quaternion_from_matrix, read_calib_file
from pykitti import odometry
import pykitti
#
# def get_calib_kitti_odom(sequence):
#     if sequence == 0:
#         return torch.tensor([[718.856, 0, 607.1928], [0, 718.856, 185.2157], [0, 0, 1]])
#     elif sequence == 3:
#         return torch.tensor([[721.5377, 0, 609.5593], [0, 721.5377, 172.854], [0, 0, 1]])
#     elif sequence in [5, 6, 7, 8, 9]:
#         return torch.tensor([[707.0912, 0, 601.8873], [0, 707.0912, 183.1104], [0, 0, 1]])
#     else:
#         raise TypeError("Sequence Not Available")


class DatasetLidarCameraKittiOdometry(Dataset):

    def __init__(self, dataset_dir, transform=None, augmentation=False, use_reflectance=False,
                 max_t=1.5, max_r=20., split='val', device='cpu', val_sequence='00', suf='.png'):
        super(DatasetLidarCameraKittiOdometry, self).__init__()
        self.use_reflectance = use_reflectance
        self.maps_folder = ''
        self.device = device
        self.max_r = max_r
        self.max_t = max_t
        self.augmentation = augmentation
        self.root_dir = dataset_dir
        self.transform = transform
        self.split = split
        self.GTs_R = {}
        self.GTs_T = {}
        self.GTs_T_cam02_velo = {}
        self.K = {}
        self.suf = suf

        self.all_files = []
        self.sequence_list = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10',
                              '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']
        # self.model = CameraModel()
        # self.model.focal_length = [7.18856e+02, 7.18856e+02]
        # self.model.principal_point = [6.071928e+02, 1.852157e+02]
        # for seq in ['00', '03', '05', '06', '07', '08', '09']:
        for seq in self.sequence_list:
            odom = odometry(self.root_dir, seq)
            calib = odom.calib
            T_cam02_velo_np = calib.T_cam2_velo #gt pose from cam02 to velo_lidar (T_cam02_velo: 4x4)
            self.K[seq] = calib.K_cam2 # 3x3
            # T_cam02_velo = torch.from_numpy(T_cam02_velo_np)
            # GT_R = quaternion_from_matrix(T_cam02_velo[:3, :3])
            # GT_T = T_cam02_velo[3:, :3]
            # self.GTs_R[seq] = GT_R # GT_R = np.array([row['qw'], row['qx'], row['qy'], row['qz']])
            # self.GTs_T[seq] = GT_T # GT_T = np.array([row['x'], row['y'], row['z']])
            self.GTs_T_cam02_velo[seq] = T_cam02_velo_np #gt pose from cam02 to velo_lidar (T_cam02_velo: 4x4)

            image_list = os.listdir(os.path.join(dataset_dir, 'sequences', seq, 'image_2'))
            image_list.sort()

            for image_name in image_list:
                if not os.path.exists(os.path.join(dataset_dir, 'sequences', seq, 'velodyne',
                                                   str(image_name.split('.')[0])+'.bin')):
                    continue
                if not os.path.exists(os.path.join(dataset_dir, 'sequences', seq, 'image_2',
                                                   str(image_name.split('.')[0])+suf)):
                    continue
                if seq == val_sequence:
                    if split.startswith('val') or split == 'test':
                        self.all_files.append(os.path.join(seq, image_name.split('.')[0]))
                elif (not seq == val_sequence) and split == 'train':
                    self.all_files.append(os.path.join(seq, image_name.split('.')[0]))

        self.val_RT = []
        if split == 'val' or split == 'test':
            # val_RT_file = os.path.join(dataset_dir, 'sequences',
            #                            f'val_RT_seq{val_sequence}_{max_r:.2f}_{max_t:.2f}.csv')
            val_RT_file = os.path.join(dataset_dir, 'sequences',
                                       f'val_RT_left_seq{val_sequence}_{max_r:.2f}_{max_t:.2f}.csv')
            if os.path.exists(val_RT_file):
                print(f'VAL SET: Using this file: {val_RT_file}')
                df_test_RT = pd.read_csv(val_RT_file, sep=',')
                for index, row in df_test_RT.iterrows():
                    self.val_RT.append(list(row))
            else:
                print(f'VAL SET - Not found: {val_RT_file}')
                print("Generating a new one")
                val_RT_file = open(val_RT_file, 'w')
                val_RT_file = csv.writer(val_RT_file, delimiter=',')
                val_RT_file.writerow(['id', 'tx', 'ty', 'tz', 'rx', 'ry', 'rz'])
                for i in range(len(self.all_files)):
                    rotz = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                    roty = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                    rotx = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                    transl_x = np.random.uniform(-max_t, max_t)
                    transl_y = np.random.uniform(-max_t, max_t)
                    transl_z = np.random.uniform(-max_t, max_t)
                    # transl_z = np.random.uniform(-max_t, min(max_t, 1.))
                    val_RT_file.writerow([i, transl_x, transl_y, transl_z,
                                           rotx, roty, rotz])
                    self.val_RT.append([i, transl_x, transl_y, transl_z,
                                         rotx, roty, rotz])

            assert len(self.val_RT) == len(self.all_files), "Something wrong with test RTs"

    def get_ground_truth_poses(self, sequence, frame):
        return self.GTs_T[sequence][frame], self.GTs_R[sequence][frame]

    def custom_transform(self, rgb, img_rotation=0., flip=False):
        to_tensor = transforms.ToTensor()
        normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])

        #rgb = crop(rgb)
        if self.split == 'train':
            color_transform = transforms.ColorJitter(0.1, 0.1, 0.1)
            rgb = color_transform(rgb)
            if flip:
                rgb = TTF.hflip(rgb)
            rgb = TTF.rotate(rgb, img_rotation)
            #io.imshow(np.array(rgb))
            #io.show()

        rgb = to_tensor(rgb)
        rgb = normalization(rgb)
        return rgb

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        item = self.all_files[idx]
        seq = str(item.split('/')[0])
        rgb_name = str(item.split('/')[1])
        img_path = os.path.join(self.root_dir, 'sequences', seq, 'image_2', rgb_name+self.suf)
        lidar_path = os.path.join(self.root_dir, 'sequences', seq, 'velodyne', rgb_name+'.bin')
        lidar_scan = np.fromfile(lidar_path, dtype=np.float32)
        pc = lidar_scan.reshape((-1, 4))
        valid_indices = pc[:, 0] < -3.
        valid_indices = valid_indices | (pc[:, 0] > 3.)
        valid_indices = valid_indices | (pc[:, 1] < -3.)
        valid_indices = valid_indices | (pc[:, 1] > 3.)
        pc = pc[valid_indices].copy()
        pc_org = torch.from_numpy(pc.astype(np.float32))
        # if self.use_reflectance:
        #     reflectance = pc[:, 3].copy()
        #     reflectance = torch.from_numpy(reflectance).float()

        RT = self.GTs_T_cam02_velo[seq].astype(np.float32)

        if pc_org.shape[1] == 4 or pc_org.shape[1] == 3:
            pc_org = pc_org.t()
        if pc_org.shape[0] == 3:
            homogeneous = torch.ones(pc_org.shape[1]).unsqueeze(0)
            pc_org = torch.cat((pc_org, homogeneous), 0)
        elif pc_org.shape[0] == 4:
            if not torch.all(pc_org[3, :] == 1.):
                pc_org[3, :] = 1.
        else:
            raise TypeError("Wrong PointCloud shape")
        
        pc_rot = np.matmul(RT, pc_org.numpy())
        pc_rot = pc_rot.astype(np.float32).copy()
        pc_in = torch.from_numpy(pc_rot)

        # pc_rot = np.matmul(RT, pc.T)
        # pc_rot = pc_rot.astype(np.float).T.copy()
        # pc_in = torch.from_numpy(pc_rot.astype(np.float32))#.float()

        # if pc_in.shape[1] == 4 or pc_in.shape[1] == 3:
        #     pc_in = pc_in.t()
        # if pc_in.shape[0] == 3:
        #     homogeneous = torch.ones(pc_in.shape[1]).unsqueeze(0)
        #     pc_in = torch.cat((pc_in, homogeneous), 0)
        # elif pc_in.shape[0] == 4:
        #      if not torch.all(pc_in[3,:] == 1.):
        #         pc_in[3,:] = 1.
        # else:
        #     raise TypeError("Wrong PointCloud shape")

        h_mirror = False
        # if np.random.rand() > 0.5 and self.split == 'train':
        #     h_mirror = True
        #     pc_in[1, :] *= -1

        img = Image.open(img_path)
        # img = cv2.imread(img_path)
        img_rotation = 0.
        # if self.split == 'train':
        #     img_rotation = np.random.uniform(-5, 5)
        try:
            img = self.custom_transform(img, img_rotation, h_mirror)
        except OSError:
            new_idx = np.random.randint(0, self.__len__())
            return self.__getitem__(new_idx)

        # Rotate PointCloud for img_rotation
        if self.split == 'train':
            R = mathutils.Euler((radians(img_rotation), 0, 0), 'XYZ')
            T = mathutils.Vector((0., 0., 0.))
            pc_in = rotate_forward(pc_in, R, T)

        if self.split == 'train':
            max_angle = self.max_r
            rotz = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            roty = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            rotx = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            transl_x = np.random.uniform(-self.max_t, self.max_t)
            transl_y = np.random.uniform(-self.max_t, self.max_t)
            transl_z = np.random.uniform(-self.max_t, self.max_t)
            # transl_z = np.random.uniform(-self.max_t, min(self.max_t, 1.))
        else:
            initial_RT = self.val_RT[idx]
            rotz = initial_RT[6]
            roty = initial_RT[5]
            rotx = initial_RT[4]
            transl_x = initial_RT[1]
            transl_y = initial_RT[2]
            transl_z = initial_RT[3]

        # 随机设置一定范围内的标定参数扰动值
        # train的时候每次都随机生成,每个epoch使用不同的参数
        # test则在初始化的时候提前设置好,每个epoch都使用相同的参数
        R = mathutils.Euler((rotx, roty, rotz), 'XYZ')
        T = mathutils.Vector((transl_x, transl_y, transl_z))

        R, T = invert_pose(R, T)
        R, T = torch.tensor(R), torch.tensor(T)

        #io.imshow(depth_img.numpy(), cmap='jet')
        #io.show()
        calib = self.K[seq]
        if h_mirror:
            calib[2] = (img.shape[2] / 2)*2 - calib[2]

        if self.split == 'test':
            sample = {'rgb': img, 'point_cloud': pc_in, 'calib': calib,
                      'tr_error': T, 'rot_error': R, 'seq': int(seq), 'img_path': img_path,
                      'rgb_name': rgb_name + '.png', 'item': item, 'extrin': RT,
                      'initial_RT': initial_RT}
        else:
            sample = {'rgb': img, 'point_cloud': pc_in, 'calib': calib,
                      'tr_error': T, 'rot_error': R, 'seq': int(seq),
                      'rgb_name': rgb_name, 'item': item, 'extrin': RT}

        return sample


class DatasetLidarCameraKittiRaw(Dataset):

    def __init__(self, dataset_dir, transform=None, augmentation=False, use_reflectance=False,
                 max_t=1.5, max_r=15.0, split='val', device='cpu', val_sequence='2011_09_26_drive_0117_sync'):
        super(DatasetLidarCameraKittiRaw, self).__init__()
        self.use_reflectance = use_reflectance
        self.maps_folder = ''
        self.device = device
        self.max_r = max_r
        self.max_t = max_t
        self.augmentation = augmentation
        self.root_dir = dataset_dir
        self.transform = transform
        self.split = split
        self.GTs_R = {}
        self.GTs_T = {}
        self.GTs_T_cam02_velo = {}
        self.max_depth = 80
        self.K_list = {}

        self.all_files = []
        date_list = ['2011_09_26', '2011_09_28', '2011_09_29', '2011_09_30', '2011_10_03']
        data_drive_list = ['0001', '0002', '0004', '0016', '0027']
        self.calib_date = {}

        for i in range(len(date_list)):
            date = date_list[i]
            data_drive = data_drive_list[i]
            data = pykitti.raw(self.root_dir, date, data_drive)
            calib = {'K2': data.calib.K_cam2, 'K3': data.calib.K_cam3,
                     'RT2': data.calib.T_cam2_velo, 'RT3': data.calib.T_cam3_velo}
            self.calib_date[date] = calib

        # date = val_sequence[:10]
        # seq = val_sequence
        # image_list = os.listdir(os.path.join(dataset_dir, date, seq, 'image_02/data'))
        # image_list.sort()
        #
        # for image_name in image_list:
        #     if not os.path.exists(os.path.join(dataset_dir, date, seq, 'velodyne_points/data',
        #                                        str(image_name.split('.')[0]) + '.bin')):
        #         continue
        #     if not os.path.exists(os.path.join(dataset_dir, date, seq, 'image_02/data',
        #                                        str(image_name.split('.')[0]) + '.jpg')):  # png
        #         continue
        #     self.all_files.append(os.path.join(date, seq, 'image_02/data', image_name.split('.')[0]))

        date = val_sequence[:10]
        test_list = ['2011_09_26_drive_0005_sync', '2011_09_26_drive_0070_sync', '2011_10_03_drive_0027_sync']
        seq_list = os.listdir(os.path.join(self.root_dir, date))

        for seq in seq_list:
            if not os.path.isdir(os.path.join(dataset_dir, date, seq)):
                continue
            image_list = os.listdir(os.path.join(dataset_dir, date, seq, 'image_02/data'))
            image_list.sort()

            for image_name in image_list:
                if not os.path.exists(os.path.join(dataset_dir, date, seq, 'velodyne_points/data',
                                                   str(image_name.split('.')[0])+'.bin')):
                    continue
                if not os.path.exists(os.path.join(dataset_dir, date, seq, 'image_02/data',
                                                   str(image_name.split('.')[0])+'.jpg')): # png
                    continue
                if seq == val_sequence and (not split == 'train'):
                    self.all_files.append(os.path.join(date, seq, 'image_02/data', image_name.split('.')[0]))
                elif (not seq == val_sequence) and split == 'train' and seq not in test_list:
                    self.all_files.append(os.path.join(date, seq, 'image_02/data', image_name.split('.')[0]))

        self.val_RT = []
        if split == 'val' or split == 'test':
            val_RT_file = os.path.join(dataset_dir,
                                       f'val_RT_seq{val_sequence}_{max_r:.2f}_{max_t:.2f}.csv')
            if os.path.exists(val_RT_file):
                print(f'VAL SET: Using this file: {val_RT_file}')
                df_test_RT = pd.read_csv(val_RT_file, sep=',')
                for index, row in df_test_RT.iterrows():
                    self.val_RT.append(list(row))
            else:
                print(f'TEST SET - Not found: {val_RT_file}')
                print("Generating a new one")
                val_RT_file = open(val_RT_file, 'w')
                val_RT_file = csv.writer(val_RT_file, delimiter=',')
                val_RT_file.writerow(['id', 'tx', 'ty', 'tz', 'rx', 'ry', 'rz'])
                for i in range(len(self.all_files)):
                    rotz = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                    roty = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                    rotx = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                    transl_x = np.random.uniform(-max_t, max_t)
                    transl_y = np.random.uniform(-max_t, max_t)
                    transl_z = np.random.uniform(-max_t, max_t)
                    # transl_z = np.random.uniform(-max_t, min(max_t, 1.))
                    val_RT_file.writerow([i, transl_x, transl_y, transl_z,
                                           rotx, roty, rotz])
                    self.val_RT.append([float(i), transl_x, transl_y, transl_z,
                                         rotx, roty, rotz])

            assert len(self.val_RT) == len(self.all_files), "Something wrong with test RTs"

    def get_ground_truth_poses(self, sequence, frame):
        return self.GTs_T[sequence][frame], self.GTs_R[sequence][frame]

    def custom_transform(self, rgb, img_rotation=0., flip=False):
        to_tensor = transforms.ToTensor()
        normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])

        #rgb = crop(rgb)
        if self.split == 'train':
            color_transform = transforms.ColorJitter(0.1, 0.1, 0.1)
            # color_transform = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3 / 3.14)
            rgb = color_transform(rgb)
            if flip:
                rgb = TTF.hflip(rgb)
            rgb = TTF.rotate(rgb, img_rotation)
            #io.imshow(np.array(rgb))
            #io.show()

        rgb = to_tensor(rgb)
        rgb = normalization(rgb)
        return rgb

    def __len__(self):
        return len(self.all_files)

    # self.all_files.append(os.path.join(date, seq, 'image_2/data', image_name.split('.')[0]))
    def __getitem__(self, idx):
        item = self.all_files[idx]
        date = str(item.split('/')[0])
        seq = str(item.split('/')[1])
        rgb_name = str(item.split('/')[4])
        img_path = os.path.join(self.root_dir, date, seq, 'image_02/data', rgb_name+'.jpg') # png
        lidar_path = os.path.join(self.root_dir, date, seq, 'velodyne_points/data', rgb_name+'.bin')
        lidar_scan = np.fromfile(lidar_path, dtype=np.float32)
        pc = lidar_scan.reshape((-1, 4))
        valid_indices = pc[:, 0] < -3.
        valid_indices = valid_indices | (pc[:, 0] > 3.)
        valid_indices = valid_indices | (pc[:, 1] < -3.)
        valid_indices = valid_indices | (pc[:, 1] > 3.)
        pc = pc[valid_indices].copy()
        pc_lidar = pc.copy()
        pc_org = torch.from_numpy(pc.astype(np.float32))
        if self.use_reflectance:
            reflectance = pc[:, 3].copy()
            reflectance = torch.from_numpy(reflectance).float()

        calib = self.calib_date[date]
        RT_cam02 = calib['RT2'].astype(np.float32)
        # camera intrinsic parameter
        calib_cam02 = calib['K2']  # 3x3

        E_RT = RT_cam02

        if pc_org.shape[1] == 4 or pc_org.shape[1] == 3:
            pc_org = pc_org.t()
        if pc_org.shape[0] == 3:
            homogeneous = torch.ones(pc_org.shape[1]).unsqueeze(0)
            pc_org = torch.cat((pc_org, homogeneous), 0)
        elif pc_org.shape[0] == 4:
            if not torch.all(pc_org[3, :] == 1.):
                pc_org[3, :] = 1.
        else:
            raise TypeError("Wrong PointCloud shape")

        pc_rot = np.matmul(E_RT, pc_org.numpy())
        pc_rot = pc_rot.astype(np.float32).copy()
        pc_in = torch.from_numpy(pc_rot)

        h_mirror = False
        # if np.random.rand() > 0.5 and self.split == 'train':
        #     h_mirror = True
        #     pc_in[0, :] *= -1

        img = Image.open(img_path)
        # print(img_path)
        # img = cv2.imread(img_path)
        img_rotation = 0.
        # if self.split == 'train':
        #     img_rotation = np.random.uniform(-5, 5)
        try:
            img = self.custom_transform(img, img_rotation, h_mirror)
        except OSError:
            new_idx = np.random.randint(0, self.__len__())
            return self.__getitem__(new_idx)

        # Rotate PointCloud for img_rotation
        # if self.split == 'train':
        #     R = mathutils.Euler((radians(img_rotation), 0, 0), 'XYZ')
        #     T = mathutils.Vector((0., 0., 0.))
        #     pc_in = rotate_forward(pc_in, R, T)

        if self.split == 'train':
            max_angle = self.max_r
            rotz = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            roty = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            rotx = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            transl_x = np.random.uniform(-self.max_t, self.max_t)
            transl_y = np.random.uniform(-self.max_t, self.max_t)
            transl_z = np.random.uniform(-self.max_t, self.max_t)
            # transl_z = np.random.uniform(-self.max_t, min(self.max_t, 1.))
            initial_RT = 0
        else:
            initial_RT = self.val_RT[idx]
            rotz = initial_RT[6]
            roty = initial_RT[5]
            rotx = initial_RT[4]
            transl_x = initial_RT[1]
            transl_y = initial_RT[2]
            transl_z = initial_RT[3]

        # 随机设置一定范围内的标定参数扰动值
        # train的时候每次都随机生成,每个epoch使用不同的参数
        # test则在初始化的时候提前设置好,每个epoch都使用相同的参数
        R = mathutils.Euler((rotx, roty, rotz), 'XYZ')
        T = mathutils.Vector((transl_x, transl_y, transl_z))

        R, T = invert_pose(R, T)
        R, T = torch.tensor(R), torch.tensor(T)

        #io.imshow(depth_img.numpy(), cmap='jet')
        #io.show()
        calib = calib_cam02
        # calib = get_calib_kitti_odom(int(seq))
        if h_mirror:
            calib[2] = (img.shape[2] / 2)*2 - calib[2]

        # sample = {'rgb': img, 'point_cloud': pc_in, 'calib': calib, 'pc_org': pc_org, 'img_path': img_path,
        #           'tr_error': T, 'rot_error': R, 'seq': int(seq), 'rgb_name': rgb_name, 'item': item,
        #           'extrin': E_RT, 'initial_RT': initial_RT}
        sample = {'rgb': img, 'point_cloud': pc_in, 'calib': calib, 'pc_org': pc_org, 'img_path': img_path,
                  'tr_error': T, 'rot_error': R, 'rgb_name': rgb_name + '.png', 'item': item,
                  'extrin': E_RT, 'initial_RT': initial_RT, 'pc_lidar': pc_lidar}

        return sample



class DatasetKittiOdometryStereo(Dataset):

    def __init__(self, dataset_dir, transform=None, augmentation=False, use_reflectance=False,
                 max_t=1.5, max_r=15.0, split='val', device='cpu', val_sequence='00', tf_range=1, suf='.png'):
        super(DatasetKittiOdometryStereo, self).__init__()
        self.use_reflectance = use_reflectance
        self.maps_folder = ''
        self.device = device
        self.max_r = max_r
        self.max_t = max_t
        self.augmentation = augmentation
        self.root_dir = dataset_dir
        self.transform = transform
        self.split = split
        self.GTs_R = {}
        self.GTs_T = {}
        self.GTs_T_cam02_velo = {}
        self.GTs_T_cam03_velo = {}
        self.Gts_cam02_cam03 = {}
        self.max_depth = 80
        self.K2_list = {}
        self.K3_list = {}
        self.tf_range = tf_range
        self.downsample = 16
        self.suf = suf

        self.all_files = []
        # self.sequence_list = ['00', '01']
        self.sequence_list = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10',
                              '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']
        self.train_list = ['01', '02', '03', '04', '05', '06', '07', '08', '09',
                           '10', '11', '12', '13', '14', '15', '19', '21']
        self.test_list = ['16', '17', '18', '20']

        # ['00', '03', '05', '06', '07', '08', '09']
        # self.model = CameraModel()
        # self.model.focal_length = [7.18856e+02, 7.18856e+02]
        # self.model.principal_point = [6.071928e+02, 1.852157e+02]
        for seq in self.sequence_list:
            odom = odometry(self.root_dir, seq)
            calib = odom.calib
            T_cam02_velo_np = calib.T_cam2_velo  # gt pose from cam02 to velo_lidar (T_cam02_velo: 4x4)
            T_cam03_velo_np = calib.T_cam3_velo
            T_cam02_cam03_np = np.matmul(T_cam02_velo_np, np.linalg.inv(T_cam03_velo_np))
            K_cam2 = calib.K_cam2  # 3x3
            K_cam3 = calib.K_cam3
            # T_cam02_velo = torch.from_numpy(T_cam02_velo_np)
            # T_cam03_velo = torch.from_numpy(T_cam03_velo_np)


            self.K2_list[seq] = K_cam2
            # GT_R = quaternion_from_matrix(T_cam02_velo[:3, :3])
            # GT_T = T_cam02_velo[3:, :3]

            self.K3_list[seq] = K_cam3
            # GT_R = quaternion_from_matrix(T_cam03_velo[:3, :3])
            # GT_T = T_cam03_velo[3:, :3]

            # self.GTs_R[seq] = GT_R  # GT_R = np.array([row['qw'], row['qx'], row['qy'], row['qz']])
            # self.GTs_T[seq] = GT_T  # GT_T = np.array([row['x'], row['y'], row['z']])
            self.GTs_T_cam02_velo[seq] = T_cam02_velo_np  # gt pose from cam02 to velo_lidar (T_cam02_velo: 4x4)
            self.GTs_T_cam03_velo[seq] = T_cam03_velo_np
            self.Gts_cam02_cam03[seq] = T_cam02_cam03_np

            image_list = os.listdir(os.path.join(dataset_dir, 'sequences', seq, 'image_2'))
            image_list.sort()

            for image_name in image_list:
                if not os.path.exists(os.path.join(dataset_dir, 'sequences', seq, 'velodyne',
                                                   str(image_name.split('.')[0]) + '.bin')):
                    continue
                if not os.path.exists(os.path.join(dataset_dir, 'sequences', seq, 'image_2',
                                                   str(image_name.split('.')[0]) + suf)):  # png
                    continue
                if seq == val_sequence and (not split == 'train'):
                    self.all_files.append(os.path.join(seq, image_name.split('.')[0]))
                elif (not seq == val_sequence) and split == 'train' and seq in self.train_list:
                    self.all_files.append(os.path.join(seq, image_name.split('.')[0]))

        self.val_RT = {'left': [], 'right': []}
        if split == 'val' or split == 'test':
            val_RT_file_left = os.path.join(dataset_dir, 'sequences',
                                            f'val_RT_left_seq{val_sequence}_{max_r:.2f}_{max_t:.2f}.csv')
            val_RT_file_right = os.path.join(dataset_dir, 'sequences',
                                             f'val_RT_right_seq{val_sequence}_{max_r:.2f}_{max_t:.2f}.csv')
            if os.path.exists(val_RT_file_left) and os.path.exists(val_RT_file_right):
                print(f'VAL SET: Using this file: {val_RT_file_left} and {val_RT_file_right}')
                df_test_RT_left = pd.read_csv(val_RT_file_left, sep=',')
                df_test_RT_right = pd.read_csv(val_RT_file_right, sep=',')
                for index, row in df_test_RT_left.iterrows():
                    self.val_RT['left'].append(list(row))
                for index, row in df_test_RT_right.iterrows():
                    self.val_RT['right'].append(list(row))
            else:
                print(f'TEST SET - Not found: {val_RT_file_left} or {val_RT_file_right}')
                print("Generating a new one")

                val_RT_file_left = open(val_RT_file_left, 'w')
                val_RT_file_left = csv.writer(val_RT_file_left, delimiter=',')
                val_RT_file_left.writerow(['id', 'tx', 'ty', 'tz', 'rx', 'ry', 'rz'])
                for i in range(len(self.all_files)):
                    rotz = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                    roty = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                    rotx = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                    transl_x = np.random.uniform(-max_t, max_t)
                    transl_y = np.random.uniform(-max_t, max_t)
                    transl_z = np.random.uniform(-max_t, max_t)
                    # transl_z = np.random.uniform(-max_t, min(max_t, 1.))
                    val_RT_file_left.writerow([i, transl_x, transl_y, transl_z,
                                               rotx, roty, rotz])
                    self.val_RT['left'].append([float(i), transl_x, transl_y, transl_z,
                                             rotx, roty, rotz])

                val_RT_file_right = open(val_RT_file_right, 'w')
                val_RT_file_right = csv.writer(val_RT_file_right, delimiter=',')
                val_RT_file_right.writerow(['id', 'tx', 'ty', 'tz', 'rx', 'ry', 'rz'])
                for i in range(len(self.all_files)):
                    rotz = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                    roty = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                    rotx = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                    transl_x = np.random.uniform(-max_t, max_t)
                    transl_y = np.random.uniform(-max_t, max_t)
                    transl_z = np.random.uniform(-max_t, max_t)
                    # transl_z = np.random.uniform(-max_t, min(max_t, 1.))
                    val_RT_file_right.writerow([i, transl_x, transl_y, transl_z,
                                                rotx, roty, rotz])
                    self.val_RT['right'].append([float(i), transl_x, transl_y, transl_z,
                                             rotx, roty, rotz])

            assert len(self.val_RT['left']) == len(self.all_files), "Something wrong with test RTs"

    # def get_ground_truth_poses(self, sequence, frame):
    #     return self.GTs_T[sequence][frame], self.GTs_R[sequence][frame]

    def custom_transform(self, rgb, img_rotation=0., flip=False):
        to_tensor = transforms.ToTensor()
        normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])

        # rgb = crop(rgb)
        if self.split == 'train':
            # color_transform = transforms.ColorJitter(0.1, 0.1, 0.1)
            # color_transform = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3 / 3.14)
            color_transform = transforms.ColorJitter(0.1, 0.1, 0.1)
            rgb = color_transform(rgb)
            if flip:
                rgb = TTF.hflip(rgb)
            rgb = TTF.rotate(rgb, img_rotation)
            # io.imshow(np.array(rgb))
            # io.show()

        rgb = to_tensor(rgb)
        rgb = normalization(rgb)
        return rgb

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        item = self.all_files[idx]
        seq = str(item.split('/')[0])
        rgb_name = str(item.split('/')[1])
        left_img_path = os.path.join(self.root_dir, 'sequences', seq, 'image_2', rgb_name + self.suf)  # png
        right_img_path = os.path.join(self.root_dir, 'sequences', seq, 'image_3', rgb_name + self.suf)  # png
        lidar_path = os.path.join(self.root_dir, 'sequences', seq, 'velodyne', rgb_name + '.bin')
        lidar_scan = np.fromfile(lidar_path, dtype=np.float32)
        pc = lidar_scan.reshape((-1, 4))
        pc_lidar = pc.copy()

        # pc = down_sample(pc)

        # offset = 0.05
        # pc1 = pc[int(pc.shape[0] * (offset / 64)):int(pc.shape[0] * ((offset + 1) / 64)), :]
        # pc_ds = pc1.copy()
        # for i in range(1, 16):
        #     pc_ = pc[int(pc.shape[0] * ((4*i + offset) / 64)):int(pc.shape[0] * ((4*i + offset + 1) / 64)), :]
        #     if i > 0:
        #         pc_ds = np.vstack((pc_ds, pc_))
        #     else:
        #         pc_ds = np.vstack((pc1, pc_))
        # pc = pc_ds

        valid_indices = pc[:, 0] < -3.
        valid_indices = valid_indices | (pc[:, 0] > 3.)
        valid_indices = valid_indices | (pc[:, 1] < -3.)
        valid_indices = valid_indices | (pc[:, 1] > 3.)
        pc = pc[valid_indices].copy()
        pc_org = torch.from_numpy(pc.astype(np.float32))

        # if self.use_reflectance:
        #     reflectance = pc[:, 3].copy()
        #     reflectance = torch.from_numpy(reflectance).float()

        E_RT_left = self.GTs_T_cam02_velo[seq].astype(np.float32)
        E_RT_right = self.GTs_T_cam03_velo[seq].astype(np.float32)
        E_RT_cam2_cam3 = self.Gts_cam02_cam03[seq].astype(np.float32)

        if pc_org.shape[1] == 4 or pc_org.shape[1] == 3:
            pc_org = pc_org.t()
        if pc_org.shape[0] == 3:
            homogeneous = torch.ones(pc_org.shape[1]).unsqueeze(0)
            pc_org = torch.cat((pc_org, homogeneous), 0)
        elif pc_org.shape[0] == 4:
            if not torch.all(pc_org[3, :] == 1.):
                pc_org[3, :] = 1.
        else:
            raise TypeError("Wrong PointCloud shape")

        pc_cam_left = np.matmul(E_RT_left, pc_org.numpy())
        pc_cam_left = pc_cam_left.astype(np.float32).copy()
        pc_cam_left = torch.from_numpy(pc_cam_left)

        pc_cam_right = np.matmul(E_RT_right, pc_org.numpy())
        pc_cam_right = pc_cam_right.astype(np.float32).copy()
        pc_cam_right = torch.from_numpy(pc_cam_right)

        # pc_cam_left = np.matmul(E_RT_left, pc.copy().T)
        # pc_cam_left = pc_cam_left.astype(np.float32).T.copy()
        # pc_in= torch.from_numpy(pc_cam_left)
        #
        # if pc_in.shape[1] == 4 or pc_in.shape[1] == 3:
        #     pc_in = pc_in.t()
        # if pc_in.shape[0] == 3:
        #     homogeneous = torch.ones(pc_in.shape[1]).unsqueeze(0)
        #     pc_in = torch.cat((pc_in, homogeneous), 0)
        # elif pc_in.shape[0] == 4:
        #     if not torch.all(pc_in[3,:] == 1.):
        #         pc_in[3,:] = 1.
        # else:
        #     raise TypeError("Wrong PointCloud shape")
        # pc_cam_left = pc_in
        #
        # pc_cam_right = np.matmul(E_RT_right, pc.copy().T)
        # pc_cam_right = pc_cam_right.astype(np.float32).T.copy()
        # pc_in = torch.from_numpy(pc_cam_right)
        #
        # if pc_in.shape[1] == 4 or pc_in.shape[1] == 3:
        #     pc_in = pc_in.t()
        # if pc_in.shape[0] == 3:
        #     homogeneous = torch.ones(pc_in.shape[1]).unsqueeze(0)
        #     pc_in = torch.cat((pc_in, homogeneous), 0)
        # elif pc_in.shape[0] == 4:
        #     if not torch.all(pc_in[3, :] == 1.):
        #         pc_in[3, :] = 1.
        # else:
        #     raise TypeError("Wrong PointCloud shape")
        # pc_cam_right = pc_in

        h_mirror = False
        # if np.random.rand() > 0.5 and self.split == 'train':
        #     h_mirror = True
        #     pc_in[0, :] *= -1

        img_left = Image.open(left_img_path)
        img_right = Image.open(right_img_path)

        # print(img_path)
        # img = cv2.imread(img_path)
        img_rotation = 0.
        # if self.split == 'train':
        #     img_rotation = np.random.uniform(-5, 5)
        try:
            img_left = self.custom_transform(img_left, img_rotation, h_mirror)
            img_right = self.custom_transform(img_right, img_rotation, h_mirror)
        except OSError:
            new_idx = np.random.randint(0, self.__len__())
            return self.__getitem__(new_idx)

        rotx = {}
        roty = {}
        rotz = {}
        transl_x = {}
        transl_y = {}
        transl_z = {}
        if self.split == 'train':
            max_angle = self.max_r
            # left camera
            rotz['left'] = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            roty['left'] = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            rotx['left'] = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            transl_x['left'] = np.random.uniform(-self.max_t, self.max_t)
            transl_y['left'] = np.random.uniform(-self.max_t, self.max_t)
            transl_z['left'] = np.random.uniform(-self.max_t, self.max_t)

            # right camera
            rotz['right'] = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            roty['right'] = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            rotx['right'] = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            transl_x['right'] = np.random.uniform(-self.max_t, self.max_t)
            transl_y['right'] = np.random.uniform(-self.max_t, self.max_t)
            transl_z['right'] = np.random.uniform(-self.max_t, self.max_t)
            # transl_z = np.random.uniform(-self.max_t, min(self.max_t, 1.))
            initial_RT_left = [transl_x['left'], transl_y['left'], transl_z['left'],
                               rotx['left'], roty['left'], rotz['left']]
            initial_RT_right = [transl_x['right'], transl_y['right'], transl_z['right'],
                                rotx['right'], roty['right'], rotz['right']]
        else:
            # left camera
            initial_RT_left = self.val_RT['left'][idx]
            rotz['left'] = initial_RT_left[6]
            roty['left'] = initial_RT_left[5]
            rotx['left'] = initial_RT_left[4]
            transl_x['left'] = initial_RT_left[1]
            transl_y['left'] = initial_RT_left[2]
            transl_z['left'] = initial_RT_left[3]

            # right camera
            initial_RT_right = self.val_RT['right'][idx]
            rotz['right'] = initial_RT_right[6]
            roty['right'] = initial_RT_right[5]
            rotx['right'] = initial_RT_right[4]
            transl_x['right'] = initial_RT_right[1]
            transl_y['right'] = initial_RT_right[2]
            transl_z['right'] = initial_RT_right[3]

        # 随机设置一定范围内的标定参数扰动值
        # train的时候每次都随机生成,每个epoch使用不同的参数
        # test则在初始化的时候提前设置好,每个epoch都使用相同的参数
        R = mathutils.Euler((rotx['left'], roty['left'], rotz['left']), 'XYZ')
        t = mathutils.Vector((transl_x['left'], transl_y['left'], transl_z['left']))

        R, t = invert_pose(R, t)
        R, t = torch.tensor(R), torch.tensor(t)

        R_dict = {'left': R}
        t_dict = {'left': t}

        R = mathutils.Euler((rotx['right'], roty['right'], rotz['right']), 'XYZ')
        t = mathutils.Vector((transl_x['right'], transl_y['right'], transl_z['right']))

        R, t = invert_pose(R, t)
        R, t = torch.tensor(R), torch.tensor(t)

        R_dict['right'] = R
        t_dict['right'] = t

        # io.imshow(depth_img.numpy(), cmap='jet')
        # io.show()
        calib_left = self.K2_list[seq]
        calib_right = self.K3_list[seq]
        if h_mirror:
            calib_left[2] = (calib_left.shape[2] / 2) * 2 - calib_left[2]
            calib_right[2] = (calib_right.shape[2] / 2) * 2 - calib_right[2]


        # sample = {'rgb': img, 'point_cloud': pc_in, 'calib': calib, 'pc_org': pc_org, 'img_path': img_path,
        #           'tr_error': T, 'rot_error': R, 'seq': int(seq), 'rgb_name': rgb_name, 'item': item,
        #           'extrin': E_RT, 'initial_RT': initial_RT, 'pc_lidar': pc_lidar}
        sample = {
            'rgb_left': img_left, 'rgb_right': img_right, 'pc_cam_left': pc_cam_left, 'pc_cam_right': pc_cam_right,
            'calib_left': calib_left, 'calib_right':calib_right, 'left_path': left_img_path, 'right_path': right_img_path,
            'initial_RT_left': initial_RT_left, 'initial_RT_right': initial_RT_right,
            'tr_error_left': t_dict['left'], 'tr_error_right': t_dict['right'],
            'rot_error_left': R_dict['left'], 'rot_error_right': R_dict['right'],
            'extrin_left': E_RT_left, 'extrin_right': E_RT_right, 'extrin_cam2_cam3': E_RT_cam2_cam3,
            'seq': int(seq), 'rgb_name': rgb_name, 'item': item, 'pc_lidar': pc_lidar, 'pc_org': pc_org
        }

        return sample


class DatasetLidarCameraKitti360(Dataset):

    def __init__(self, dataset_dir, transform=None, augmentation=False, use_reflectance=False, lidar_type='velodyne',
                 max_t=1.5, max_r=15.0, split='val', device='cpu', val_sequence='03', tf_range=1, camera='left'):
        super(DatasetLidarCameraKitti360, self).__init__()
        self.use_reflectance = use_reflectance
        self.maps_folder = ''
        self.device = device
        self.max_r = max_r
        self.max_t = max_t
        self.augmentation = augmentation
        self.root_dir = dataset_dir
        self.transform = transform
        self.split = split
        self.max_depth = 80
        self.tf_range = tf_range
        self.lidar_type = lidar_type
        self.camera = camera

        calib = os.path.join(self.root_dir, 'calibration')
        cam00_velo = os.path.join(calib, 'calib_cam_to_velo.txt')
        sick_velo = os.path.join(calib, 'calib_sick_to_velo.txt')
        cam_calib = os.path.join(calib, 'perspective.txt')
        cam_calib_dict = read_calib_file(cam_calib)
        # R_rect_00 = cam_calib_dict['R_rect_00']
        P_rect_00 = cam_calib_dict['P_rect_00']

        R_rect_00 = np.eye(4)
        R_rect_00[0:3, 0:3] = np.reshape(cam_calib_dict['R_rect_00'], (3, 3))

        # left camera
        # intrinsic of camera00 (cam00_K: 3x3)
        self.cam00_K = np.reshape(P_rect_00, (3, 4))[:, :3]
        T_cam00_velo = np.reshape(np.loadtxt(cam00_velo), (3, 4))
        T_sick_velo = np.reshape(np.loadtxt(sick_velo), (3, 4))
        # gt pose from cam00 to velo_lidar (T_cam00_velo: 4x4)
        # self.GTs_T_cam00_velo = np.vstack([T_cam00_velo, [0, 0, 0, 1]])

        self.GTs_T_cam00_velo = np.vstack([T_cam00_velo, [0, 0, 0, 1]])
        self.GTs_T_sick_velo = np.vstack([T_sick_velo, [0, 0, 0, 1]])
        # Compute the rectified extrinsics from cam0 to camN
        # T0 = np.eye(4)
        # P_rect_00 = np.reshape(P_rect_00, (3, 4))
        # T0[0, 3] = P_rect_00[0, 3] / P_rect_00[0, 0]
        # Compute the velodyne to rectified camera coordinate transforms
        # self.GTs_T_cam00_velo = T0.dot(R_rect_00.dot(self.GTs_T_cam00_velo))

        T_cam00_velo = torch.from_numpy(self.GTs_T_cam00_velo)
        # GT_R = np.array([row['qw'], row['qx'], row['qy'], row['qz']])
        self.GT_R = quaternion_from_matrix(T_cam00_velo[:3, :3])
        # GT_T = np.array([row['x'], row['y'], row['z']])
        self.GT_T = T_cam00_velo[3:, :3]

        # right camera
        # error need dubug
        T_cam0unrect_velo = self.GTs_T_cam00_velo
        filedata = cam_calib_dict

        # Create 3x4 projection matrices
        P_rect_00 = np.reshape(filedata['P_rect_00'], (3, 4))
        P_rect_10 = np.reshape(filedata['P_rect_01'], (3, 4))

        # Create 4x4 matrices from the rectifying rotation matrices
        R_rect_00 = np.eye(4)
        R_rect_00[0:3, 0:3] = np.reshape(filedata['R_rect_00'], (3, 3))
        R_rect_10 = np.eye(4)
        R_rect_10[0:3, 0:3] = np.reshape(filedata['R_rect_01'], (3, 3))

        T0 = np.eye(4)
        T0[0, 3] = P_rect_00[0, 3] / P_rect_00[0, 0]
        T1 = np.eye(4)
        T1[0, 3] = P_rect_10[0, 3] / P_rect_10[0, 0]

        # self.GTs_T_cam00_velo = T1.dot(R_rect_00.dot(T_cam0unrect_velo))
        self.GTs_T_cam01_velo = T1.dot(T_cam0unrect_velo)
        self.cam01_K = P_rect_10[0:3, 0:3]


        # P_rect_01 = cam_calib_dict['P_rect_01']
        # R_rect_01 = np.eye(4)
        # R_rect_01[0:3, 0:3] = np.reshape(cam_calib_dict['R_rect_01'], (3, 3))
        # P_rect_01 = np.reshape(P_rect_01, (3, 4))
        #
        # T1 = np.eye(4)
        # T1[0, 3] = P_rect_01[0, 3] / P_rect_01[0, 0]
        #
        # self.GTs_T_cam01_velo = T1.dot(R_rect_00.dot(T_cam00_velo))
        # self.cam01_K = P_rect_01[0:3, 0:3]

        if self.camera == 'right':
            T_cam01_velo = torch.from_numpy(self.GTs_T_cam01_velo)
            # GT_R = np.array([row['qw'], row['qx'], row['qy'], row['qz']])
            self.GT_R = quaternion_from_matrix(T_cam01_velo[:3, :3])
            # GT_T = np.array([row['x'], row['y'], row['z']])
            self.GT_T = T_cam01_velo[3:, :3]

        self.all_files = []
        val_sequence = '2013_05_28_drive_00' + str(val_sequence) + '_sync'
        self.test_list = ['2013_05_28_drive_0003_sync', '2013_05_28_drive_0007_sync', '2013_05_28_drive_0010_sync']
        self.train_list = ['2013_05_28_drive_0000_sync', '2013_05_28_drive_0002_sync']
        # self.train_list = ['2013_05_28_drive_0000_sync', '2013_05_28_drive_0002_sync',
        #                    '2013_05_28_drive_0004_sync', '2013_05_28_drive_0006_sync', '2013_05_28_drive_0009_sync', ]

        self.sequence_list = ['2013_05_28_drive_0000_sync', '2013_05_28_drive_0002_sync', '2013_05_28_drive_0003_sync',
                              '2013_05_28_drive_0004_sync', '2013_05_28_drive_0005_sync', '2013_05_28_drive_0006_sync',
                              '2013_05_28_drive_0007_sync', '2013_05_28_drive_0009_sync', '2013_05_28_drive_0010_sync']
        self.lidar_dict = {}
        for seq in self.sequence_list:
            image_list = os.listdir(os.path.join(dataset_dir, 'data_2d_raw', seq, 'image_00/data_rect'))
            image_list.sort()

            for image_name in image_list:
                if not os.path.exists(os.path.join(dataset_dir, 'data_3d_raw', seq, 'velodyne_points/data',
                                                   str(image_name.split('.')[0])+'.bin')):
                    continue
                if not os.path.exists(os.path.join(dataset_dir, 'data_2d_raw', seq, 'image_00/data_rect',
                                                   str(image_name.split('.')[0])+'.png')): # png
                    continue
                if seq == val_sequence:
                    self.all_files.append(os.path.join(seq, image_name.split('.')[0]))
                elif (not seq == val_sequence) and split == 'train' and seq in self.train_list:
                    self.all_files.append(os.path.join(seq, image_name.split('.')[0]))

            associate_path = os.path.join(dataset_dir, 'lidar_associate', seq, 'associate.txt')
            with open(associate_path, 'r') as f:
                associate_list = f.readlines()
            as_dict = {}
            for as_file in associate_list:
                as_file_s = as_file.split(' ')
                as_dict[as_file_s[0]] = as_file_s[1][:-1]
            self.lidar_dict[seq] = as_dict

        self.val_RT = []
        if split == 'val' or split == 'test':
            val_RT_file = os.path.join(dataset_dir,
                                       f'val_RT_seq{val_sequence}_{max_r:.2f}_{max_t:.2f}.csv')
            if os.path.exists(val_RT_file):
                print(f'VAL SET: Using this file: {val_RT_file}')
                df_test_RT = pd.read_csv(val_RT_file, sep=',')
                for index, row in df_test_RT.iterrows():
                    self.val_RT.append(list(row))
            else:
                print(f'TEST SET - Not found: {val_RT_file}')
                print("Generating a new one")
                val_RT_file = open(val_RT_file, 'w')
                val_RT_file = csv.writer(val_RT_file, delimiter=',')
                val_RT_file.writerow(['id', 'tx', 'ty', 'tz', 'rx', 'ry', 'rz'])
                for i in range(len(self.all_files)):
                    rotz = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                    roty = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                    rotx = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                    transl_x = np.random.uniform(-max_t, max_t)
                    transl_y = np.random.uniform(-max_t, max_t)
                    transl_z = np.random.uniform(-max_t, max_t)
                    # transl_z = np.random.uniform(-max_t, min(max_t, 1.))
                    val_RT_file.writerow([i, transl_x, transl_y, transl_z,
                                           rotx, roty, rotz])
                    self.val_RT.append([float(i), transl_x, transl_y, transl_z,
                                         rotx, roty, rotz])

            assert len(self.val_RT) == len(self.all_files), "Something wrong with test RTs"

    def custom_transform(self, rgb, img_rotation=0., flip=False):
        to_tensor = transforms.ToTensor()
        normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])

        #rgb = crop(rgb)
        if self.split == 'train':
            color_transform = transforms.ColorJitter(0.1, 0.1, 0.1)
            # color_transform = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3 / 3.14)
            rgb = color_transform(rgb)
            if flip:
                rgb = TTF.hflip(rgb)
            rgb = TTF.rotate(rgb, img_rotation)
            #io.imshow(np.array(rgb))
            #io.show()

        rgb = to_tensor(rgb)
        rgb = normalization(rgb)
        return rgb

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        item = self.all_files[idx]
        seq = str(item.split('/')[0])
        rgb_name = str(item.split('/')[1])
        if self.camera == 'left':
            img_path = os.path.join(self.root_dir, 'data_2d_raw', seq, 'image_00/data_rect', rgb_name+'.png') # png
        else:
            img_path = os.path.join(self.root_dir, 'data_2d_raw', seq, 'image_01/data_rect', rgb_name+'.png') # png

        # velodyne_pointcloud
        if self.lidar_type == 'velodyne':
            velo_path = os.path.join(self.root_dir, 'data_3d_raw', seq, 'velodyne_points/data', rgb_name+'.bin')
            velo_scan = np.fromfile(velo_path, dtype=np.float32)
            pc = velo_scan.reshape((-1, 4))
            valid_indices = pc[:, 0] < -3.
            valid_indices = valid_indices | (pc[:, 0] > 3.)
            valid_indices = valid_indices | (pc[:, 1] < -3.)
            valid_indices = valid_indices | (pc[:, 1] > 3.)
            pc = pc[valid_indices].copy()
            pc_org = torch.from_numpy(pc.astype(np.float32))
            if self.camera == 'left':
                E_RT = np.linalg.inv(self.GTs_T_cam00_velo.astype(np.float32))
            else:
                E_RT = np.linalg.inv(self.GTs_T_cam01_velo.astype(np.float32))

        # sick_pointcloud
        elif self.lidar_type == 'sick':
            lidar_dict = self.lidar_dict[seq]
            sick_path = os.path.join(self.root_dir, 'data_3d_raw', seq, 'sick_points/data', lidar_dict[rgb_name+'.bin'])
            if not os.path.isfile(sick_path):
                raise RuntimeError('%s does not exist!' % sick_path)
            sick_scan = np.fromfile(sick_path, dtype=np.float32)
            sick_scan = np.reshape(sick_scan, [-1, 2])
            sick_scan = np.concatenate([np.zeros_like(sick_scan[:, 0:1]), -sick_scan[:, 0:1], sick_scan[:, 1:2]], axis=1)
            pc_org = torch.from_numpy(sick_scan.astype(np.float32))
            E_RT = np.linalg.inv(np.matmul(self.GTs_T_sick_velo.astype(np.float32),
                             self.GTs_T_cam00_velo.astype(np.float32)))

        # E_RT = np.linalg.inv(self.GTs_T_cam00_velo.astype(np.float32))

        # pc_rot = np.matmul(E_RT, pc.T)
        # pc_rot = pc_rot.astype(np.float).T.copy()
        # pc_in = torch.from_numpy(pc_rot.astype(np.float32))#.float()
        #
        # if pc_in.shape[1] == 4 or pc_in.shape[1] == 3:
        #     pc_in = pc_in.t()
        # if pc_in.shape[0] == 3:
        #     homogeneous = torch.ones(pc_in.shape[1]).unsqueeze(0)
        #     pc_in = torch.cat((pc_in, homogeneous), 0)
        # elif pc_in.shape[0] == 4:
        #     if not torch.all(pc_in[3,:] == 1.):
        #         pc_in[3,:] = 1.
        # else:
        #     raise TypeError("Wrong PointCloud shape")

        if pc_org.shape[1] == 4 or pc_org.shape[1] == 3:
            pc_org = pc_org.t()
        if pc_org.shape[0] == 3:
            homogeneous = torch.ones(pc_org.shape[1]).unsqueeze(0)
            pc_org = torch.cat((pc_org, homogeneous), 0)
        elif pc_org.shape[0] == 4:
            if not torch.all(pc_org[3, :] == 1.):
                pc_org[3, :] = 1.
        else:
            raise TypeError("Wrong PointCloud shape")

        pc_rot = np.matmul(E_RT, pc_org.numpy())
        pc_rot = pc_rot.astype(np.float32).copy()
        pc_in = torch.from_numpy(pc_rot)

        h_mirror = False
        # if np.random.rand() > 0.5 and self.split == 'train':
        #     h_mirror = True
        #     pc_in[0, :] *= -1

        img = Image.open(img_path)
        # print(img_path)
        # img = cv2.imread(img_path)
        img_rotation = 0.
        # if self.split == 'train':
        #     img_rotation = np.random.uniform(-5, 5)
        try:
            img = self.custom_transform(img, img_rotation, h_mirror)
        except OSError:
            new_idx = np.random.randint(0, self.__len__())
            return self.__getitem__(new_idx)

        # Rotate PointCloud for img_rotation
        # if self.split == 'train':
        #     R = mathutils.Euler((radians(img_rotation), 0, 0), 'XYZ')
        #     T = mathutils.Vector((0., 0., 0.))
        #     pc_in = rotate_forward(pc_in, R, T)

        if self.split == 'train':
            max_angle = self.max_r
            rotz = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            roty = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            rotx = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            transl_x = np.random.uniform(-self.max_t, self.max_t)
            transl_y = np.random.uniform(-self.max_t, self.max_t)
            transl_z = np.random.uniform(-self.max_t, self.max_t)
            # transl_z = np.random.uniform(-self.max_t, min(self.max_t, 1.))
            initial_RT = 0
        else:
            if self.tf_range > 1:
                err_idx = idx // self.tf_range
            else:
                err_idx = idx
            initial_RT = self.val_RT[err_idx]
            rotz = initial_RT[6]
            roty = initial_RT[5]
            rotx = initial_RT[4]
            transl_x = initial_RT[1]
            transl_y = initial_RT[2]
            transl_z = initial_RT[3]

        # 随机设置一定范围内的标定参数扰动值
        # train的时候每次都随机生成,每个epoch使用不同的参数
        # test则在初始化的时候提前设置好,每个epoch都使用相同的参数
        R = mathutils.Euler((rotx, roty, rotz), 'XYZ')
        T = mathutils.Vector((transl_x, transl_y, transl_z))

        R, T = invert_pose(R, T)
        R, T = torch.tensor(R), torch.tensor(T)

        # calib = self.cam00_K.copy()
        # cx = calib[0, 2]
        # width = 1240
        #
        # crop_cj = 704
        # cx = cx + float(width-1)/2 - crop_cj
        #
        # calib[0, 2] = cx
        #
        # # calib = get_calib_kitti_odom(int(seq))
        # if h_mirror:
        #     calib[2] = (img.shape[2] / 2)*2 - calib[2]
        #
        # img = img[:, :, 84:1324]

        calib = self.cam00_K
        if h_mirror:
            calib[2] = (img.shape[2] / 2)*2 - calib[2]

        sample = {'rgb': img, 'point_cloud': pc_in, 'calib': calib, 'pc_org': pc_org, 'img_path': img_path,
                  'tr_error': T, 'rot_error': R, 'rgb_name': rgb_name, 'item': item,
                  'extrin': E_RT, 'initial_RT': initial_RT}

        return sample
