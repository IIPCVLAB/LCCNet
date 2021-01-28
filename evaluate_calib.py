# -------------------------------------------------------------------
# Copyright (C) 2020 Harbin Institute of Technology, China
# Author: Xudong Lv (15B901019@hit.edu.cn)
# Released under Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# http://creativecommons.org/licenses/by-nc-sa/4.0/
# -------------------------------------------------------------------

import csv
import random
import open3d as o3

import cv2
import mathutils
# import matplotlib
# matplotlib.use('Qt5Agg')
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
from skimage import io
from tqdm import tqdm
import time

from models.LCCNet import LCCNet

from quaternion_distances import quaternion_distance
from utils import (mat2xyzrpy, merge_inputs, overlay_imgs, quat2mat,
                   quaternion_from_matrix, rotate_back, rotate_forward,
                   tvector2mat)

from torch.utils.data import Dataset
from pykitti import odometry
import pandas as pd
from PIL import Image
from math import radians
from utils import invert_pose
from torchvision import transforms

class DatasetTest(Dataset):

    def __init__(self, dataset_dir, transform=None, augmentation=False, use_reflectance=False,
                 max_t=1.5, max_r=20., split='random', device='cpu', test_sequence='00', est='rot'):
        super(DatasetTest, self).__init__()
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

        self.all_files = []

        seq = test_sequence
        odom = odometry(self.root_dir, seq)
        calib = odom.calib
        T_cam02_velo_np = calib.T_cam2_velo #gt pose from cam02 to velo_lidar (T_cam02_velo: 4x4)
        self.K[seq] = calib.K_cam2 # 3x3
        T_cam02_velo = torch.from_numpy(T_cam02_velo_np)
        GT_R = quaternion_from_matrix(T_cam02_velo[:3, :3])
        GT_T = T_cam02_velo[3:, :3]
        self.GTs_R[seq] = GT_R # GT_R = np.array([row['qw'], row['qx'], row['qy'], row['qz']])
        self.GTs_T[seq] = GT_T # GT_T = np.array([row['x'], row['y'], row['z']])
        self.GTs_T_cam02_velo[seq] = T_cam02_velo_np #gt pose from cam02 to velo_lidar (T_cam02_velo: 4x4)

        image_list = os.listdir(os.path.join(dataset_dir, 'sequences', seq, 'image_2'))
        image_list.sort()

        for image_name in image_list:
            if not os.path.exists(os.path.join(dataset_dir, 'sequences', seq, 'velodyne',
                                               str(image_name.split('.')[0])+'.bin')):
                continue
            if not os.path.exists(os.path.join(dataset_dir, 'sequences', seq, 'image_2',
                                               str(image_name.split('.')[0])+'.png')):
                continue
            self.all_files.append(os.path.join(seq, image_name.split('.')[0]))

        self.test_RT = []
        if split == 'random':
            test_RT_file = os.path.join(dataset_dir, 'sequences',
                                       f'test_RT_seq{test_sequence}_{max_r:.2f}_{max_t:.2f}.csv')
            if os.path.exists(test_RT_file):
                print(f'TEST SET: Using this file: {test_RT_file}')
                df_test_RT = pd.read_csv(test_RT_file, sep=',')
                for index, row in df_test_RT.iterrows():
                    self.test_RT.append(list(row))
            else:
                print(f'TEST SET - Not found: {test_RT_file}')
                print("Generating a new one")
                test_RT_file = open(test_RT_file, 'w')
                test_RT_file = csv.writer(test_RT_file, delimiter=',')
                test_RT_file.writerow(['id', 'tx', 'ty', 'tz', 'rx', 'ry', 'rz'])
                for i in range(len(self.all_files)):
                    rotz = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                    roty = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                    rotx = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                    transl_x = np.random.uniform(-max_t, max_t)
                    transl_y = np.random.uniform(-max_t, max_t)
                    # transl_z = np.random.uniform(-max_t, max_t)
                    transl_z = np.random.uniform(-max_t, min(max_t, 1.))
                    test_RT_file.writerow([i, transl_x, transl_y, transl_z,
                                           rotx, roty, rotz])
                    self.test_RT.append([float(i), transl_x, transl_y, transl_z,
                                         rotx, roty, rotz])

            assert len(self.test_RT) == len(self.all_files), "Something wrong with test RTs"
        elif split == 'constant':
            for i in range(len(self.all_files)):
                if est == 'rot':
                    rotz = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                    roty = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                    rotx = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                    transl_x = max_t
                    transl_y = max_t
                    transl_z = max_t
                elif est == 'tran':
                    rotz = max_r * (3.141592 / 180.0)
                    roty = max_r * (3.141592 / 180.0)
                    rotx = max_r * (3.141592 / 180.0)
                    transl_x = np.random.uniform(-max_t, max_t)
                    transl_y = np.random.uniform(-max_t, max_t)
                    transl_z = np.random.uniform(-max_t, max_t)
                # transl_z = np.random.uniform(-max_t, min(max_t, 1.))
                self.test_RT.append([i, transl_x, transl_y, transl_z,
                                     rotx, roty, rotz])

    def get_ground_truth_poses(self, sequence, frame):
        return self.GTs_T[sequence][frame], self.GTs_R[sequence][frame]

    def custom_transform(self, rgb):
        to_tensor = transforms.ToTensor()
        normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])

        rgb = to_tensor(rgb)
        rgb = normalization(rgb)
        return rgb

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        item = self.all_files[idx]
        seq = str(item.split('/')[0])
        rgb_name = str(item.split('/')[1])
        img_path = os.path.join(self.root_dir, 'sequences', seq, 'image_2', rgb_name+'.png')
        lidar_path = os.path.join(self.root_dir, 'sequences', seq, 'velodyne', rgb_name+'.bin')
        lidar_scan = np.fromfile(lidar_path, dtype=np.float32)
        pc = lidar_scan.reshape((-1, 4))
        valid_indices = pc[:, 0] < -3.
        valid_indices = valid_indices | (pc[:, 0] > 3.)
        valid_indices = valid_indices | (pc[:, 1] < -3.)
        valid_indices = valid_indices | (pc[:, 1] > 3.)
        pc = pc[valid_indices].copy()
        if self.use_reflectance:
            reflectance = pc[:, 3].copy()
            reflectance = torch.from_numpy(reflectance).float()

        RT = self.GTs_T_cam02_velo[seq].astype(np.float32)

        pc_rot = np.matmul(RT, pc.T)
        pc_rot = pc_rot.astype(np.float).T.copy()
        pc_in = torch.from_numpy(pc_rot.astype(np.float32))#.float()

        if pc_in.shape[1] == 4 or pc_in.shape[1] == 3:
            pc_in = pc_in.t()
        if pc_in.shape[0] == 3:
            homogeneous = torch.ones(pc_in.shape[1]).unsqueeze(0)
            pc_in = torch.cat((pc_in, homogeneous), 0)
        elif pc_in.shape[0] == 4:
            if not torch.all(pc_in[3,:] == 1.):
                pc_in[3,:] = 1.
        else:
            raise TypeError("Wrong PointCloud shape")

        img = Image.open(img_path)

        try:
            img = self.custom_transform(img)
        except OSError:
            new_idx = np.random.randint(0, self.__len__())
            return self.__getitem__(new_idx)


        initial_RT = self.test_RT[idx]
        rotz = initial_RT[6]
        roty = initial_RT[5]
        rotx = initial_RT[4]
        transl_x = initial_RT[1]
        transl_y = initial_RT[2]
        transl_z = initial_RT[3]

        R = mathutils.Euler((rotx, roty, rotz), 'XYZ')
        T = mathutils.Vector((transl_x, transl_y, transl_z))

        R, T = invert_pose(R, T)
        R, T = torch.tensor(R), torch.tensor(T)
        calib = self.K[seq]

        sample = {'rgb': img, 'point_cloud': pc_in, 'calib': calib,
                  'tr_error': T, 'rot_error': R, 'seq': int(seq), 'img_path': img_path,
                  'rgb_name': rgb_name + '.png', 'item': item, 'extrin': RT,
                  'initial_RT': initial_RT}

        return sample

# import matplotlib
# matplotlib.rc("font",family='AR PL UMing CN')
plt.rcParams['axes.unicode_minus'] = False
# plt.rc('font',family='Times New Roman')
font_EN = {'family': 'Times New Roman', 'weight': 'normal', 'size': 16}
font_CN = {'family': 'AR PL UMing CN', 'weight': 'normal', 'size': 16}
plt_size = 10.5

ex = Experiment("LCCNet-evaluate-iterative")
ex.captured_out_filter = apply_backspaces_and_linefeeds

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# noinspection PyUnusedLocal
@ex.config
def config():
    dataset = 'kitti/odom'
    data_folder = '/home/wangshuo/Datasets/KITTI/odometry_color/'
    test_sequence = 0
    use_prev_output = False
    max_t = 1.5
    max_r = 20.
    occlusion_kernel = 5
    occlusion_threshold = 3.0
    network = 'Res_f1'
    norm = 'bn'
    show = False
    use_reflectance = False
    weight = None  # List of weights' path, for iterative refinement
    save_name = None
    # Set to True only if you use two network, the first for rotation and the second for translation
    rot_transl_separated = False
    random_initial_pose = False
    save_log = False
    dropout = 0.0
    max_depth = 80.
    iterative_method = 'multi_range' # ['multi_range', 'single_range', 'single']
    output = '../output'
    save_image = False
    outlier_filter = True
    outlier_filter_th = 10
    out_fig_lg = 'EN' # [EN, CN]

weights = [
   './pretrained/kitti_iter1.tar',
   './pretrained/kitti_iter2.tar',
   './pretrained/kitti_iter3.tar',
   './pretrained/kitti_iter4.tar',
   './pretrained/kitti_iter5.tar',
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCH = 1


def _init_fn(worker_id, seed):
    seed = seed + worker_id + EPOCH * 100
    print(f"Init worker {worker_id} with seed {seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_2D_lidar_projection(pcl, cam_intrinsic):
    pcl_xyz = cam_intrinsic @ pcl.T
    pcl_xyz = pcl_xyz.T
    pcl_z = pcl_xyz[:, 2]
    pcl_xyz = pcl_xyz / (pcl_xyz[:, 2, None] + 1e-10)
    pcl_uv = pcl_xyz[:, :2]

    return pcl_uv, pcl_z


def lidar_project_depth(pc_rotated, cam_calib, img_shape):
    pc_rotated = pc_rotated[:3, :].detach().cpu().numpy()
    cam_intrinsic = cam_calib.numpy()
    pcl_uv, pcl_z = get_2D_lidar_projection(pc_rotated.T, cam_intrinsic)
    mask = (pcl_uv[:, 0] > 0) & (pcl_uv[:, 0] < img_shape[1]) & (pcl_uv[:, 1] > 0) & (
            pcl_uv[:, 1] < img_shape[0]) & (pcl_z > 0)
    pcl_uv = pcl_uv[mask]
    pcl_z = pcl_z[mask]
    pcl_uv = pcl_uv.astype(np.uint32)
    pcl_z = pcl_z.reshape(-1, 1)
    depth_img = np.zeros((img_shape[0], img_shape[1], 1))
    depth_img[pcl_uv[:, 1], pcl_uv[:, 0]] = pcl_z
    depth_img = torch.from_numpy(depth_img.astype(np.float32))
    depth_img = depth_img.cuda()
    depth_img = depth_img.permute(2, 0, 1)
    pc_valid = pc_rotated.T[mask]

    return depth_img, pcl_uv, pc_valid


@ex.automain
def main(_config, seed):
    global EPOCH, weights
    if _config['weight'] is not None:
        weights = _config['weight']

    if _config['iterative_method'] == 'single':
        weights = [weights[0]]

    # dataset_class = DatasetLidarCameraKittiOdometry
    dataset_class = DatasetTest
    img_shape = (384, 1280)
    input_size = (256, 512)

    # split = 'test'
    if _config['random_initial_pose']:
        split = 'test_random'

    if _config['test_sequence'] is None:
        raise TypeError('test_sequences cannot be None')
    else:
        if isinstance(_config['test_sequence'], int):
            _config['test_sequence'] = f"{_config['test_sequence']:02d}"
        # dataset_val = dataset_class(_config['data_folder'], max_r=_config['max_r'], max_t=_config['max_t'],
        #                             split=split, use_reflectance=_config['use_reflectance'],
        #                             val_sequence=_config['test_sequence'])
        dataset_val = dataset_class(_config['data_folder'], max_r=_config['max_r'], max_t=_config['max_t'],
                                    split='random', use_reflectance=_config['use_reflectance'],
                                    test_sequence=_config['test_sequence'], est='rot')

    np.random.seed(seed)
    torch.random.manual_seed(seed)

    def init_fn(x):
        return _init_fn(x, seed)

    num_worker = 6
    batch_size = 1

    TestImgLoader = torch.utils.data.DataLoader(dataset=dataset_val,
                                                shuffle=False,
                                                batch_size=batch_size,
                                                num_workers=num_worker,
                                                worker_init_fn=init_fn,
                                                collate_fn=merge_inputs,
                                                drop_last=False,
                                                pin_memory=False)

    print(len(TestImgLoader))

    models = [] # iterative model
    for i in range(len(weights)):
        # network choice and settings
        if _config['network'].startswith('Res'):
            feat = 1
            md = 4
            split = _config['network'].split('_')
            for item in split[1:]:
                if item.startswith('f'):
                    feat = int(item[-1])
                elif item.startswith('md'):
                    md = int(item[2:])
            assert 0 < feat < 7, "Feature Number from PWC have to be between 1 and 6"
            assert 0 < md, "md must be positive"
            model = LCCNet(input_size, use_feat_from=feat, md=md,
                             use_reflectance=_config['use_reflectance'], dropout=_config['dropout'])
        else:
            raise TypeError("Network unknown")

        checkpoint = torch.load(weights[i], map_location='cpu')
        saved_state_dict = checkpoint['state_dict']
        model.load_state_dict(saved_state_dict)
        model = model.to(device)
        model.eval()
        models.append(model)


    if _config['save_log']:
        log_file = f'./results_for_paper/log_seq{_config["test_sequence"]}.csv'
        log_file = open(log_file, 'w')
        log_file = csv.writer(log_file)
        header = ['frame']
        for i in range(len(weights) + 1):
            header += [f'iter{i}_error_t', f'iter{i}_error_r', f'iter{i}_error_x', f'iter{i}_error_y',
                       f'iter{i}_error_z', f'iter{i}_error_r', f'iter{i}_error_p', f'iter{i}_error_y']
        log_file.writerow(header)

    show = _config['show']
    # save image to the output path
    _config['output'] = os.path.join(_config['output'], _config['iterative_method'])
    rgb_path = os.path.join(_config['output'], 'rgb')
    if not os.path.exists(rgb_path):
        os.makedirs(rgb_path)
    depth_path = os.path.join(_config['output'], 'depth')
    if not os.path.exists(depth_path):
        os.makedirs(depth_path)
    input_path = os.path.join(_config['output'], 'input')
    if not os.path.exists(input_path):
        os.makedirs(input_path)
    gt_path = os.path.join(_config['output'], 'gt')
    if not os.path.exists(gt_path):
        os.makedirs(gt_path)
    if _config['out_fig_lg'] == 'EN':
        results_path = os.path.join(_config['output'], 'results_en')
    elif _config['out_fig_lg'] == 'CN':
        results_path = os.path.join(_config['output'], 'results_cn')
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    pred_path = os.path.join(_config['output'], 'pred')
    for it in range(len(weights)):
        if not os.path.exists(os.path.join(pred_path, 'iteration_'+str(it+1))):
            os.makedirs(os.path.join(pred_path, 'iteration_'+str(it+1)))

    # save pointcloud to the output path
    pc_lidar_path = os.path.join(_config['output'], 'pointcloud', 'lidar')
    if not os.path.exists(pc_lidar_path):
        os.makedirs(pc_lidar_path)
    pc_input_path = os.path.join(_config['output'], 'pointcloud', 'input')
    if not os.path.exists(pc_input_path):
        os.makedirs(pc_input_path)
    pc_pred_path = os.path.join(_config['output'], 'pointcloud', 'pred')
    if not os.path.exists(pc_pred_path):
        os.makedirs(pc_pred_path)


    errors_r = []
    errors_t = []
    errors_t2 = []
    errors_xyz = []
    errors_rpy = []
    all_RTs = []
    mis_calib_list = []
    total_time = 0

    prev_tr_error = None
    prev_rot_error = None

    for i in range(len(weights) + 1):
        errors_r.append([])
        errors_t.append([])
        errors_t2.append([])
        errors_rpy.append([])

    for batch_idx, sample in enumerate(tqdm(TestImgLoader)):
        N = 100 # 500
        # if batch_idx > 200:
        #    break

        log_string = [str(batch_idx)]

        lidar_input = []
        rgb_input = []
        lidar_gt = []
        shape_pad_input = []
        real_shape_input = []
        pc_rotated_input = []
        RTs = []
        shape_pad = [0, 0, 0, 0]
        outlier_filter = False

        if batch_idx == 0 or not _config['use_prev_output']:
            # Qui dare posizione di input del frame corrente rispetto alla GT
            sample['tr_error'] = sample['tr_error'].cuda()
            sample['rot_error'] = sample['rot_error'].cuda()
        else:
            sample['tr_error'] = prev_tr_error
            sample['rot_error'] = prev_rot_error

        for idx in range(len(sample['rgb'])):
            # ProjectPointCloud in RT-pose
            real_shape = [sample['rgb'][idx].shape[1], sample['rgb'][idx].shape[2], sample['rgb'][idx].shape[0]]

            sample['point_cloud'][idx] = sample['point_cloud'][idx].cuda()  # 变换到相机坐标系下的激光雷达点云
            pc_lidar = sample['point_cloud'][idx].clone()

            if _config['max_depth'] < 80.:
                pc_lidar = pc_lidar[:, pc_lidar[0, :] < _config['max_depth']].clone()

            depth_gt, uv_gt, pc_gt_valid = lidar_project_depth(pc_lidar, sample['calib'][idx], real_shape)  # image_shape
            depth_gt /= _config['max_depth']

            if _config['save_image']:
                # save the Lidar pointcloud
                pcl_lidar = o3.PointCloud()
                pc_lidar = pc_lidar.detach().cpu().numpy()
                pcl_lidar.points = o3.Vector3dVector(pc_lidar.T[:, :3])

                # o3.draw_geometries(downpcd)
                o3.write_point_cloud(pc_lidar_path + '/{}.pcd'.format(batch_idx), pcl_lidar)


            R = quat2mat(sample['rot_error'][idx])
            T = tvector2mat(sample['tr_error'][idx])
            RT_inv = torch.mm(T, R)
            RT = RT_inv.clone().inverse()

            pc_rotated = rotate_back(sample['point_cloud'][idx], RT_inv)  # Pc` = RT * Pc

            if _config['max_depth'] < 80.:
                pc_rotated = pc_rotated[:, pc_rotated[0, :] < _config['max_depth']].clone()

            depth_img, uv_input, pc_input_valid = lidar_project_depth(pc_rotated, sample['calib'][idx], real_shape)  # image_shape
            depth_img /= _config['max_depth']

            if _config['outlier_filter'] and uv_input.shape[0] <= _config['outlier_filter_th']:
                outlier_filter = True
            else:
                outlier_filter = False

            if _config['save_image']:
                # save the RGB input pointcloud
                img = cv2.imread(sample['img_path'][0])
                R = img[uv_input[:, 1], uv_input[:, 0], 0] / 255
                G = img[uv_input[:, 1], uv_input[:, 0], 1] / 255
                B = img[uv_input[:, 1], uv_input[:, 0], 2] / 255
                pcl_input = o3.PointCloud()
                pcl_input.points = o3.Vector3dVector(pc_input_valid[:, :3])
                pcl_input.colors = o3.Vector3dVector(np.vstack((R, G, B)).T)

                # o3.draw_geometries(downpcd)
                o3.write_point_cloud(pc_input_path + '/{}.pcd'.format(batch_idx), pcl_input)

            # PAD ONLY ON RIGHT AND BOTTOM SIDE
            rgb = sample['rgb'][idx].cuda()
            shape_pad = [0, 0, 0, 0]

            shape_pad[3] = (img_shape[0] - rgb.shape[1])  # // 2
            shape_pad[1] = (img_shape[1] - rgb.shape[2])  # // 2 + 1

            rgb = F.pad(rgb, shape_pad)
            depth_img = F.pad(depth_img, shape_pad)
            depth_gt = F.pad(depth_gt, shape_pad)

            rgb_input.append(rgb)
            lidar_input.append(depth_img)
            lidar_gt.append(depth_gt)
            real_shape_input.append(real_shape)
            shape_pad_input.append(shape_pad)
            pc_rotated_input.append(pc_rotated)
            RTs.append(RT)

        if outlier_filter:
            continue

        lidar_input = torch.stack(lidar_input)
        rgb_input = torch.stack(rgb_input)
        rgb_resize = F.interpolate(rgb_input, size=[256, 512], mode="bilinear")
        lidar_resize = F.interpolate(lidar_input, size=[256, 512], mode="bilinear")


        if _config['save_image']:
            out0 = overlay_imgs(rgb_input[0], lidar_input)
            out0 = out0[:376, :1241, :]
            cv2.imwrite(os.path.join(input_path, sample['rgb_name'][0]), out0[:, :, [2, 1, 0]]*255)
            out1 = overlay_imgs(rgb_input[0], lidar_gt[0].unsqueeze(0))
            out1 = out1[:376, :1241, :]
            cv2.imwrite(os.path.join(gt_path, sample['rgb_name'][0]), out1[:, :, [2, 1, 0]]*255)

            depth_img = depth_img.detach().cpu().numpy()
            depth_img = (depth_img / np.max(depth_img)) * 255
            cv2.imwrite(os.path.join(depth_path, sample['rgb_name'][0]), depth_img[0, :376, :1241])

        if show:
            out0 = overlay_imgs(rgb_input[0], lidar_input)
            out1 = overlay_imgs(rgb_input[0], lidar_gt[0].unsqueeze(0))
            cv2.imshow("INPUT", out0[:, :, [2, 1, 0]])
            cv2.imshow("GT", out1[:, :, [2, 1, 0]])
            cv2.waitKey(1)

        rgb = rgb_input.to(device)
        lidar = lidar_input.to(device)
        rgb_resize = rgb_resize.to(device)
        lidar_resize = lidar_resize.to(device)

        target_transl = sample['tr_error'].to(device)
        target_rot = sample['rot_error'].to(device)

        # the initial calibration errors before sensor calibration
        RT1 = RTs[0]
        mis_calib = torch.stack(sample['initial_RT'])[1:]
        mis_calib_list.append(mis_calib)

        T_composed = RT1[:3, 3]
        R_composed = quaternion_from_matrix(RT1)
        errors_t[0].append(T_composed.norm().item())
        errors_t2[0].append(T_composed)
        errors_r[0].append(quaternion_distance(R_composed.unsqueeze(0),
                                               torch.tensor([1., 0., 0., 0.], device=R_composed.device).unsqueeze(0),
                                               R_composed.device))
        # rpy_error = quaternion_to_tait_bryan(R_composed)
        rpy_error = mat2xyzrpy(RT1)[3:]

        rpy_error *= (180.0 / 3.141592)
        errors_rpy[0].append(rpy_error)
        log_string += [str(errors_t[0][-1]), str(errors_r[0][-1]), str(errors_t2[0][-1][0].item()),
                       str(errors_t2[0][-1][1].item()), str(errors_t2[0][-1][2].item()),
                       str(errors_rpy[0][-1][0].item()), str(errors_rpy[0][-1][1].item()),
                       str(errors_rpy[0][-1][2].item())]

        # if batch_idx == 0.:
        #     print(f'Initial T_erorr: {errors_t[0]}')
        #     print(f'Initial R_erorr: {errors_r[0]}')
        start = 0
        # t1 = time.time()

        # Run model
        with torch.no_grad():
            for iteration in range(start, len(weights)):
                # Run the i-th network
                t1 = time.time()
                if _config['iterative_method'] == 'single_range' or _config['iterative_method'] == 'single':
                    T_predicted, R_predicted = models[0](rgb_resize, lidar_resize)
                elif _config['iterative_method'] == 'multi_range':
                    T_predicted, R_predicted = models[iteration](rgb_resize, lidar_resize)
                run_time = time.time() - t1

                if _config['rot_transl_separated'] and iteration == 0:
                    T_predicted = torch.tensor([[0., 0., 0.]], device='cuda')
                if _config['rot_transl_separated'] and iteration == 1:
                    R_predicted = torch.tensor([[1., 0., 0., 0.]], device='cuda')

                # Project the points in the new pose predicted by the i-th network
                R_predicted = quat2mat(R_predicted[0])
                T_predicted = tvector2mat(T_predicted[0])
                RT_predicted = torch.mm(T_predicted, R_predicted)
                RTs.append(torch.mm(RTs[iteration], RT_predicted)) # inv(H_gt)*H_pred_1*H_pred_2*.....H_pred_n
                if iteration == 0:
                    rotated_point_cloud = pc_rotated_input[0]
                else:
                    rotated_point_cloud = rotated_point_cloud

                rotated_point_cloud = rotate_forward(rotated_point_cloud, RT_predicted) # H_pred*X_init

                depth_img_pred, uv_pred, pc_pred_valid = lidar_project_depth(rotated_point_cloud, sample['calib'][0], real_shape_input[0]) # image_shape
                depth_img_pred /= _config['max_depth']
                depth_pred = F.pad(depth_img_pred, shape_pad_input[0])
                lidar = depth_pred.unsqueeze(0)
                lidar_resize = F.interpolate(lidar, size=[256, 512], mode="bilinear")

                if iteration == len(weights)-1 and _config['save_image']:
                    # save the RGB pointcloud
                    img = cv2.imread(sample['img_path'][0])
                    R = img[uv_pred[:, 1], uv_pred[:, 0], 0] / 255
                    G = img[uv_pred[:, 1], uv_pred[:, 0], 1] / 255
                    B = img[uv_pred[:, 1], uv_pred[:, 0], 2] / 255
                    pcl_pred = o3.PointCloud()
                    pcl_pred.points = o3.Vector3dVector(pc_pred_valid[:, :3])
                    pcl_pred.colors = o3.Vector3dVector(np.vstack((R, G, B)).T)

                    # o3.draw_geometries(downpcd)
                    o3.write_point_cloud(pc_pred_path + '/{}.pcd'.format(batch_idx), pcl_pred)


                if _config['save_image']:
                    out2 = overlay_imgs(rgb_input[0], lidar)
                    out2 = out2[:376, :1241, :]
                    cv2.imwrite(os.path.join(os.path.join(pred_path, 'iteration_'+str(iteration+1)),
                                             sample['rgb_name'][0]), out2[:, :, [2, 1, 0]]*255)
                if show:
                    out2 = overlay_imgs(rgb_input[0], lidar)
                    cv2.imshow(f'Pred_Iter_{iteration}', out2[:, :, [2, 1, 0]])
                    cv2.waitKey(1)

                # inv(H_init)*H_pred
                T_composed = RTs[iteration + 1][:3, 3]
                R_composed = quaternion_from_matrix(RTs[iteration + 1])
                errors_t[iteration + 1].append(T_composed.norm().item())
                errors_t2[iteration + 1].append(T_composed)
                errors_r[iteration + 1].append(quaternion_distance(R_composed.unsqueeze(0),
                                                                   torch.tensor([1., 0., 0., 0.], device=R_composed.device).unsqueeze(0),
                                                                   R_composed.device))

                # rpy_error = quaternion_to_tait_bryan(R_composed)
                rpy_error = mat2xyzrpy(RTs[iteration + 1])[3:]
                rpy_error *= (180.0 / 3.141592)
                errors_rpy[iteration + 1].append(rpy_error)
                log_string += [str(errors_t[iteration + 1][-1]), str(errors_r[iteration + 1][-1]),
                               str(errors_t2[iteration + 1][-1][0].item()), str(errors_t2[iteration + 1][-1][1].item()),
                               str(errors_t2[iteration + 1][-1][2].item()), str(errors_rpy[iteration + 1][-1][0].item()),
                               str(errors_rpy[iteration + 1][-1][1].item()), str(errors_rpy[iteration + 1][-1][2].item())]

        # run_time = time.time() - t1
        total_time += run_time

        # final calibration error
        all_RTs.append(RTs[-1])
        prev_RT = RTs[-1].inverse()
        prev_tr_error = prev_RT[:3, 3].unsqueeze(0)
        prev_rot_error = quaternion_from_matrix(prev_RT).unsqueeze(0)

        if _config['save_log']:
            log_file.writerow(log_string)

    # Yaw（偏航）：欧拉角向量的y轴
    # Pitch（俯仰）：欧拉角向量的x轴
    # Roll（翻滚）： 欧拉角向量的z轴
    # mis_calib_input[transl_x, transl_y, transl_z, rotx, roty, rotz] Nx6
    mis_calib_input = torch.stack(mis_calib_list)[:, :, 0]

    if _config['save_log']:
        log_file.close()
    print("Iterative refinement: ")
    for i in range(len(weights) + 1):
        errors_r[i] = torch.tensor(errors_r[i]).abs() * (180.0 / 3.141592)
        errors_t[i] = torch.tensor(errors_t[i]).abs() * 100

        for k in range(len(errors_rpy[i])):
            # errors_rpy[i][k] = torch.tensor(errors_rpy[i][k])
            # errors_t2[i][k] = torch.tensor(errors_t2[i][k]) * 100
            errors_rpy[i][k] = errors_rpy[i][k].clone().detach().abs()
            errors_t2[i][k] = errors_t2[i][k].clone().detach().abs() * 100

        print(f"Iteration {i}: \tMean Translation Error: {errors_t[i].mean():.4f} cm "
              f"     Mean Rotation Error: {errors_r[i].mean():.4f} °")
        print(f"Iteration {i}: \tMedian Translation Error: {errors_t[i].median():.4f} cm "
              f"     Median Rotation Error: {errors_r[i].median():.4f} °")
        print(f"Iteration {i}: \tStd. Translation Error: {errors_t[i].std():.4f} cm "
              f"     Std. Rotation Error: {errors_r[i].std():.4f} °\n")

        # translation xyz
        print(f"Iteration {i}: \tMean Translation X Error: {errors_t2[i][0].mean():.4f} cm "
              f"     Median Translation X Error: {errors_t2[i][0].median():.4f} cm "
              f"     Std. Translation X Error: {errors_t2[i][0].std():.4f} cm ")
        print(f"Iteration {i}: \tMean Translation Y Error: {errors_t2[i][1].mean():.4f} cm "
              f"     Median Translation Y Error: {errors_t2[i][1].median():.4f} cm "
              f"     Std. Translation Y Error: {errors_t2[i][1].std():.4f} cm ")
        print(f"Iteration {i}: \tMean Translation Z Error: {errors_t2[i][2].mean():.4f} cm "
              f"     Median Translation Z Error: {errors_t2[i][2].median():.4f} cm "
              f"     Std. Translation Z Error: {errors_t2[i][2].std():.4f} cm \n")

        # rotation rpy
        print(f"Iteration {i}: \tMean Rotation Roll Error: {errors_rpy[i][0].mean(): .4f} °"
              f"     Median Rotation Roll Error: {errors_rpy[i][0].median():.4f} °"
              f"     Std. Rotation Roll Error: {errors_rpy[i][0].std():.4f} °")
        print(f"Iteration {i}: \tMean Rotation Pitch Error: {errors_rpy[i][1].mean(): .4f} °"
              f"     Median Rotation Pitch Error: {errors_rpy[i][1].median():.4f} °"
              f"     Std. Rotation Pitch Error: {errors_rpy[i][1].std():.4f} °")
        print(f"Iteration {i}: \tMean Rotation Yaw Error: {errors_rpy[i][2].mean(): .4f} °"
              f"     Median Rotation Yaw Error: {errors_rpy[i][2].median():.4f} °"
              f"     Std. Rotation Yaw Error: {errors_rpy[i][2].std():.4f} °\n")


        with open(os.path.join(_config['output'], 'results.txt'),
                  'a', encoding='utf-8') as f:
            f.write(f"Iteration {i}: \n")
            f.write("Translation Error && Rotation Error:\n")
            f.write(f"Iteration {i}: \tMean Translation Error: {errors_t[i].mean():.4f} cm "
                    f"     Mean Rotation Error: {errors_r[i].mean():.4f} °\n")
            f.write(f"Iteration {i}: \tMedian Translation Error: {errors_t[i].median():.4f} cm "
                    f"     Median Rotation Error: {errors_r[i].median():.4f} °\n")
            f.write(f"Iteration {i}: \tStd. Translation Error: {errors_t[i].std():.4f} cm "
                    f"     Std. Rotation Error: {errors_r[i].std():.4f} °\n\n")

            # translation xyz
            f.write("Translation Error XYZ:\n")
            f.write(f"Iteration {i}: \tMean Translation X Error: {errors_t2[i][0].mean():.4f} cm "
                    f"     Median Translation X Error: {errors_t2[i][0].median():.4f} cm "
                    f"     Std. Translation X Error: {errors_t2[i][0].std():.4f} cm \n")
            f.write(f"Iteration {i}: \tMean Translation Y Error: {errors_t2[i][1].mean():.4f} cm "
                    f"     Median Translation Y Error: {errors_t2[i][1].median():.4f} cm "
                    f"     Std. Translation Y Error: {errors_t2[i][1].std():.4f} cm \n")
            f.write(f"Iteration {i}: \tMean Translation Z Error: {errors_t2[i][2].mean():.4f} cm "
                    f"     Median Translation Z Error: {errors_t2[i][2].median():.4f} cm "
                    f"     Std. Translation Z Error: {errors_t2[i][2].std():.4f} cm \n\n")

            # rotation rpy
            f.write("Rotation Error RPY:\n")
            f.write(f"Iteration {i}: \tMean Rotation Roll Error: {errors_rpy[i][0].mean(): .4f} °"
                    f"     Median Rotation Roll Error: {errors_rpy[i][0].median():.4f} °"
                    f"     Std. Rotation Roll Error: {errors_rpy[i][0].std():.4f} °\n")
            f.write(f"Iteration {i}: \tMean Rotation Pitch Error: {errors_rpy[i][1].mean(): .4f} °"
                    f"     Median Rotation Pitch Error: {errors_rpy[i][1].median():.4f} °"
                    f"     Std. Rotation Pitch Error: {errors_rpy[i][1].std():.4f} °\n")
            f.write(f"Iteration {i}: \tMean Rotation Yaw Error: {errors_rpy[i][2].mean(): .4f} °"
                    f"     Median Rotation Yaw Error: {errors_rpy[i][2].median():.4f} °"
                    f"     Std. Rotation Yaw Error: {errors_rpy[i][2].std():.4f} °\n\n\n")

    for i in range(len(errors_t2)):
        errors_t2[i] = torch.stack(errors_t2[i]).abs() / 100
        errors_rpy[i] = torch.stack(errors_rpy[i]).abs()

    # mis_calib_input
    # t_x = mis_calib_input[:, 0]
    # t_y = mis_calib_input[:, 1]
    # t_z = mis_calib_input[:, 2]
    # r_roll = mis_calib_input[:, 5]
    # r_pitch = mis_calib_input[:, 3]
    # r_yaw = mis_calib_input[:, 4]

    # plot_error
    # plot_x = errors_t2[:, 0]
    # plot_y = errors_t2[:, 1]
    # plot_z = errors_t2[:, 2]
    # plot_roll = errors_rpy[:, 0]
    # plot_pitch = errors_rpy[:, 1]
    # plot_yaw = errors_rpy[:, 2]

    # translation error
    # fig = plt.figure(figsize=(6, 3))  # 设置图大小 figsize=(6,3)
    # plt.title('Calibration Translation Error')
    plot_x = np.zeros((mis_calib_input.shape[0], 2))
    plot_x[:, 0] = mis_calib_input[:, 0].cpu().numpy()
    plot_x[:, 1] = errors_t2[-1][:, 0].cpu().numpy()
    plot_x = plot_x[np.lexsort(plot_x[:, ::-1].T)]

    plot_y = np.zeros((mis_calib_input.shape[0], 2))
    plot_y[:, 0] = mis_calib_input[:, 1].cpu().numpy()
    plot_y[:, 1] = errors_t2[-1][:, 1].cpu().numpy()
    plot_y = plot_y[np.lexsort(plot_y[:, ::-1].T)]

    plot_z = np.zeros((mis_calib_input.shape[0], 2))
    plot_z[:, 0] = mis_calib_input[:, 2].cpu().numpy()
    plot_z[:, 1] = errors_t2[-1][:, 2].cpu().numpy()
    plot_z = plot_z[np.lexsort(plot_z[:, ::-1].T)]

    N_interval = plot_x.shape[0] // N
    plot_x = plot_x[::N_interval]
    plot_y = plot_y[::N_interval]
    plot_z = plot_z[::N_interval]

    plt.plot(plot_x[:, 0], plot_x[:, 1], c='red', label='X')
    plt.plot(plot_y[:, 0], plot_y[:, 1], c='blue', label='Y')
    plt.plot(plot_z[:, 0], plot_z[:, 1], c='green', label='Z')
    # plt.legend(loc='best')

    if _config['out_fig_lg'] == 'EN':
        plt.xlabel('Miscalibration (m)', font_EN)
        plt.ylabel('Absolute Error (m)', font_EN)
        plt.legend(loc='best', prop=font_EN)
    elif _config['out_fig_lg'] == 'CN':
        plt.xlabel('初始标定外参偏差/米', font_CN)
        plt.ylabel('绝对误差/米', font_CN)
        plt.legend(loc='best', prop=font_CN)

    plt.xticks(fontproperties='Times New Roman', size=plt_size)
    plt.yticks(fontproperties='Times New Roman', size=plt_size)

    plt.savefig(os.path.join(results_path, 'xyz_plot.png'))
    plt.close('all')

    errors_t = errors_t[-1].numpy()
    errors_t = np.sort(errors_t, axis=0)[:-10] # 去掉一些异常值
    # plt.title('Calibration Translation Error Distribution')
    plt.hist(errors_t / 100, bins=50)
    # ax = plt.gca()
    # ax.set_xlabel('Absolute Translation Error (m)')
    # ax.set_ylabel('Number of instances')
    # ax.set_xticks([0.00, 0.25, 0.00, 0.25, 0.50])

    if _config['out_fig_lg'] == 'EN':
        plt.xlabel('Absolute Translation Error (m)', font_EN)
        plt.ylabel('Number of instances', font_EN)
    elif _config['out_fig_lg'] == 'CN':
        plt.xlabel('绝对平移误差/米', font_CN)
        plt.ylabel('实验序列数目/个', font_CN)
    plt.xticks(fontproperties='Times New Roman', size=plt_size)
    plt.yticks(fontproperties='Times New Roman', size=plt_size)

    plt.savefig(os.path.join(results_path, 'translation_error_distribution.png'))
    plt.close('all')

    # rotation error
    # fig = plt.figure(figsize=(6, 3))  # 设置图大小 figsize=(6,3)
    # plt.title('Calibration Rotation Error')
    plot_pitch = np.zeros((mis_calib_input.shape[0], 2))
    plot_pitch[:, 0] = mis_calib_input[:, 3].cpu().numpy() * (180.0 / 3.141592)
    plot_pitch[:, 1] = errors_rpy[-1][:, 1].cpu().numpy()
    plot_pitch = plot_pitch[np.lexsort(plot_pitch[:, ::-1].T)]

    plot_yaw = np.zeros((mis_calib_input.shape[0], 2))
    plot_yaw[:, 0] = mis_calib_input[:, 4].cpu().numpy() * (180.0 / 3.141592)
    plot_yaw[:, 1] = errors_rpy[-1][:, 2].cpu().numpy()
    plot_yaw = plot_yaw[np.lexsort(plot_yaw[:, ::-1].T)]

    plot_roll = np.zeros((mis_calib_input.shape[0], 2))
    plot_roll[:, 0] = mis_calib_input[:, 5].cpu().numpy() * (180.0 / 3.141592)
    plot_roll[:, 1] = errors_rpy[-1][:, 0].cpu().numpy()
    plot_roll = plot_roll[np.lexsort(plot_roll[:, ::-1].T)]

    N_interval = plot_roll.shape[0] // N
    plot_pitch = plot_pitch[::N_interval]
    plot_yaw = plot_yaw[::N_interval]
    plot_roll = plot_roll[::N_interval]

    # Yaw（偏航）：欧拉角向量的y轴
    # Pitch（俯仰）：欧拉角向量的x轴
    # Roll（翻滚）： 欧拉角向量的z轴

    if _config['out_fig_lg'] == 'EN':
        plt.plot(plot_yaw[:, 0], plot_yaw[:, 1], c='red', label='Yaw(Y)')
        plt.plot(plot_pitch[:, 0], plot_pitch[:, 1], c='blue', label='Pitch(X)')
        plt.plot(plot_roll[:, 0], plot_roll[:, 1], c='green', label='Roll(Z)')
        plt.xlabel('Miscalibration (°)', font_EN)
        plt.ylabel('Absolute Error (°)', font_EN)
        plt.legend(loc='best', prop=font_EN)
    elif _config['out_fig_lg'] == 'CN':
        plt.plot(plot_yaw[:, 0], plot_yaw[:, 1], c='red', label='偏航角')
        plt.plot(plot_pitch[:, 0], plot_pitch[:, 1], c='blue', label='俯仰角')
        plt.plot(plot_roll[:, 0], plot_roll[:, 1], c='green', label='翻滚角')
        plt.xlabel('初始标定外参偏差/度', font_CN)
        plt.ylabel('绝对误差/度', font_CN)
        plt.legend(loc='best', prop=font_CN)

    plt.xticks(fontproperties='Times New Roman', size=plt_size)
    plt.yticks(fontproperties='Times New Roman', size=plt_size)
    plt.savefig(os.path.join(results_path, 'rpy_plot.png'))
    plt.close('all')

    errors_r = errors_r[-1].numpy()
    errors_r = np.sort(errors_r, axis=0)[:-10] # 去掉一些异常值
    # np.savetxt('rot_error.txt', arr_, fmt='%0.8f')
    # print('max rotation_error: {}'.format(max(errors_r)))
    # plt.title('Calibration Rotation Error Distribution')
    plt.hist(errors_r, bins=50)
    #plt.xlim([0, 1.5])  # x轴边界
    #plt.xticks([0.0, 0.3, 0.6, 0.9, 1.2, 1.5])  # 设置x刻度
    # ax = plt.gca()

    if _config['out_fig_lg'] == 'EN':
        plt.xlabel('Absolute Rotation Error (°)', font_EN)
        plt.ylabel('Number of instances', font_EN)
    elif _config['out_fig_lg'] == 'CN':
        plt.xlabel('绝对旋转误差/度', font_CN)
        plt.ylabel('实验序列数目/个', font_CN)
    plt.xticks(fontproperties='Times New Roman', size=plt_size)
    plt.yticks(fontproperties='Times New Roman', size=plt_size)
    plt.savefig(os.path.join(results_path, 'rotation_error_distribution.png'))
    plt.close('all')


    if _config["save_name"] is not None:
        torch.save(torch.stack(errors_t).cpu().numpy(), f'./results_for_paper/{_config["save_name"]}_errors_t')
        torch.save(torch.stack(errors_r).cpu().numpy(), f'./results_for_paper/{_config["save_name"]}_errors_r')
        torch.save(torch.stack(errors_t2).cpu().numpy(), f'./results_for_paper/{_config["save_name"]}_errors_t2')
        torch.save(torch.stack(errors_rpy).cpu().numpy(), f'./results_for_paper/{_config["save_name"]}_errors_rpy')

    avg_time = total_time / len(TestImgLoader)
    print("average runing time on {} iteration: {} s".format(len(weights), avg_time))
    print("End!")


