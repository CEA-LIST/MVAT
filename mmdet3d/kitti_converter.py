# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict
from pathlib import Path
import os
import mmcv
import mmengine
import numpy as np
from mmdet3d.structures.ops import box_np_ops

class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 
    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 
    'barrier'
]

def get_object_centric_info(data_path,
                            image_ids,
                            relative_path=True):
    """
    Custom info gatherer for the Object-Centric dataset.
    """
    data_path = Path(data_path)
    infos = []

    for idx, img_id in enumerate(mmengine.track_iter_progress(image_ids)):
        info = dict()
        info['image'] = dict(
            image_idx=idx, 
            image_path=None, 
            image_shape=None,
        )
        
        # Point Cloud Info
        if relative_path:
            v_path = Path('velodyne') / f'{img_id}.bin'
        else:
            v_path = data_path / 'velodyne' / f'{img_id}.bin'
        
        info['point_cloud'] = dict(
            velodyne_path=str(v_path),
            num_features=4 
        )

        # --- CALIBRATION ---
        # 1. Standard Identity matrices for compatibility
        rect = np.eye(4)
        Trv2c = np.eye(4)
        P2 = np.eye(4) 

        # 2. Load Frame-Specific Projection Matrices
        # Format in file: frame_idx p00 p01 ... p33
        frame_calibs = dict()
        calib_file = data_path / 'calib' / f'{img_id}.txt'
        
        if calib_file.exists():
            with open(calib_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) < 17: continue 
                    
                    f_idx = int(parts[0])
                    mat_vals = [float(x) for x in parts[1:]]
                    mat = np.array(mat_vals).reshape(4, 4)
                    frame_calibs[f_idx] = mat
        
        info['calib'] = dict(
            R0_rect=rect,
            Tr_velo_to_cam=Trv2c,
            P2=P2,
            Tr_imu_to_velo=np.eye(4),
            frame_calibs=frame_calibs 
        )

        # --- ANNOTATION ---
        label_path = data_path / 'label_2' / f'{img_id}.txt'
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        content = [line.strip().split(' ') for line in lines]
        
        annos = dict()
        annos['name'] = np.array([x[0] for x in content])
        annos['truncated'] = np.array([float(x[1]) for x in content])
        annos['occluded'] = np.array([int(x[2]) for x in content])
        annos['alpha'] = np.array([float(x[3]) for x in content])
        annos['bbox'] = np.array([[float(x[4]), float(x[5]), float(x[6]), float(x[7])] for x in content])
        annos['dimensions'] = np.array([[float(x[8]), float(x[9]), float(x[10])] for x in content])
        annos['location'] = np.array([[float(x[11]), float(x[12]), float(x[13])] for x in content])
        annos['rotation_y'] = np.array([float(x[14]) for x in content])
        annos['difficulty'] = np.zeros(len(annos['name']), dtype=np.int32)
        annos['index'] = np.arange(len(annos['name']), dtype=np.int32)

        info['annos'] = annos
        infos.append(info)

    return infos

class _NumPointsInGTCalculater:
    """Calculate the number of points inside the ground truth box."""

    def __init__(self,
                 data_path,
                 relative_path,
                 remove_outside=False, 
                 num_features=4,
                 num_worker=8) -> None:
        self.data_path = data_path
        self.relative_path = relative_path
        self.remove_outside = remove_outside
        self.num_features = num_features
        self.num_worker = num_worker

    def calculate_single(self, info):
        pc_info = info['point_cloud']
        calib = info['calib']
        
        if self.relative_path:
            v_path = str(Path(self.data_path) / pc_info['velodyne_path'])
        else:
            v_path = pc_info['velodyne_path']
            
        points_v = np.fromfile(
            v_path, dtype=np.float32,
            count=-1).reshape([-1, self.num_features])
            
        rect = calib['R0_rect']
        Trv2c = calib['Tr_velo_to_cam']

        annos = info['annos']
        num_obj = len([n for n in annos['name'] if n != 'DontCare'])
        
        dims = annos['dimensions'][:num_obj]
        loc = annos['location'][:num_obj]
        rots = annos['rotation_y'][:num_obj]
        
        gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)
        gt_boxes_lidar = box_np_ops.box_camera_to_lidar(gt_boxes_camera, rect, Trv2c)
        
        indices = box_np_ops.points_in_rbbox(points_v[:, :3], gt_boxes_lidar)
        num_points_in_gt = indices.sum(0)
        
        num_ignored = len(annos['dimensions']) - num_obj
        num_points_in_gt = np.concatenate([num_points_in_gt, -np.ones([num_ignored])])
        annos['num_points_in_gt'] = num_points_in_gt.astype(np.int32)
        return info

    def calculate(self, infos):
        ret_infos = mmengine.track_parallel_progress(self.calculate_single,
                                                     infos, self.num_worker)
        for i, ret_info in enumerate(ret_infos):
            infos[i] = ret_info


def create_kitti_info_file(data_path,
                           pkl_prefix='kitti',
                           save_path=None,
                           relative_path=True):
    """Create info file of Object-Centric dataset."""
    data_path = Path(data_path)
    if save_path is None:
        save_path = data_path
    else:
        save_path = Path(save_path)

    # 1. Generate IDs from existing files in label_2
    label_dir = data_path / 'label_2'
    if not label_dir.exists():
        raise FileNotFoundError(f"Could not find label directory: {label_dir}")
        
    all_ids = [f.stem for f in label_dir.glob('*.txt')]
    all_ids.sort()
    
    print(f"Found {len(all_ids)} samples.")

    # 2. Simple Split (All to train for now)
    train_img_ids = all_ids
    val_img_ids = [] 

    print('Generate info. This may take several minutes.')

    # --- Train Info ---
    kitti_infos_train = get_object_centric_info(
        data_path,
        image_ids=train_img_ids,
        relative_path=relative_path
    )
    
    calculator = _NumPointsInGTCalculater(
        data_path, relative_path, num_features=4, remove_outside=False
    )
    calculator.calculate(kitti_infos_train)

    filename = save_path / f'{pkl_prefix}_infos_train.pkl'
    print(f'Info train file is saved to {filename}')
    mmengine.dump(kitti_infos_train, filename)

    # --- Val/Trainval Info ---
    if val_img_ids:
        kitti_infos_val = get_object_centric_info(
            data_path,
            image_ids=val_img_ids,
            relative_path=relative_path
        )
        calculator.calculate(kitti_infos_val)
        
        filename = save_path / f'{pkl_prefix}_infos_val.pkl'
        print(f'Info val file is saved to {filename}')
        mmengine.dump(kitti_infos_val, filename)
    else:
        filename = save_path / f'{pkl_prefix}_infos_trainval.pkl'
        print(f'Info trainval file is saved to {filename}')
        mmengine.dump(kitti_infos_train, filename)


def create_reduced_point_cloud(data_path,
                               pkl_prefix,
                               train_info_path=None,
                               val_info_path=None,
                               test_info_path=None,
                               save_path=None,
                               with_back=False):
    """Wrapper for creating reduced point clouds."""
    if train_info_path is None:
        train_info_path = Path(data_path) / f'{pkl_prefix}_infos_train.pkl'

    print('create reduced point cloud for training set')
    # Dummy implementation as points are already reduced
    pass