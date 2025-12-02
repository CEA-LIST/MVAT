import os
import pickle
import argparse
import yaml
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from collections import defaultdict

# Third party imports
import tri3d.tri3d
from tri3d.tri3d.datasets import NuScenes
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Local import
import utils

# Device Setup
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.enable_cudnn_sdp(False)
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")
np.random.seed(3)

# 3D Box Geometry Constants
BBOX_EDGES = torch.tensor([
    (-0.5, -0.5, -0.5), (+0.5, -0.5, -0.5), (+0.5, +0.5, -0.5), (-0.5, +0.5, -0.5),
    (-0.5, -0.5, +0.5), (+0.5, -0.5, +0.5), (+0.5, +0.5, +0.5), (-0.5, +0.5, +0.5),
])
BBOX_PATH_2D = [0, 1, 2, 3]

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="NuScenes Object-Centric Extraction")
    parser.add_argument('number', type=int, help='Start index for sequence processing')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    args = parser.parse_args()

    # --- Load Config ---
    cfg = load_config(args.config)
    
    # Aliases for config sections
    c_data = cfg['dataset']
    c_paths = cfg['paths']
    c_model = cfg['model']
    c_proc = cfg['processing']

    # --- Load Dataset ---
    print('-----------------------------> Loading Dataset')
    dataset = NuScenes(c_data['root_path'], c_data['version'])
    print('-----------------------------> Finished Loading Dataset')

    # --- Load Model ---
    sam2_model = build_sam2(c_model['sam2_config'], c_model['sam2_checkpoint'], device=device)
    predictor = SAM2ImagePredictor(sam2_model)

    # --- Load Transformations (if not trainval logic) ---
    transformations_pkl = None
    if "trainval" not in c_data['version']:
        with open(c_paths['global_transformations'], 'rb') as f:
            transformations_pkl = pickle.load(f)

    # --- Prepare Output ---
    output_dir = f"{c_paths['output_root']}_{args.number}"
    os.makedirs(os.path.join(output_dir, "calib"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "label_2"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "velodyne"), exist_ok=True)

    # --- Processing Loop ---
    batch_size = c_proc['batch_size']
    sequences_slice = dataset.sequences()[args.number : args.number + batch_size]

    for seq in tqdm(sequences_slice, desc="Sequences"):
        
        # Load Transform (Handling trainval pattern vs global)
        if "trainval" in c_data['version']:
            try:
                path = c_paths['sequence_transformations_pattern'].format(seq=seq)
                with open(path, 'rb') as f:
                    transformations = pickle.load(f)
            except FileNotFoundError:
                print(f"Transformations not found for {seq}, skipping.")
                continue
        else:
            transformations = transformations_pkl[seq]
            
        lidar_frames = dataset.keyframes(seq, c_data['lidar_sensor'])
        track_buffer = defaultdict(list) 

        for idx, lidar_frame in enumerate(lidar_frames):
            
            # Load Lidar
            full_pcl = dataset.points(seq=seq, frame=lidar_frame, coords=c_data['lidar_sensor'])[:, :3]
            
            # Loop Cameras
            for pov in range(len(c_data['cameras'])):
                camera = c_data['cameras'][pov]
                image = c_data['images'][pov]
                
                image_frame = dataset.keyframes(seq, image)[idx]
                sample_annotations = dataset.boxes(seq, image_frame, coords=image)
                sample_image = dataset.image(seq, image_frame, sensor=camera)
                input_image = np.array(sample_image.convert("RGB"))
                img_size = sample_image.size

                # SAM2 Inference
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16), torch.inference_mode():
                    predictor.set_image(input_image)
                
                # Matrices
                t_l2c = dataset.alignment(seq=seq, frame=(lidar_frame, image_frame), coords=(c_data['lidar_sensor'], image)).as_matrix()
                t2 = transformations[idx] 
                
                # Frustum Check
                uvz = full_pcl @ t_l2c[:3, :3].T + t_l2c[:3, 3]
                uv = utils.homo2img(uvz, img_size).to(torch.int64)
                
                in_frustum = (
                    (uvz[:,2] > 0) & 
                    (uv[:,0] > 0) & (uv[:,0] < img_size[0]) & 
                    (uv[:,1] > 0) & (uv[:,1] < img_size[1])
                )
                
                # Loop Annotations
                for ann in sample_annotations:
                    
                    # 3D -> 2D Box
                    obj_size = ann.size
                    bbox2img = ann.transform.as_matrix()
                    pts_3d_corners = (BBOX_EDGES.numpy() * obj_size) @ bbox2img[:3, :3].T + bbox2img[:3, 3]
                    pts_2d = utils.homo2img(pts_3d_corners, img_size)
                    
                    # Visibility Check
                    invisible = all(
                        (pts_2d[:,2] < 0) | 
                        (pts_2d[:,0] < 0) | (pts_2d[:,0] > img_size[0]) | 
                        (pts_2d[:,1] < 0) | (pts_2d[:,1] > img_size[1])
                    )
                    if invisible: continue

                    # Clip & Check size ratio
                    pts_2d_clip = pts_2d[BBOX_PATH_2D, 0:2]
                    min_x, min_y = np.min(pts_2d_clip.numpy(), axis=0)
                    max_x, max_y = np.max(pts_2d_clip.numpy(), axis=0)
                    box_2d = utils.clip_box([min_x, min_y, max_x, max_y], img_size)
                    
                    w_box, h_box = box_2d[2]-box_2d[0], box_2d[3]-box_2d[1]
                    ratio = c_proc['max_box_size_ratio']
                    if w_box > ratio*img_size[0] or h_box > ratio*img_size[1]: 
                        continue

                    # SAM Prediction
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16), torch.inference_mode():
                        masks, scores, _ = predictor.predict(box=np.array([box_2d]), multimask_output=False)
                    obj_mask = masks[0]

                    # Filter Points
                    frustum_uv = uv[in_frustum]
                    frustum_pcl = full_pcl[in_frustum]
                    
                    y_idx = np.clip(frustum_uv[:, 1].numpy(), 0, img_size[1]-1)
                    x_idx = np.clip(frustum_uv[:, 0].numpy(), 0, img_size[0]-1)
                    
                    mask_hits = obj_mask[y_idx, x_idx] == 1
                    box_hits = (x_idx >= box_2d[0]) & (x_idx <= box_2d[2]) & (y_idx >= box_2d[1]) & (y_idx <= box_2d[3])
                    
                    final_hits = mask_hits & box_hits
                    points_obj_local = frustum_pcl[final_hits]

                    if len(points_obj_local) < c_proc['min_points_per_frame']: 
                        continue

                    # World Transform & Store
                    points_world = points_obj_local @ t2[:3, :3].T + t2[:3, 3]
                    
                    lidar_box_gt = [b for b in dataset.boxes(seq, lidar_frame, coords=c_data['lidar_sensor']) if b.uid == ann.uid]
                    if not lidar_box_gt: continue
                        
                    obj_center_local = lidar_box_gt[0].center
                    obj_center_world = (t2 @ np.append(obj_center_local, 1))[:3]
                    
                    lidar2image = t_l2c
                    lidar2world = t2
                    world2lidar = np.linalg.inv(lidar2world)
                    world2image = lidar2image @ world2lidar

                    track_buffer[ann.uid].append({
                        'frame_idx': idx,              
                        'points': points_world,        
                        'centroid': obj_center_world,
                        'calibration': world2image,
                        'orig_label': ann.label,
                        'box_2d': box_2d 
                    })

        # --- Aggregation & Saving ---
        for uid, frames_data in track_buffer.items():
            if not frames_data: continue
            
            # Static Check
            centroids = [d['centroid'] for d in frames_data]
            if not utils.check_static_status(centroids, threshold=c_proc['static_threshold']):
                continue 

            # Aggregate
            pts_list = []
            for d in frames_data:
                N = d['points'].shape[0]
                idx_col = np.full((N, 1), d['frame_idx'], dtype=np.float32)
                pts_with_t = np.hstack([d['points'], idx_col])
                pts_list.append(pts_with_t)
                
            all_points_4d = np.concatenate(pts_list, axis=0)
            
            # Clean & Box (Passing config params)
            clean_mask, pca_box = utils.get_clean_cluster_and_box(
                all_points_4d[:, :3], 
                eps=c_proc['dbscan_eps'],
                min_samples=c_proc['dbscan_min_samples'],
                iou_thresh=c_proc['iou_threshold']
            )
            
            if pca_box is None: continue
            
            clean_points_4d = all_points_4d[clean_mask]

            # Save Velodyne
            velodyne_path = os.path.join(output_dir, "velodyne", f"{uid}.bin")
            clean_points_4d.astype(np.float32).tofile(velodyne_path)

            # Save Label
            first = frames_data[0]
            class_type = first['orig_label']
            box2d = first['box_2d']
            l, w, h = pca_box['size']
            cx, cy, cz = pca_box['center']
            rot = pca_box['heading']
            
            label_str = f"{class_type} 0.00 0 0.00 {box2d[0]:.2f} {box2d[1]:.2f} {box2d[2]:.2f} {box2d[3]:.2f} " \
                        f"{h:.3f} {w:.3f} {l:.3f} {cx:.3f} {cy:.3f} {cz:.3f} {rot:.3f}"
            
            with open(os.path.join(output_dir, "label_2", f"{uid}.txt"), "w") as f:
                f.write(label_str)

            # Save Calib
            calib_path = os.path.join(output_dir, "calib", f"{uid}.txt")
            with open(calib_path, "w") as f:
                for d in frames_data:
                    f_idx = d['frame_idx']
                    mat_flat = d['calibration'].flatten()
                    mat_str = " ".join([f"{x:.12e}" for x in mat_flat])
                    f.write(f"{f_idx} {mat_str}\n")
            
            print(f"Saved {uid}: {len(clean_points_4d)} points")

    print(f"Finished. Data saved in {output_dir}")

if __name__ == "__main__":
    main()