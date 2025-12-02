import numpy as np
import torch
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon

def homo2img(x, img_size):
    """
    Projects 3D homogeneous coordinates to 2D image coordinates.
    Handles points behind the camera.
    """
    x = torch.tensor(x) if not isinstance(x, torch.Tensor) else x
    out = x.reshape(-1, 3).clone()
    out[:, 0] = out[:, 0] / out[:, 2]
    out[:, 1] = out[:, 1] / out[:, 2]
    
    # Mirror points behind camera
    behind = out[:, 2] < 0
    out[behind, 0] = img_size[0] - 1 - out[behind, 0]
    out[behind, 1] = img_size[1] - 1 - out[behind, 1]
    
    return out[0] if x.dim() == 1 else out

def clip_box(box_2d, img_size):
    """
    Clips a 2D bounding box to the image boundaries.
    """
    x_min, y_min, x_max, y_max = box_2d
    w, h = img_size
    return [max(0, min(x_min, w-1)), max(0, min(y_min, h-1)), 
            max(0, min(x_max, w-1)), max(0, min(y_max, h-1))]

def check_static_status(centroids, threshold=0.5):
    """
    Determines if an object is static based on the movement of its centroid.
    """
    if len(centroids) < 2: 
        return True # Assume static if only seen once
    
    centroids = np.array(centroids)
    # Calculate pairwise distances between all recorded centroids
    diff = centroids[:, np.newaxis, :] - centroids[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=-1))
    
    # If the maximum distance the object moved is within threshold, it's static
    return np.max(dist) < threshold

def get_clean_cluster_and_box(points_xyz, eps=0.5, min_samples=10, iou_thresh=0.3):
    """
    1. Clusters points using DBSCAN to remove noise.
    2. Estimates a 3D bounding box using PCA on the largest cluster.
    3. Verifies the box using IoU against the cluster's Convex Hull.
    
    Returns: 
       mask (N,): Boolean mask of points belonging to the clean cluster.
       box_data: Dictionary containing 'center', 'size', and 'heading'.
    """
    if len(points_xyz) < min_samples: 
        return None, None

    # --- 1. DBSCAN Cleaning ---
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points_xyz)
    labels = clustering.labels_
    unique_labels = set(labels) - {-1} # Exclude noise (-1)
    
    if not unique_labels: 
        return None, None

    # Select largest cluster
    best_label = max(unique_labels, key=lambda k: np.sum(labels == k))
    mask = (labels == best_label)
    clean_points = points_xyz[mask]

    if len(clean_points) < 4: 
        return None, None

    # --- 2. PCA Estimation (Bird's Eye View) ---
    points_bev = clean_points[:, :2] # X, Y
    pca = PCA(n_components=2).fit(points_bev)
    v1 = pca.components_[0] # Principal axis
    theta = np.arctan2(v1[1], v1[0])

    # Rotation matrix to align points with axes
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, s], [-s, c]])
    points_local = points_bev @ R.T
    
    min_uv = np.min(points_local, axis=0)
    max_uv = np.max(points_local, axis=0)
    
    # Dimensions (Length, Width)
    l, w = max_uv - min_uv
    c_uv = (max_uv + min_uv) / 2
    
    # Rotate center back to global frame
    c_xy = c_uv @ R
    
    # Vertical Parameters (Z)
    z_vals = clean_points[:, 2]
    h = np.max(z_vals) - np.min(z_vals)
    c_z = (np.max(z_vals) + np.min(z_vals)) / 2

    box_data = {
        'center': np.array([c_xy[0], c_xy[1], c_z]),
        'size': np.array([l, w, h]), 
        'heading': theta
    }

    # --- 3. Geometric Verification ---
    try:
        # Reconstruct Box Polygon
        corners_local = [
            [min_uv[0], min_uv[1]], [max_uv[0], min_uv[1]],
            [max_uv[0], max_uv[1]], [min_uv[0], max_uv[1]]
        ]
        corners_global = np.array(corners_local) @ R
        box_poly = Polygon(corners_global)
        
        # Convex Hull of points
        hull = ConvexHull(points_bev)
        hull_poly = Polygon(points_bev[hull.vertices])
        
        if not box_poly.is_valid or not hull_poly.is_valid: 
            return None, None

        intersection = box_poly.intersection(hull_poly).area
        union = box_poly.union(hull_poly).area
        
        iou = intersection / union if union > 0 else 0
        
        if iou > iou_thresh:
            return mask, box_data
            
    except Exception:
        pass
        
    return None, None