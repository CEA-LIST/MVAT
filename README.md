# MVAT: Multi-View Aware Teacher for Weakly Supervised 3D Object Detection

This repository contains the code for **MVAT**, a weakly supervised 3D object detection framework that bootstraps 3D object detection solely from 2D bounding box annotations. 

*will be Presented at IEEE/CVF Winter Conference on Applications of Computer Vision WACV 2026*

**[Paper](https://arxiv.org/abs/2509.07507)**


## **📝 Abstract**

Annotating 3D data remains a costly bottleneck for 3D object detection, motivating the development of weakly supervised annotation methods that rely on more accessible 2D box annotations. However, relying solely on 2D boxes introduces projection ambiguities since a single 2D box can correspond to multiple valid 3D poses. Furthermore, partial object visibility under a single viewpoint setting makes accurate 3D box estimation difficult. We propose MVAT, a novel framework that leverages temporal multi-view present in sequential data to address these challenges. Our approach aggregates object-centric point clouds across time to build 3D object representations as dense and complete as possible. A Teacher-Student distillation paradigm is employed: The Teacher network learns from single viewpoints but targets are derived from temporally aggregated static objects. Then the Teacher generates high quality pseudo-labels that the Student learns to predict from a single viewpoint for both static and moving objects. The whole framework incorporates a multi-view 2D projection loss to enforce consistency between predicted 3D boxes and all available 2D annotations. Experiments on the nuScenes and Waymo Open datasets demonstrate that MVAT achieves state-of-the-art performance for weakly supervised 3D object detection, significantly narrowing the gap with fully supervised methods without requiring any 3D box annotations. 

## 📂 Project Structure

The codebase is organized into two distinct modules:

* **`dataset/`**: A standalone pipeline for data generation. It handles ICP alignment, SAM 2 segmentation, temporal aggregation, and coarse 3D box estimation to create the "Object-Centric" dataset.
* **`mmdet3d/`**: Contains the plugin files, custom heads, and configurations required to train the model using the [MMDetection3D](https://github.com/open-mmlab/mmdetection3d) framework.

```text
MVAT/
├── dataset/                    # --- Phase 1: Data Generation ---
│   ├── config.yaml             # Hyperparameters for preprocessing
│   ├── icp_nuscenes.py         # Refines NuScenes ego-motion via ICP
│   ├── preprocessor.py         # Main script: SAM 2 + Aggregation + Coarse Estimation
│   ├── utils.py                # Geometry, DBSCAN, and PCA utilities
│   └── checkpoints/            # Directory for SAM 2 weights
│
├── mmdet3d/                    # --- Phase 2: MMDetection3D Plugins ---
│   ├── kitti_converter.py      # Generates .pkl info files from MVAT data
│   ├── kitti_dataset.py        # Custom Dataset class for Object-Centric data
│   ├── projected_loss.py       # Implements the Multi-View 2D Projection Loss
│   ├── mvat_roi_head.py        # Custom RoI Head integrating the 2D loss
│   ├── custom_transforms.py    # Pipeline transforms for multi-view metadata
│   ├── kitti-3d-3class.py      # Dataset configuration
│   └── pv_rcnn_8xb2-80e...py   # Model configuration (PV-RCNN + MVAT)
│
└── README.md
````

-----

## 🚀 Phase 1: Data Generation

This phase processes raw NuScenes data to generate a clean, Object-Centric dataset in KITTI format.

### Prerequisites

1.  **NuScenes Dataset**: Download the `v1.0-trainval` or `v1.0-mini` dataset.
2.  **SAM 2**: Install [Segment Anything 2](https://github.com/facebookresearch/segment-anything-2) and download the `sam2.1_hiera_large.pt` checkpoint.
3.  **Dependencies**: `open3d`, `torch`, `sklearn`, `shapely`.

### Step 1: Calibration Refinement (ICP)

Standard NuScenes projection matrices may lack the precision required for dense temporal aggregation. Run the ICP script to generate refined frame-to-frame transformations.

```bash
cd dataset
# Edit 'repo_path' in the script to your desired output location
python icp_nuscenes.py
```

### Step 2: Object-Centric Extraction

Configure `dataset/config.yaml` to point to your NuScenes root and the `.pickle` transformations generated in Step 1. Then run the preprocessor:

```bash
# Process sequence starting at index 0
python preprocessor.py 0 --config config.yaml
```

**Output Structure (`kitti_format_output_{N}/`):**

  * **`velodyne/`**: Aggregated, clean point clouds for static objects.
  * **`label_2/`**: Single 3D Coarse Box label per object.
  * **`calib/`**: Frame-specific calibration matrices.

-----

## 🛠️ Phase 2: Training with MMDetection3D

This phase trains the 3D detector. We provide custom plugins to integrate the **Multi-View 2D Projection Loss** into the standard PV-RCNN architecture.

### 1\. Installation & Integration

Ensure you have **MMDetection3D v1.1+** installed. You can incorporate the provided files by copying them into your `mmdet3d` installation or ensuring the `mmdet3d/` directory of this repo is in your `PYTHONPATH`.

### 2\. Generate Info Files

Convert the custom KITTI-format data generated in Phase 1 into MMDetection `.pkl` info files.

```bash
# Update the paths inside kitti_converter.py to point to your 'kitti_format_output'
python mmdet3d/kitti_converter.py
```

### 3\. Configuration

We provide a complete configuration setup for PV-RCNN:

  * **Dataset Config (`kitti-3d-3class.py`)**: Uses `LoadMultiViewGT` and `PackMultiViewInputs` (from `custom_transforms.py`) to load the `multiview_meta` files alongside standard data.
  * **Model Config (`pv_rcnn...10class.py`)**: Replaces the standard RoI Head with `MVATPVRCNNRoiHead` (from `mvat_roi_head.py`), enabling the joint optimization of $\mathcal{L}_{3D}$ and $\mathcal{L}_{2D}$.

### 4\. Training

Train the model using standard MMDetection3D tools.

-----


## 📄 Citation

If you find this work useful, please cite our paper:

```bibtex
@article{lahlali2025mvat,
  title={MVAT: Multi-View Aware Teacher for Weakly Supervised 3D Object Detection},
  author={Lahlali, Saad and Fournier-Montgieux, Alexandre and Granger, Nicolas and Le Borgne, Hervé and Pham, Quoc Cuong},
  booktitle={2026 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year={2026},
  organization={IEEE}
}
```