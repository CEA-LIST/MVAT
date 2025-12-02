import torch
from mmdet3d.registry import MODELS
from mmdet3d.models.roi_heads import PVRCNNRoiHead
from mmdet3d.structures import LiDARInstance3DBoxes
from .projected_loss import MultiView2DProjectionLoss

@MODELS.register_module()
class MVATPVRCNNRoiHead(PVRCNNRoiHead):
    def __init__(self, loss_2d_cfg=None, **kwargs):
        super().__init__(**kwargs)
        if loss_2d_cfg is not None:
            self.loss_2d = MultiView2DProjectionLoss(**loss_2d_cfg)
        else:
            self.loss_2d = None

    def loss(self, x, rpn_results_list, batch_data_samples):
        """
        Overridden loss method to include L_2D (Multi-view Projection Loss).
        """
        # 1. Run standard PVRCNN 3D loss (L_3D)
        losses = super().loss(x, rpn_results_list, batch_data_samples)
        
        # If L_2D is not configured, return early
        if self.loss_2d is None:
            return losses

        # 2. Retrieve the refined 3D bounding boxes calculated during super().loss()
        # Note: PVRCNNRoiHead doesn't store the predictions in self during loss(),
        # so we effectively need to re-run the bbox_forward to get the boxes 
        # or hook into the sampling results. 
        # For efficiency in this custom implementation, we reconstruct the necessary flow 
        # to get the predicted boxes associated with positive samples.

        # Recalculate roi_losses logic to access predictions
        # (This is a slight compute overhead but ensures we get the exact boxes used for L_3D)
        
        # ... Extract features ...
        batch_gt_instances_3d = []
        batch_gt_instances_ignore = []
        for data_sample in batch_data_samples:
            batch_gt_instances_3d.append(data_sample.gt_instances_3d)
            batch_gt_instances_ignore.append(data_sample.ignored_instances)

        # Assign and Sample
        sampling_results = []
        bbox_results_list = []
        
        # Forward through the network again to get the boxes (or cache them if modifying base code)
        # Here we perform the forward pass steps manually to get 'rcnn_boxes_3d'
        
        # 2a. RoI extraction
        roi_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rpn_results_list)
        
        # 2b. BBox Head Forward
        cls_score, bbox_pred = self.bbox_head(roi_feats)
        
        # 2c. Recover 3D boxes
        # We process batch by batch
        loss_2d_total = 0
        num_positive_samples = 0
        
        for i in range(len(batch_data_samples)):
            # Get Ground Truths for this sample
            gt_instances = batch_data_samples[i].gt_instances_3d
            gt_bboxes_3d = gt_instances.bboxes_3d
            gt_labels_3d = gt_instances.labels_3d
            
            # Get RPN proposals
            rpn_results = rpn_results_list[i]
            
            # Assign (Match proposals to GT)
            assign_result = self.bbox_assigner.assign(
                rpn_results, gt_bboxes_3d, gt_labels_3d)
            
            # Sample (Select positives/negatives)
            sampling_result = self.bbox_sampler.sample(
                assign_result, rpn_results, gt_bboxes_3d)
            
            # Get Positive Proposals indices
            pos_inds = sampling_result.pos_inds
            if len(pos_inds) == 0:
                continue

            # Extract predicted deltas for positives
            # bbox_pred shape: (Batch*Proposals, Code_size)
            # We need to slice correct batch index
            # NOTE: ROI extractor flattens batch. We need to handle offsets if batch > 1
            # For simplicity, assuming standard MMDetection logic where feats are stacked:
            
            # Correct logic for stacked features in MMDet3D:
            # The features are stacked (B*N, C). We need to know which belong to batch i.
            num_proposals = [res.bboxes.shape[0] for res in rpn_results_list]
            batch_offset = sum(num_proposals[:i])
            local_pos_inds = pos_inds + batch_offset
            
            pos_bbox_pred = bbox_pred[local_pos_inds]
            pos_rois = rpn_results.bboxes[pos_inds] # These are the anchors/proposals
            
            # Decode to get final Refined 3D Boxes
            # The bbox_head.bbox_coder expects (rois, deltas)
            refined_bboxes_3d = self.bbox_head.bbox_coder.decode(
                pos_rois, pos_bbox_pred)
            
            # 3. Prepare Inputs for L_2D
            # We need:
            #   - predicted 3D boxes (refined_bboxes_3d)
            #   - calibration matrices (lidar2img)
            #   - ground truth 2D boxes
            
            # Retrieve Metadata from batch_data_samples
            # Note: You added 'calib_matrices' and 'gt_bboxes_2d' in the pipeline.
            # In MMDets, these custom keys usually end up in data_sample.metainfo 
            # or data_sample.gt_instances if explicitly mapped.
            # Based on your pipeline, they are likely in metainfo or direct attributes.
            
            # Assuming they are accessible via data_sample.eval_ann_info or custom fields
            # You might need to adjust this access based on where Pack3DDetInputs put them.
            # Let's assume they are stored in data_sample.gt_instances_3d for simplicity 
            # (requires modifying LoadAnnotations3D to put them there)
            # OR typically: data_sample.metainfo['calib_matrices'] 
            
            calib_matrices_dict = batch_data_samples[i].get('calib_matrices', None)
            gt_bboxes_2d_dict = batch_data_samples[i].get('gt_bboxes_2d', None)
            
            # If standard pipeline puts them in gt_instances, access there:
            # (This part depends heavily on how you hacked Pack3DDetInputs)
            
            if calib_matrices_dict is None:
                # Fallback check if stored in gt_instances
                if hasattr(gt_instances, 'calib_matrices'):
                    calib_matrices_dict = gt_instances.calib_matrices
                else:
                    continue # Cannot compute loss without calibs
            
            # The network predicts N positive boxes. 
            # Each positive box matches a specific GT object (assigned_gt_inds).
            matched_gt_inds = sampling_result.pos_assigned_gt_inds
            
            # Prepare flattened lists for the loss function
            loss_pred_3d = []
            loss_gt_2d = []
            loss_lidar2img = []
            loss_img_shapes = []
            loss_obj_ids = []
            
            for k, gt_idx in enumerate(matched_gt_inds):
                # The prediction
                pred_box = refined_bboxes_3d[k] # (7,)
                
                # The matched GT Object info
                # Note: your data loader seems to load 1 object per sample (Object-Centric)?
                # If so, gt_idx is always 0 for the single object.
                # If multiple objects, we need to find which 2D views belong to this GT.
                
                # Assuming dictionary structure: {frame_idx: matrix}
                # And gt_bboxes_2d is a list or dict corresponding to these frames.
                
                # Iterate over all available views for this object
                # (Provided by your kitti_converter/loader logic)
                current_calibs = calib_matrices_dict # Expecting dict {frame_id: 4x4}
                current_2d_boxes = gt_bboxes_2d_dict # Expecting dict {frame_id: [x1,y1,x2,y2]}
                
                # If the dataset structure is flat (list of all views), iterate
                if isinstance(current_calibs, list):
                     # If it's a list, assume alignment with 2d_boxes list
                     views_iter = zip(current_calibs, current_2d_boxes)
                elif isinstance(current_calibs, dict):
                     views_iter = []
                     for fid, mat in current_calibs.items():
                         if fid in current_2d_boxes:
                             views_iter.append((mat, current_2d_boxes[fid]))
                else:
                    continue

                for mat, box2d in views_iter:
                    loss_pred_3d.append(pred_box)
                    loss_gt_2d.append(torch.tensor(box2d, device=pred_box.device))
                    loss_lidar2img.append(torch.tensor(mat, device=pred_box.device).float())
                    
                    # Placeholder image shape (H, W) - strictly for clipping
                    # If not passed, pick reasonable default or pass via data loader
                    loss_img_shapes.append(torch.tensor([900, 1600], device=pred_box.device)) 
                    
                    loss_obj_ids.append(gt_idx)

            if len(loss_pred_3d) > 0:
                # Stack tensors
                stack_pred = torch.stack(loss_pred_3d)
                stack_gt = torch.stack(loss_gt_2d)
                stack_calib = torch.stack(loss_lidar2img)
                stack_shapes = torch.stack(loss_img_shapes)
                stack_ids = torch.tensor(loss_obj_ids, device=stack_pred.device)
                
                # Compute L_2D for this batch item
                l2d = self.loss_2d(
                    stack_pred, 
                    stack_gt, 
                    stack_calib, 
                    stack_shapes, 
                    object_ids=stack_ids
                )
                loss_2d_total += l2d
                num_positive_samples += 1

        # Normalize loss
        if num_positive_samples > 0:
            losses['loss_2d'] = loss_2d_total / num_positive_samples
        else:
            losses['loss_2d'] = x[0].new_tensor(0.0)

        return losses