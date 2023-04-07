import torch
from .detector2d_template import Detector2DTemplate
from ..image_backbone.image_center_net.decode import bbox3d_decode
from ...utils.bev_bbox_visualizer import BEVBboxVisualizer
import numpy as np

class CenterNet(Detector2DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(
            model_cfg=model_cfg,
            num_class=num_class,
            dataset=dataset
        )
        self.module_list = self.build_networks()
        # Eval settings
        self.eval_cfg = getattr(self.model_cfg, 'EVAL_CONFIG', {})
        if 'EVAL_LOSS' in self.eval_cfg.keys():
            self.eval_loss = self.eval_cfg['EVAL_LOSS']
        else:
            self.eval_loss = False

        if 'EVAL_MODE' in self.eval_cfg.keys():
            self.eval_mode = self.eval_cfg['EVAL_MODE']
        else:
            self.eval_mode = '3d'

        # generate 3d bbox in postprocessing
        self.gen_3d = getattr(self.eval_cfg, 'GENERATE_3D_RES', False)
        self.filter_box = getattr(self.eval_cfg, 'FILTER_UNREASONABLE_BOX', False)

        if self.gen_3d:
            self.down_ratio = getattr(
                getattr(self.model_cfg, 'IMAGE_BACKBONE'), 'DOWN_RATIO', 4)
            self.peak_map_thred = getattr(
                getattr(self.model_cfg, 'IMAGE_BACKBONE'),
                'PEAK_MAP_THRESH', 0.25)
            self.vis_3d_step = getattr(self.eval_cfg, 'TB_VIS_3D_STEP', 1)

            self.range = dataset.point_cloud_range
            self.grid_size = dataset.grid_size
            self.voxel_size = dataset.voxel_size

            all_class = ['unknown'] # zero id for gt to get the same color
            all_class.extend(self.class_names)
            self.vis = BEVBboxVisualizer(self.num_class + 1, all_class)

    def forward(self, batch_dict):
        tb_dict = {}
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)
            # Tensorboard visualization
            tb_visualize = getattr(cur_module, 'tb_visualize', False)
            if tb_visualize:
                tb_dict = cur_module.visualize_tb(batch_dict, tb_dict=tb_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss(
                batch_dict, tb_dict)
            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            if tb_visualize:
                if self.gen_3d:
                    temp_tb_dict = self.vis_bev_peseudo_box(pred_dicts, batch_dict)
                    pred_dicts[0]['temp_tb_dict'] = temp_tb_dict
                    pred_dicts[0]['vis_temp_tb_step'] = self.vis_3d_step
                pred_dicts[0]['tb_dict'] = tb_dict
            if self.eval_loss:
                _, loss_dict, _ = self.get_training_loss(batch_dict, {})
                return pred_dicts, recall_dicts, loss_dict
            else:
                return pred_dicts, recall_dicts

    def get_training_loss(self, data_dict, tb_dict=None):
        disp_dict = {}
        if tb_dict is None:
            tb_dict = {}
        loss, tb_dict = self.image_backbone.get_loss(
            data_dict=data_dict, tb_dict=tb_dict)

        return loss, tb_dict, disp_dict

    def vis_bev_peseudo_box(self, pred_dicts, batch_dict):
        temp_tb_dict = {}

        img_range = (self.range[[0, 1, 3, 4]]).astype(np.int) # (4,1)

        img = np.ones([self.grid_size[0], self.grid_size[1], 3])

        full_img = img.copy()
        gt_img = img.copy()
        pred_img = img.copy()

        pred_boxes = pred_dicts[0]['pred_boxes'].cpu().numpy()
        pred_ids = pred_dicts[0]['pred_labels'].cpu().numpy()
        pred_bev_box_list = []
        for pred_box in pred_boxes:
            rot_angle = -pred_box[6] / np.pi * 180
            # bev tuple type
            temp_box = (((pred_box[0] - img_range[0]) / self.voxel_size[0], \
                        (pred_box[1] - img_range[1]) / self.voxel_size[1]),
                        (pred_box[3] / self.voxel_size[0], pred_box[4] / self.voxel_size[1]), rot_angle)
            pred_bev_box_list.append(temp_box)

        full_img = self.vis.add_boxes_2_img(full_img, pred_bev_box_list, pred_ids)
        pred_img = self.vis.add_boxes_2_img(pred_img, pred_bev_box_list, pred_ids)

        gt_boxes = batch_dict['gt_boxes'][0].cpu().numpy()
        gt_id_list = batch_dict['gt_boxes'][0, :, -1].cpu().numpy().astype(np.int)
        gt_bev_box_list = []
        for gt_box  in gt_boxes:
            if gt_box[0] <= img_range[0] or gt_box[1] <= img_range[1] or \
                gt_box[0] >= img_range[2] or gt_box[1] >= img_range[2]:
                continue
            # bev tuple type
            rot_angle = -gt_box[6] / np.pi * 180
            temp_box = (((gt_box[0] - img_range[0]) / self.voxel_size[0], \
                        (gt_box[1] - img_range[1]) / self.voxel_size[1]),
                        (gt_box[3] / self.voxel_size[0], gt_box[4] / self.voxel_size[1]), rot_angle)
            gt_bev_box_list.append(temp_box)

        gt_ids = [0] * len(gt_bev_box_list)
        full_img = self.vis.add_boxes_2_img(full_img, gt_bev_box_list, gt_ids)
        gt_img = self.vis.add_boxes_2_img(gt_img, gt_bev_box_list, gt_id_list)

        temp_tb_dict['visual:pseudo_bev_boxes_with_gt'] = full_img.transpose([2, 1, 0])  # CHW
        temp_tb_dict['visual:pseudo_bev_boxes'] = pred_img.transpose([2, 1, 0])
        temp_tb_dict['visual:pseudo_gt_boxes'] = gt_img.transpose([2, 1, 0])

        return temp_tb_dict

    def vis_bev_peseudo_box_v0(self, pred_dicts, batch_dict):
        temp_tb_dict = {}

        img_range = (self.range[[0, 1, 3, 4]]).astype(np.int) # (4,1)
        vis_scale = 10 # which is important
        img = np.ones([ vis_scale * (img_range[2] - img_range[0] + 1), \
                        vis_scale * (img_range[3] - img_range[1] + 1), 3])

        pred_boxes = pred_dicts[0]['pred_boxes'].cpu().numpy()
        pred_ids = pred_dicts[0]['pred_labels'].cpu().numpy()
        pred_bev_box_list = []
        # for pred_box in pred_boxes:
            # bev tuple type
            # temp_box = (((pred_box[0] - img_range[0]),(pred_box[1] - img_range[1])),
            #             (pred_box[3], pred_box[4]), pred_box[6])
        x_c, y_c = vis_scale * (pred_boxes[:, 0] - img_range[0]), \
                    vis_scale * (pred_boxes[:, 1] - img_range[1])
        pred_bev_box_list = np.array([x_c, y_c, x_c + vis_scale * pred_boxes[:, 3], \
                                        y_c + vis_scale * pred_boxes[:, 4]]).astype(np.int).transpose()

        img = self.vis.add_boxes_2_img(img, pred_bev_box_list, pred_ids)

        gt_boxes = batch_dict['gt_boxes'][0].cpu().numpy()
        gt_bev_box_list = []
        for gt_box  in gt_boxes:
            if gt_box[0] <= img_range[0] or gt_box[1] <= img_range[1] or \
                gt_box[0] >= img_range[2] or gt_box[1] >= img_range[2]:
                continue
            x_c, y_c = vis_scale * (gt_box[0] - img_range[0]), \
                       vis_scale * (gt_box[1] - img_range[1])
            temp_box = [int(x_c), int(y_c), int(x_c + vis_scale * gt_box[3]), int(y_c + vis_scale * gt_box[4])]
            gt_bev_box_list.append(temp_box)

        gt_ids = [0] * len(gt_bev_box_list)
        img = self.vis.add_boxes_2_img(img, gt_bev_box_list, gt_ids)

        temp_tb_dict['visual:pseudo_bev_boxes_with_gt'] = img.transpose([2, 1, 0])  # CHW

        return temp_tb_dict

    def post_processing(self, batch_dict):
        """
        Post-processing without NMS

        Args:
            batch_dict:
        Returns:
            pred_dicts
            recall_dict
        """
        post_process_cfg = self.model_cfg.POST_PROCESSING
        score_thresh = post_process_cfg.SCORE_THRESH
        batch_size = batch_dict['batch_size']
        recall_dict = {}
        pred_dicts = []
        if 'use_tta' in batch_dict:
            batch_size = int(batch_size / 2)
            ori_shape = batch_dict['image_shape_ori']

        if self.gen_3d:
            assert 'img_3d_dim_pred' in batch_dict.keys() and \
                    'pred_3dcts' in batch_dict.keys() and \
                    'img_3dct_depth_pred' in batch_dict.keys()

            for index in range(batch_size):
                record_dict = {
                        'pred_boxes': [],
                        'pred_scores': [],
                        'pred_labels': []
                    }
                for cam in batch_dict['pred_boxes2d'].keys():
                    # process for every batch since masks will be different
                    img_hm_pred = batch_dict['img_hm_pred'][cam]
                    if 'use_tta' in batch_dict:
                        img_hm_pred = batch_dict['{}_hm'.format(cam)]
                    temp_pred_boxes3d = bbox3d_decode(
                        img_hm_pred[[index]], batch_dict['pred_3dcts'][cam][[index]],
                        batch_dict['img_3d_dim_pred'][cam][[index]], batch_dict['img_3dct_depth_pred'][cam][[index]],
                        batch_dict['img_to_cam'][cam][[index]], batch_dict['cam_to_lidar'][cam][[index]],
                        gt_boxes=batch_dict['gt_boxes'][[index]], down_ratio=self.down_ratio,
                        max_num_det=batch_dict['filter_max_num'][cam]
                    )

                    temp_pred_boxes3d = temp_pred_boxes3d.squeeze().reshape(-1,9)
                    vaild_box_ids = []
                    for box_id, line in enumerate(temp_pred_boxes3d):
                        pred_score = line[7]
                        if pred_score > score_thresh:
                            vaild_box_ids.append(box_id)
                    vaild_box_ids = torch.tensor(
                        vaild_box_ids, device=temp_pred_boxes3d.device, dtype=torch.long)

                    final_boxes3d = torch.index_select(temp_pred_boxes3d, 0, vaild_box_ids)
                    record_dict['pred_boxes'].append(final_boxes3d[:, :7])
                    record_dict['pred_scores'].append(final_boxes3d[:, -2])
                    record_dict['pred_labels'].append(final_boxes3d[:, -1] + 1)

                recall_dict = self.generate_recall_record(
                    box_preds=final_boxes3d,
                    recall_dict=recall_dict,
                    batch_index=index,
                    data_dict=batch_dict,
                    thresh_list=post_process_cfg.RECALL_THRESH_LIST)

                record_dict['pred_boxes'] = torch.cat(record_dict['pred_boxes'], dim=0)
                record_dict['pred_scores'] = torch.cat(record_dict['pred_scores'], dim=0)
                record_dict['pred_labels'] = torch.cat(record_dict['pred_labels'], dim=0).int()

                pred_dicts.append(record_dict)
        else:
            for index in range(batch_size):
                record_dict = {
                        'pred_boxes2d': {},
                        'pred_scores': {},
                        'pred_labels': {}
                    }
                for cam in batch_dict['pred_boxes2d'].keys():
                    pred_boxes2d = batch_dict['pred_boxes2d'][cam][index]

                    vaild_box_ids = []
                    for box_id, line in enumerate(pred_boxes2d):
                        pred_score = line[4]
                        if pred_score > score_thresh:
                            vaild_box_ids.append(box_id)
                    vaild_box_ids = torch.tensor(
                        vaild_box_ids, device=pred_boxes2d.device, dtype=torch.long)
                    final_boxes2d = torch.index_select(
                        pred_boxes2d[:, 0:4], 0, vaild_box_ids)
                    final_labels = torch.index_select(
                        pred_boxes2d[:, -1], 0, vaild_box_ids).long() + 1
                    final_scores = torch.index_select(
                        pred_boxes2d[:, -2], 0, vaild_box_ids)

                    # GT Heatmap is already in prob domain
                    # if not self.model_cfg.IMAGE_BACKBONE.DECODE_GT_HEATMAP:
                    #     final_scores = final_scores.sigmoid()

                    '''
                    recall_dict = self.generate_recall_record(
                        box_preds=final_boxes if 'rois' not in batch_dict else src_box_preds,
                        recall_dict=recall_dict,
                        batch_index=index,
                        data_dict=batch_dict,
                        thresh_list=post_process_cfg.RECALL_THRESH_LIST)
                    '''

                    record_dict['pred_boxes2d'][cam] = final_boxes2d
                    record_dict['pred_scores'][cam] = final_scores
                    record_dict['pred_labels'][cam] = final_labels

                pred_dicts.append(record_dict)

        # since CenterPoint and Fusion use exp to restrict box's dim
        # we just filter out unreasonable(dim<0) boxes in image branch
        if self.filter_box:
            new_pred_dicts = []
            for pred_dict in pred_dicts:
                if self.gen_3d:
                    mask = (pred_dict['pred_boxes'][:, 3] > 0) & (pred_dict['pred_boxes'][:, 4] > 0) & \
                            (pred_dict['pred_boxes'][:, 5] > 0)
                    pred_dict['pred_boxes'] = pred_dict['pred_boxes'][mask]
                else:
                    mask = ((pred_dict['pred_boxes2d'][:, 2] - pred_dict['pred_boxes2d'][:, 0]) > 0) & \
                            ((pred_dict['pred_boxes2d'][:, 3] - pred_dict['pred_boxes2d'][:, 1]) > 0)
                    pred_dict['pred_boxes2d'] = pred_dict['pred_boxes2d'][mask]

                    pred_dict['pred_scores'] = pred_dict['pred_scores'][mask]
                    pred_dict['pred_labels'] = pred_dict['pred_labels'][mask]
                new_pred_dicts.append(pred_dict)

            pred_dicts = new_pred_dicts

        return pred_dicts, recall_dict
