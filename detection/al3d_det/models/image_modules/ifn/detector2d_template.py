import os

import torch
import torch.nn as nn

from . import image_backbone
from al3d_det.ops.iou3d_nms import iou3d_nms_utils
from al3d_det.models import vfe


class Detector2DTemplate(nn.Module):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.dataset = dataset
        self.class_names = dataset.class_names
        self.tta = dataset.tta
        self.register_buffer('global_step', torch.LongTensor(1).zero_())

        self.module_topology = [
            'vfe',
            'image_backbone',
            # 'depth_backbone',
            # 'roi_heads'
        ]

    @property
    def mode(self):
        return 'TRAIN' if self.training else 'TEST'

    def update_global_step(self):
        self.global_step += 1

    def build_networks(self):
        model_info_dict = {
            'module_list': [],
            'image_size': self.dataset.image_size,
            'depth_downsample_factor': self.dataset.depth_downsample_factor \
                if self.dataset.use_depth else None,
            'num_rawpoint_features': self.dataset.point_feature_encoder.num_point_features,
            'num_point_features': self.dataset.point_feature_encoder.num_point_features,
            'grid_size': self.dataset.grid_size,
            'point_cloud_range': self.dataset.point_cloud_range,
            'voxel_size': self.dataset.voxel_size,
        }
        for module_name in self.module_topology:
            module, model_info_dict = getattr(self, 'build_%s' % module_name)(
                model_info_dict=model_info_dict
            )
            self.add_module(module_name, module)
        return model_info_dict['module_list']

    def build_vfe(self, model_info_dict):
        if self.model_cfg.get('VFE', None) is None:
            return None, model_info_dict

        vfe_module = vfe.__all__[self.model_cfg.VFE.NAME](
            model_cfg=self.model_cfg.VFE,
            num_point_features=model_info_dict['num_rawpoint_features'],
            point_cloud_range=model_info_dict['point_cloud_range'],
            voxel_size=model_info_dict['voxel_size'],
            # for caddn
            grid_size=model_info_dict['grid_size'],
            depth_downsample_factor=model_info_dict['depth_downsample_factor'],
            num_class=self.num_class,
            class_names=self.class_names
        )
        model_info_dict['num_point_features'] = vfe_module.get_output_feature_dim()
        model_info_dict['module_list'].append(vfe_module)
        return vfe_module, model_info_dict

    def build_image_backbone(self, model_info_dict):
        if self.model_cfg.get('IMAGE_BACKBONE', None) is None:
            return None, model_info_dict

        image_backbone_module = \
            image_backbone.__all__[self.model_cfg.IMAGE_BACKBONE.NAME](
                model_cfg=self.model_cfg.IMAGE_BACKBONE,
                num_class=self.num_class,
                class_names=self.class_names,
                tta=self.tta
            )
        if self.model_cfg.IMAGE_BACKBONE.NAME == 'BaseImageBackbone':
            model_info_dict['num_image_features'] = \
                image_backbone_module.get_output_feature_dim()
        model_info_dict['module_list'].append(image_backbone_module)

        return image_backbone_module, model_info_dict

    def build_depth_backbone(self, model_info_dict):
        if self.model_cfg.get('DEPTH_BACKBONE', None) is None:
            return None, model_info_dict

        depth_backbone_module = \
            depth_backbone.__all__[self.model_cfg.DEPTH_BACKBONE.NAME](
                model_cfg=self.model_cfg.DEPTH_BACKBONE,
                downsample_factor=model_info_dict['depth_downsample_factor'],
            )
        model_info_dict['module_list'].append(depth_backbone_module)
        return depth_backbone_module, model_info_dict

    def build_roi_heads(self, model_info_dict):
        if self.model_cfg.get('ROI_HEAD', None) is None:
            return None, model_info_dict

        roi_head_module = roi_heads.__all__[
            self.model_cfg.ROI_HEAD.NAME](
                model_cfg=self.model_cfg.ROI_HEAD,
                input_feat_dim=self.model_cfg.ROI_HEAD.INPUT_FEAT_DIM,
                num_class=self.num_class if not \
                    self.model_cfg.ROI_HEAD.CLASS_AGNOSTIC else 1,
                image_size=model_info_dict['image_size']
            )

        model_info_dict['module_list'].append(roi_head_module)
        return roi_head_module, model_info_dict

    @staticmethod
    def generate_recall_record(
            box_preds,
            recall_dict,
            batch_index,
            data_dict=None,
            thresh_list=None):
        if 'gt_boxes' not in data_dict:
            return recall_dict

        rois = data_dict['rois'][batch_index] if 'rois' in data_dict else None
        gt_boxes = data_dict['gt_boxes'][batch_index]

        if recall_dict.__len__() == 0:
            recall_dict = {'gt': 0}
            for cur_thresh in thresh_list:
                recall_dict['roi_%s' % (str(cur_thresh))] = 0
                recall_dict['rcnn_%s' % (str(cur_thresh))] = 0

        cur_gt = gt_boxes
        k = cur_gt.__len__() - 1
        while k > 0 and cur_gt[k].sum() == 0:
            k -= 1
        cur_gt = cur_gt[:k + 1]

        if cur_gt.shape[0] > 0:
            if box_preds.shape[0] > 0:
                iou3d_rcnn = iou3d_nms_utils.boxes_iou3d_gpu(
                    box_preds[:, 0:7], cur_gt[:, 0:7])
            else:
                iou3d_rcnn = torch.zeros((0, cur_gt.shape[0]))

            if rois is not None:
                iou3d_roi = iou3d_nms_utils.boxes_iou3d_gpu(
                    rois[:, 0:7], cur_gt[:, 0:7])

            for cur_thresh in thresh_list:
                if iou3d_rcnn.shape[0] == 0:
                    recall_dict['rcnn_%s' % str(cur_thresh)] += 0
                else:
                    rcnn_recalled = (
                        iou3d_rcnn.max(
                            dim=0)[0] > cur_thresh).sum().item()
                    recall_dict['rcnn_%s' % str(cur_thresh)] += rcnn_recalled
                if rois is not None:
                    roi_recalled = (
                        iou3d_roi.max(
                            dim=0)[0] > cur_thresh).sum().item()
                    recall_dict['roi_%s' % str(cur_thresh)] += roi_recalled

            recall_dict['gt'] += cur_gt.shape[0]
        else:
            gt_iou = box_preds.new_zeros(box_preds.shape[0])
        return recall_dict

    def load_params_from_file(self, filename, logger, to_cpu=False):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info(
            '==> Loading parameters from checkpoint %s to %s' %
            (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        model_state_disk = checkpoint['model_state']

        if 'version' in checkpoint:
            logger.info(
                '==> Checkpoint trained from version: %s' %
                checkpoint['version'])

        update_model_state = {}
        for key, val in model_state_disk.items():
            if key in self.state_dict() and self.state_dict()[
                    key].shape == model_state_disk[key].shape:
                update_model_state[key] = val
                # logger.info('Update weight %s: %s' % (key, str(val.shape)))

        state_dict = self.state_dict()
        state_dict.update(update_model_state)
        self.load_state_dict(state_dict)

        for key in state_dict:
            if key not in update_model_state:
                logger.info('Not updated weight %s: %s' %
                            (key, str(state_dict[key].shape)))

        logger.info('==> Done (loaded %d/%d)' %
                    (len(update_model_state), len(self.state_dict())))

    def load_params_with_optimizer(
            self, filename, to_cpu=False, optimizer=None, logger=None):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info(
            '==> Loading parameters from checkpoint %s to %s' %
            (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        epoch = checkpoint.get('epoch', -1)
        it = checkpoint.get('it', 0.0)

        self.load_state_dict(checkpoint['model_state'])

        if optimizer is not None:
            if 'optimizer_state' in checkpoint and checkpoint['optimizer_state'] is not None:
                logger.info(
                    '==> Loading optimizer parameters from checkpoint %s to %s' %
                    (filename, 'CPU' if to_cpu else 'GPU'))
                optimizer.load_state_dict(checkpoint['optimizer_state'])
            else:
                assert filename[-4] == '.', filename
                src_file, ext = filename[:-4], filename[-3:]
                optimizer_filename = '%s_optim.%s' % (src_file, ext)
                if os.path.exists(optimizer_filename):
                    optimizer_ckpt = torch.load(
                        optimizer_filename, map_location=loc_type)
                    optimizer.load_state_dict(
                        optimizer_ckpt['optimizer_state'])

        if 'version' in checkpoint:
            print(
                '==> Checkpoint trained from version: %s' %
                checkpoint['version'])
        logger.info('==> Done')

        return it, epoch

