import os
from collections import OrderedDict
import torch
import torch.nn as nn
import numpy as np
import spconv.pytorch as spconv

from al3d_utils import common_utils
from al3d_utils.ops.iou3d_nms import iou3d_nms_utils

from al3d_det.models import fusion_modules
from .anchor_kitti import ANCHORKITTI
from al3d_det.utils import nms_utils
from al3d_det.models import image_modules as img_modules
from al3d_det.models import modules as cp_modules

class ANCHORKITTIMM_LiDAR(ANCHORKITTI):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg, num_class, dataset)

    def forward(self, batch_dict, cur_module=None, end=False):
        if not end:
            return cur_module(batch_dict)
        else:
            if self.training:
                loss, tb_dict, disp_dict = self.get_training_loss()

                ret_dict = {
                    'loss': loss
                }

                return ret_dict, tb_dict, disp_dict
            else:
                pred_dicts, recall_dicts = self.post_processing(batch_dict)
                return pred_dicts, recall_dicts

class ANCHORKITTIMM_Camera(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.img_backbone = img_modules.__all__[model_cfg.IMAGE_BACKBONE.NAME](
            model_cfg=model_cfg.IMAGE_BACKBONE
        )
        if 'IMGPRETRAINED_MODEL' in model_cfg.IMAGE_BACKBONE and model_cfg.IMAGE_BACKBONE.IMGPRETRAINED_MODEL is not None:
            checkpoint= torch.load(model_cfg.IMAGE_BACKBONE.IMGPRETRAINED_MODEL, map_location='cpu')
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}
            ckpt = state_dict
            new_ckpt = OrderedDict()
            for k, v in ckpt.items():
                if k.startswith('backbone'):
                    new_v = v
                    new_k = k.replace('backbone.', 'img_backbone.')
                else:
                    continue
                new_ckpt[new_k] = new_v
            self.img_backbone.load_state_dict(new_ckpt, strict=False)


class ANCHORMMKITTI(nn.Module):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__()
        self.lidar = ANCHORKITTIMM_LiDAR(
            model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.camera = ANCHORKITTIMM_Camera(model_cfg)
        self.training = self.lidar.training
        self.second_stage = self.lidar.second_stage
        self.grid_size = self.lidar.dataset.grid_size[::-1] + [1, 0, 0]
        voxel_size  = self.lidar.dataset.voxel_size
        point_cloud_range = self.lidar.dataset.point_cloud_range
        self.freeze_img = model_cfg.IMAGE_BACKBONE.get('FREEZE_IMGBACKBONE', False)
        self.freeze()
    def freeze(self):
        if self.freeze_img:
            for param in self.camera.img_backbone.img_backbone.parameters():
                param.requires_grad = False

            for param in self.camera.img_backbone.neck.parameters():
                param.requires_grad = False
    def forward(self, batch_dict):
        batch_dict = self.camera.img_backbone(batch_dict)
        batch_dict = self.lidar(batch_dict, self.lidar.module_list[0])
        batch_dict = self.lidar(batch_dict, self.lidar.module_list[1])
        batch_dict = self.lidar(batch_dict, self.lidar.module_list[2])
        batch_dict = self.lidar(batch_dict, self.lidar.module_list[3])
        batch_dict = self.lidar(batch_dict, self.lidar.module_list[4])
        if self.second_stage:
            batch_dict = self.lidar(batch_dict, self.lidar.module_list[5])
        ret_lidar = self.lidar(batch_dict, end=True)

        return ret_lidar

    def update_global_step(self):
        if hasattr(self.lidar, 'update_global_step'):
            self.lidar.update_global_step()
        else:
            self.module.lidar.update_global_step()

    def load_params_from_file(self, filename, logger, to_cpu=False):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' %
                    (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        model_state_disk = checkpoint['model_state']

        if 'version' in checkpoint:
            logger.info('==> Checkpoint trained from version: %s' %
                        checkpoint['version'])

        update_model_state = {}
        for key, val in model_state_disk.items():
            if key in self.state_dict() and self.state_dict()[key].shape == model_state_disk[key].shape:
                update_model_state[key] = val

        state_dict = self.state_dict()
        state_dict.update(update_model_state)
        self.load_state_dict(state_dict)

        for key in state_dict:
            if key not in update_model_state:
                logger.info('Not updated weight %s: %s' %
                            (key, str(state_dict[key].shape)))

        logger.info('==> Done (loaded %d/%d)' %
                    (len(update_model_state), len(self.state_dict())))

    def load_params_with_optimizer(self, filename, to_cpu=False, optimizer=None, logger=None):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' %
                    (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        epoch = checkpoint.get('epoch', -1)
        it = checkpoint.get('it', 0.0)

        self.load_state_dict(checkpoint['model_state'])

        if optimizer is not None:
            if 'optimizer_state' in checkpoint and checkpoint['optimizer_state'] is not None:
                logger.info('==> Loading optimizer parameters from checkpoint %s to %s'
                            % (filename, 'CPU' if to_cpu else 'GPU'))
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
            print('==> Checkpoint trained from version: %s' %
                  checkpoint['version'])
        logger.info('==> Done')

        return it, epoch

    
