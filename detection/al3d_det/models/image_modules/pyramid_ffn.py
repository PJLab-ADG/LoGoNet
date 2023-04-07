import torch
import torch.nn as nn
import torch.nn.functional as F

from al3d_det.models.image_modules.ifn.basic_blocks import BasicBlock2D
from al3d_det.models.image_modules import ifn

class PyramidFeat2D(nn.Module):

    def __init__(self, model_cfg):
        """
        Initialize 2D feature network via pretrained model
        Args:
            model_cfg: EasyDict, Dense classification network config
        """
        super().__init__()
        self.model_cfg = model_cfg
        self.is_optimize = model_cfg.OPTIMIZE

        # Create modules
        self.ifn = ifn.__all__[model_cfg.IFN.NAME](
            num_classes=model_cfg.IFN.NUM_CLASS,
            backbone_name=model_cfg.IFN.BACKBONE,
            **model_cfg.IFN.ARGS
        )
        self.reduce_blocks = torch.nn.ModuleList()
        self.out_channels = {}
        for _idx, _channel in enumerate(model_cfg.IFN.CHANNEL_REDUCE["in_channels"]):
            _channel_out = model_cfg.IFN.CHANNEL_REDUCE["out_channels"][_idx]
            self.out_channels[model_cfg.IFN.ARGS['feat_extract_layer'][_idx]] = _channel_out
            block_cfg = {"in_channels": _channel,
                         "out_channels": _channel_out,
                         "kernel_size": model_cfg.IFN.CHANNEL_REDUCE["kernel_size"][_idx],
                         "stride": model_cfg.IFN.CHANNEL_REDUCE["stride"][_idx],
                         "bias": model_cfg.IFN.CHANNEL_REDUCE["bias"][_idx]}
            self.reduce_blocks.append(BasicBlock2D(**block_cfg))

    def get_output_feature_dim(self):
        return self.out_channels

    def forward(self, batch_dict):
        """
        Predicts depths and creates image depth feature volume using depth distributions
        Args:
            images: (N, H_in, W_in, 3), Input images
        Returns:
            batch_dict:
                frustum_features: (N, C, D, H_out, W_out), Image depth features
        """
        # Pixel-wise depth classification
        data = batch_dict['images']
        bs = batch_dict['batch_size']
        batch_dict['image_features'] = {}
        image_list = []
        
        for cam in data.keys():
            single_result = {}
            image_list.append(data[cam].unsqueeze(1))
        images = torch.cat(image_list, dim=1)
        B, N, C, H, W = images.shape
        images = images.reshape(B*N, C, H, W)
        ifn_result = self.ifn(images)
        for _idx, _layer in enumerate(self.model_cfg.IFN.ARGS['feat_extract_layer']):
            image_features = ifn_result[_layer]
            if self.reduce_blocks[_idx] is not None:
                image_features = self.reduce_blocks[_idx](image_features)
            
            single_result[_layer+"_feat2d"] = image_features
        if self.training:
            # detach feature from graph if not optimize
            if "logits" in ifn_result:
                ifn_result["logits"].detach_()
            if not self.is_optimize:
                image_features.detach_()

        for layer in single_result.keys():
            if layer not in batch_dict['image_features'].keys():
                batch_dict['image_features'][layer] = {}
            for i in range(5):
                cam = 'camera_{}'.format(i)
                batch_dict['image_features'][layer][cam] = single_result[layer][i*bs:(i+1)*bs]
        # import pdb; pdb.set_trace()
        return batch_dict

    def get_loss(self):
        """
        Gets loss
        Args:
        Returns:
            loss: (1), Network loss
            tb_dict: dict[float], All losses to log in tensorboard
        """
        return None, None