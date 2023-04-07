from mmdet.models.builder import build_backbone, build_neck
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from kornia import normalize
except:
    pass
from collections import OrderedDict
from al3d_det.models.image_modules.ifn.basic_blocks import BasicBlock2D
class MMDETFPNKITTI(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        self.imgconfig = model_cfg.IMGCONFIG
        self.fpnconfig = model_cfg.FPNCONFIG
        self.img_backbone = build_backbone(self.imgconfig)
        if self.model_cfg.get('PRETRAINEDPATH', None) is not None:
            self.img_backbone.init_weights(self.model_cfg.PRETRAINEDPATH)
        if self.imgconfig.get('pretrained', None) is not None: 
            self.img_backbone.init_weights(self.imgconfig.pretrained)

        self.neck = build_neck(self.fpnconfig)
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
        Forward pass
        Args:
            images: (N, 3, H_in, W_in), Input images
        Returns
            result: dict[torch.Tensor], Depth distribution result
                features: (N, C, H_out, W_out), Image features
                logits: (N, num_classes, H_out, W_out), Classification logits
                aux: (N, num_classes, H_out, W_out), Auxillary classification logits
        """

        # Extract features
        result = OrderedDict()
        images = batch_dict['images']
        bs = batch_dict['batch_size']
        batch_dict['image_features'] = {}
        single_result = {}
        B, C, H, W = images.shape
        x = self.img_backbone(images)
        x_neck = self.neck(x)
        for _idx, _layer in enumerate(self.model_cfg.IFN.ARGS['feat_extract_layer']):
            image_features = x_neck[_idx]
            if self.reduce_blocks[_idx] is not None:
                image_features = self.reduce_blocks[_idx](image_features)
            single_result[_layer+"_feat2d"] = image_features
        for layer in single_result.keys():
            if layer not in batch_dict['image_features'].keys():
                batch_dict['image_features'][layer] = {}
            batch_dict['image_features'][layer]= single_result[layer]
        return batch_dict

    def preprocess(self, images):
        """
        Preprocess images
        Args:
            images: (N, 3, H, W), Input images
        Return
            x: (N, 3, H, W), Preprocessed images
        """
        x = images
        # if self.pretrained:
        #     # Create a mask for padded pixels
        #     mask = torch.isnan(x)

        #     # Match ResNet pretrained preprocessing
        #     x = normalize(x, mean=self.norm_mean, std=self.norm_std)

        #     # Make padded pixels = 0
        #     x[mask] = 0

        return x