import torch.nn as nn


class HeightCompression(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """
        if 'fusion_feature' not in batch_dict and 'encoded_spconv_tensorlist' not in batch_dict:
            encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']
            spatial_features = encoded_spconv_tensor.dense()
        elif 'encoded_spconv_tensorlist' in batch_dict:
            spatial_features = batch_dict['encoded_spconv_tensorlist'][0].dense()
            N, C, D, H, W = spatial_features.shape
            spatial_features = spatial_features.reshape(N, C * D, H, W)
            for each in batch_dict['encoded_spconv_tensorlist'][1:]:
                each = each.dense()
                N, C, D, H, W = each.shape
                each = each.reshape(N, C * D, H, W)
                spatial_features = spatial_features + each
            batch_dict['spatial_features'] = spatial_features
            batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride']
            return batch_dict
        else:
            spatial_features = batch_dict['fusion_feature']
        N, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.reshape(N, C * D, H, W)
        batch_dict['spatial_features'] = spatial_features
        batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride']
        return batch_dict
