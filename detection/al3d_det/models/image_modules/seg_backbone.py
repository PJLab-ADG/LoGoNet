import torch
import torch.nn as nn

from al3d_det.models.image_modules import semanticseg

class SegIFN(nn.Module):

    def __init__(self, model_cfg):
        """
        Initialize Image Segmentation feature network
        Args:
            model_cfg: EasyDict, Depth classification network config
        """
        super().__init__()
        self.model_cfg = model_cfg

        # Create modules
        self.pyramid_ffn = semanticseg.__all__[model_cfg.DNN.NAME](
            optimize=model_cfg.DNN.OPTIMIZE,
            model_cfg=model_cfg.DNN
        )
        
    def get_output_feature_dim(self):
        return 

    def forward(self, batch_dict):
        """
        Predicts depths and creates image depth feature volume using depth distributions
        Args:
            batch_dict:
                images: (N, 3, H_in, W_in), Input images
        Returns:
            batch_dict:
                frustum_features: (N, C, D, H_out, W_out), Image depth features
        """
        # Pixel-wise depth classification
        images = batch_dict["images"]
        ffn_result = self.pyramid_ffn(images)
        image_features = ffn_result['layer1_feat2d']
        batch_dict["image_features2d"] = image_features

        return batch_dict

    def create_frustum_features(self, image_features, depth_logits):
        """
        Create image depth feature volume by multiplying image features with depth distributions
        Args:
            image_features: (N, C, H, W), Image features
            depth_logits: (N, D+1, H, W), Depth classification logits
        Returns:
            frustum_features: (N, C, D, H, W), Image features
        """
        channel_dim = 1
        depth_dim = 2

        # Resize to match dimensions
        image_features = image_features.unsqueeze(depth_dim)
        depth_logits = depth_logits.unsqueeze(channel_dim)

        # Apply softmax along depth axis and remove last depth category (> Max Range)
        depth_probs = F.softmax(depth_logits, dim=depth_dim)
        depth_probs = depth_probs[:, :, :-1]

        # Multiply to form image depth feature volume
        frustum_features = depth_probs * image_features
        return frustum_features