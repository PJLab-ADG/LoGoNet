from collections import namedtuple

import numpy as np
import torch
import kornia
from .centerpoint_waymo import CenterPointPC
from .centerpoint_MM_waymo import CenterPointMM
from .anchor_kitti import ANCHORKITTI
from .anchor_MM_kitti import ANCHORMMKITTI
__all__ = {
    'CenterPointPC': CenterPointPC,
    'CenterPointMM': CenterPointMM,
    'ANCHORKITTI': ANCHORKITTI,
    'ANCHORMMKITTI': ANCHORMMKITTI,
}


def build_network(model_cfg, num_class, dataset):
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg,
        num_class=num_class,
        dataset=dataset
    )
    return model

def load_data_to_gpu(batch_dict):
    for key, val in batch_dict.items():
        if key in ['frame_id', 'sequence_name', 'pose', 'tta_ops', 'aug_matrix_inv','db_flag', 'frame_id', 'metadata', 'calib', 'sequence_name']:
            continue
        elif key in ['images', 'extrinsic', 'intrinsic']:
            temp = {}
            for cam in val.keys():
                temp[cam] = torch.from_numpy(val[cam]).float().cuda().contiguous()
            batch_dict[key] = temp
        elif key in ['image_shape']:
            temp = {}
            for cam in val.keys():
                temp[cam] = torch.from_numpy(val[cam]).int().cuda()
            batch_dict[key] = temp
        elif isinstance(val, np.ndarray):
            batch_dict[key] = torch.from_numpy(val).float().cuda()
        else:
            continue
            
def load_data_to_gpukitti(batch_dict):
    for key, val in batch_dict.items():
        if not isinstance(val, np.ndarray):
            continue
        elif key in ['frame_id', 'metadata', 'calib', 'pose', 'tta_ops', 'aug_matrix_inv','db_flag', 'frame_id', 'metadata', 'calib', 'sequence_name']:
            continue
        elif key in ['images']:
            batch_dict[key] = kornia.image_to_tensor(val).float().cuda().contiguous()
        elif key in ['image_shape']:
            batch_dict[key] = torch.from_numpy(val).int().cuda()
        elif isinstance(val, np.ndarray):
            batch_dict[key] = torch.from_numpy(val).float().cuda()
        else:
            continue
def model_fn_decorator():
    ModelReturn = namedtuple('ModelReturn', ['loss', 'tb_dict', 'disp_dict'])

    def model_func(model, batch_dict):
        if 'calib' in batch_dict.keys():
            load_data_to_gpukitti(batch_dict)
        else:
            load_data_to_gpu(batch_dict)
        ret_dict, tb_dict, disp_dict = model(batch_dict)

        loss = ret_dict['loss'].mean()
        if hasattr(model, 'update_global_step'):
            model.update_global_step()
        else:
            model.module.update_global_step()

        return ModelReturn(loss, tb_dict, disp_dict)

    return model_func
