from functools import partial

import numpy as np

from al3d_utils import common_utils

from . import augmentor_utils, database_sampler, database_sampler_kitti, database_sampler_kittimm


class DataAugmentor(object):
    def __init__(self, root_path, augmentor_configs, class_names, use_image, logger=None):
        self.root_path = root_path
        self.class_names = class_names
        self.logger = logger
        self.use_image = use_image

        self.data_augmentor_queue = []
        aug_config_list = augmentor_configs if isinstance(augmentor_configs, list) \
            else augmentor_configs.AUG_CONFIG_LIST

        for cur_cfg in aug_config_list:
            if not isinstance(augmentor_configs, list):
                if cur_cfg.NAME in augmentor_configs.DISABLE_AUG_LIST:
                    continue
            cur_augmentor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_augmentor_queue.append(cur_augmentor)

    def gt_sampling(self, config=None):
        if not self.use_image:
            db_sampler = database_sampler.DataBaseSampler(
                root_path=self.root_path,
                sampler_cfg=config,
                class_names=self.class_names,
                logger=self.logger
            )
 
        return db_sampler
    def random_image_flip(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_image_flip, config=config)
        images = data_dict["images"]
        depth_maps = data_dict["depth_maps"]
        gt_boxes = data_dict['gt_boxes']
        gt_boxes2d = data_dict["gt_boxes2d"]
        calib = data_dict["calib"]
        for cur_axis in config['ALONG_AXIS_LIST']:
            assert cur_axis in ['horizontal']
            images, depth_maps, gt_boxes = getattr(augmentor_utils, 'random_image_flip_%s' % cur_axis)(
                images, depth_maps, gt_boxes, calib,
            )
        
        data_dict['images'] = images
        data_dict['depth_maps'] = depth_maps
        data_dict['gt_boxes'] = gt_boxes
        return data_dict
    def gt_samplingkitti(self, config=None):
        # if not self.use_image:
        db_sampler = database_sampler_kitti.DataBaseSamplerKITTI(
            root_path=self.root_path,
            sampler_cfg=config,
            class_names=self.class_names,
            logger=self.logger
        )
        return db_sampler
    def gt_samplingkittimm(self, config=None):
        # if not self.use_image:
        db_sampler = database_sampler_kittimm.DataBaseSamplerKITTIMM(
            root_path=self.root_path,
            sampler_cfg=config,
            class_names=self.class_names,
            logger=self.logger
        )
        return db_sampler

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    def random_world_flip(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_flip, config=config)
        gt_boxes, points = data_dict['gt_boxes'], data_dict['points']
        
        return_noise_flag = False
        if config.get('RETURN_NOISE_FLIP', None) is not None:
            if config['RETURN_NOISE_FLIP']:
                enable_xy = []
                return_noise_flag = True
        
        for cur_axis in config['ALONG_AXIS_LIST']:
            assert cur_axis in ['x', 'y']
            if return_noise_flag:
                gt_boxes, points, enable_curaxis = getattr(augmentor_utils, 'random_flip_along_%s' % cur_axis)(
                    gt_boxes, points, return_enable_xy=True
                )
                enable_xy.append(enable_curaxis)
            else:
                gt_boxes, points = getattr(augmentor_utils, 'random_flip_along_%s' % cur_axis)(
                    gt_boxes, points,
                )
        
        if return_noise_flag:
            if len(enable_xy)!=2 and cur_axis=='x':
                enable_xy.append(0)
            # print(enable_xy)
            flip_mat_T_inv = np.array(
                        [[[1, -1][enable_xy[1]], 0, 0],
                         [0, [1, -1][enable_xy[0]], 0],
                         [0, 0, 1]],
                        dtype=points.dtype,
                    )
            data_dict['aug_matrix_inv']['flip'] = flip_mat_T_inv
        
        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def random_world_rotation(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_rotation, config=config)
        rot_range = config['WORLD_ROT_ANGLE']
        if not isinstance(rot_range, list):
            rot_range = [-rot_range, rot_range]
        
        return_noise_flag = False
        if config.get('RETURN_NOISE_ROTATE', None) is not None:
            if config['RETURN_NOISE_ROTATE']:
                return_noise_flag = True
        
        if return_noise_flag:
            gt_boxes, points, noise_rotation = augmentor_utils.global_rotation(
                data_dict['gt_boxes'],
                data_dict['points'],
                rot_range=rot_range,
                return_rotate_noise=True
            )
            rot_sin = np.sin(noise_rotation*-1)
            rot_cos = np.cos(noise_rotation*-1)
            rot_mat_T_inv = np.array(
                    [[rot_cos, rot_sin, 0], [-rot_sin, rot_cos, 0], [0, 0, 1]],
                    dtype=points.dtype,
                )
            data_dict['aug_matrix_inv']['rotate'] = rot_mat_T_inv
        else:
            gt_boxes, points = augmentor_utils.global_rotation(
                data_dict['gt_boxes'], data_dict['points'], rot_range=rot_range
            )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def random_world_scaling(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_scaling, config=config)
        
        return_noise_flag = False
        if config.get('RETURN_NOISE_SCALE', None) is not None:
            if config['RETURN_NOISE_SCALE']:
                return_noise_flag = True
        
        if return_noise_flag:
            gt_boxes, points, noise_scale = augmentor_utils.global_scaling(
                data_dict['gt_boxes'],
                data_dict['points'],
                config['WORLD_SCALE_RANGE'],
                return_scale_noise=True
            )
            scale_mat_T_inv = np.array(
                    [[1/noise_scale, 0, 0], [0, 1/noise_scale, 0], [0, 0, 1/noise_scale]],
                    dtype=points.dtype,
                )
            data_dict['aug_matrix_inv']['rescale'] = scale_mat_T_inv
        else:
            gt_boxes, points = augmentor_utils.global_scaling(
                data_dict['gt_boxes'], data_dict['points'], config['WORLD_SCALE_RANGE']
            )
        
        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def random_world_translation(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_translation, config=config)
        
        return_noise_flag = False
        if config.get('RETURN_NOISE_TRANSLATION', None) is not None:
            if config['RETURN_NOISE_TRANSLATION']:
                return_noise_flag = True

        if return_noise_flag:
            gt_boxes, points, noise_trans_mat_T = augmentor_utils.global_translation(
                data_dict['gt_boxes'],
                data_dict['points'],
                config['STD'],
                return_std_noise=True
            )
            trans_mat_T_inv = noise_trans_mat_T * -1
            data_dict['aug_matrix_inv']['translate'] = trans_mat_T_inv
        else:
            gt_boxes, points = augmentor_utils.global_translation(
                data_dict['gt_boxes'], data_dict['points'], config['STD']
            )
        
        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict
    def random_image_pc_flip(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_image_pc_flip, config=config)

        # imgs, points, calib are required
        images = data_dict["images"]
        points = data_dict["points"]
        calib = data_dict["calib"]
        # optional
        gt_boxes = data_dict.get('gt_boxes',None)
        depth_maps = data_dict.get('depth_maps',None)
        gt_boxes2d = data_dict.get('gt_boxes2d',None)

        # used when input image is reshaped
        W = data_dict.get('image_shape',None)
        if W is not None:
            W = W[1]

        # add W and gt box 2D
        for cur_axis in config['ALONG_AXIS_LIST']:
            assert cur_axis in ['horizontal']
            images, depth_maps, gt_boxes, points, gt_boxes2d = getattr(augmentor_utils, 'random_image_pc_flip_%s' % cur_axis)(
                images, depth_maps, gt_boxes, calib, points, gt_boxes2d, W
            )

        data_dict['images'] = images
        data_dict['points'] = points
        if gt_boxes is not None:
            data_dict['gt_boxes'] = gt_boxes
        if depth_maps is not None:
            data_dict['depth_maps'] = depth_maps
        if gt_boxes2d is not None:
            data_dict['gt_boxes2d'] = gt_boxes2d

        return data_dict

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7) [x, y, z, dx, dy, dz, heading]
                gt_names: optional, (N), string
                ...

        Returns:
        """
        data_dict['aug_matrix_inv'] = {}
        for cur_augmentor in self.data_augmentor_queue:
            data_dict = cur_augmentor(data_dict=data_dict)

        data_dict['gt_boxes'][:, 6] = common_utils.limit_period(
            data_dict['gt_boxes'][:, 6], offset=0.5, period=2 * np.pi
        )
        if len(data_dict['aug_matrix_inv'])==0:
            data_dict.pop('aug_matrix_inv')
        # if 'calib' in data_dict:
        #     data_dict.pop('calib')
        if 'road_plane' in data_dict:
            data_dict.pop('road_plane')
        if 'gt_boxes_mask' in data_dict:
            gt_boxes_mask = data_dict['gt_boxes_mask']
            data_dict['gt_boxes'] = data_dict['gt_boxes'][gt_boxes_mask]
            data_dict['gt_names'] = data_dict['gt_names'][gt_boxes_mask]
            data_dict.pop('gt_boxes_mask')
        return data_dict
