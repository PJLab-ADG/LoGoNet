import numpy as np
import copy

from al3d_utils import common_utils


def random_flip_along_x(gt_boxes, points, return_enable_xy=False):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C)
    Returns:
    """
    enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
    if enable:
        gt_boxes[:, 1] = -gt_boxes[:, 1]
        gt_boxes[:, 6] = -gt_boxes[:, 6]
        points[:, 1] = -points[:, 1]

        if gt_boxes.shape[1] > 7:
            gt_boxes[:, 8] = -gt_boxes[:, 8]
    if return_enable_xy:
        return gt_boxes, points, int(enable)
    return gt_boxes, points


def random_flip_along_y(gt_boxes, points, return_enable_xy=False):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C)
    Returns:
    """
    enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
    if enable:
        gt_boxes[:, 0] = -gt_boxes[:, 0]
        gt_boxes[:, 6] = -(gt_boxes[:, 6] + np.pi)
        points[:, 0] = -points[:, 0]

        if gt_boxes.shape[1] > 7:
            gt_boxes[:, 7] = -gt_boxes[:, 7]
    if return_enable_xy:
        return gt_boxes, points, int(enable)
    return gt_boxes, points


def global_rotation(gt_boxes, points, rot_range, return_rotate_noise=False):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C),
        rot_range: [min, max]
    Returns:
    """
    noise_rotation = np.random.uniform(rot_range[0], rot_range[1])
    points = common_utils.rotate_points_along_z(points[np.newaxis, :, :], np.array([noise_rotation]))[0]
    gt_boxes[:, 0:3] = common_utils.rotate_points_along_z(gt_boxes[np.newaxis, :, 0:3], np.array([noise_rotation]))[0]
    gt_boxes[:, 6] += noise_rotation
    if gt_boxes.shape[1] > 7:
        gt_boxes[:, 7:9] = common_utils.rotate_points_along_z(
            np.hstack((gt_boxes[:, 7:9], np.zeros((gt_boxes.shape[0], 1))))[np.newaxis, :, :],
            np.array([noise_rotation])
        )[0][:, 0:2]
    if return_rotate_noise:
        return gt_boxes, points, noise_rotation
    return gt_boxes, points

def random_image_flip_horizontal(image, depth_map, gt_boxes, calib):
    """
    Performs random horizontal flip augmentation
    Args:
        image: (H_image, W_image, 3), Image
        depth_map: (H_depth, W_depth), Depth map
        gt_boxes: (N, 7), 3D box labels in LiDAR coordinates [x, y, z, w, l, h, ry]
        calib: calibration.Calibration, Calibration object
    Returns:
        aug_image: (H_image, W_image, 3), Augmented image
        aug_depth_map: (H_depth, W_depth), Augmented depth map
        aug_gt_boxes: (N, 7), Augmented 3D box labels in LiDAR coordinates [x, y, z, w, l, h, ry]
    """
    # Randomly augment with 50% chance
    enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])

    if enable:
        # Flip images
        aug_image = np.fliplr(image)
        aug_depth_map = np.fliplr(depth_map)
        
        # Flip 3D gt_boxes by flipping the centroids in image space
        aug_gt_boxes = copy.copy(gt_boxes)
        locations = aug_gt_boxes[:, :3]
        img_pts, img_depth = calib.lidar_to_img(locations)
        W = image.shape[1]
        img_pts[:, 0] = W - img_pts[:, 0]
        pts_rect = calib.img_to_rect(u=img_pts[:, 0], v=img_pts[:, 1], depth_rect=img_depth)
        pts_lidar = calib.rect_to_lidar(pts_rect)
        aug_gt_boxes[:, :3] = pts_lidar
        aug_gt_boxes[:, 6] = -1 * aug_gt_boxes[:, 6]

    else:
        aug_image = image
        aug_depth_map = depth_map
        aug_gt_boxes = gt_boxes

    return aug_image, aug_depth_map, aug_gt_boxes

def global_scaling(gt_boxes, points, scale_range, return_scale_noise=False):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading]
        points: (M, 3 + C),
        scale_range: [min, max]
    Returns:
    """
    if scale_range[1] - scale_range[0] < 1e-3:
        return gt_boxes, points
    noise_scale = np.random.uniform(scale_range[0], scale_range[1])
    points[:, :3] *= noise_scale
    gt_boxes[:, :6] *= noise_scale

    if gt_boxes.shape[1] > 7:
        gt_boxes[:, 7:9] *= noise_scale
    if return_scale_noise:
        return gt_boxes, points, noise_scale
    return gt_boxes, points

def random_image_pc_flip_horizontal(image, depth_map, gt_boxes, calib, points, gt_boxes2d = None, W = None):
    """
    Performs random horizontal flip and points augmentation
    Args:
        image: (H_image, W_image, 3), Image
        depth_map: (H_depth, W_depth), Depth map
        gt_boxes: (N, 7), 3D box labels in LiDAR coordinates [x, y, z, w, l, h, ry]
        calib: calibration.Calibration, Calibration object
        points:
        gt_boxes2d:
        W: original W before resize
    Returns:
        aug_image: (H_image, W_image, 3), Augmented image
        aug_depth_map: (H_depth, W_depth), Augmented depth map
        aug_gt_boxes: (N, 7), Augmented 3D box labels in LiDAR coordinates [x, y, z, w, l, h, ry]
        aug_points:
    """
    # Randomly augment with 50% chance
    enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
    W = image.shape[1] if W is None else W

    if enable:
        # Flip images
        aug_image = np.fliplr(image)

        # Flip depth maps
        if depth_map is not None:
            aug_depth_map = np.fliplr(depth_map)
        else:
            aug_depth_map = depth_map

        # Flip 2D boxes
        if gt_boxes2d is not None:
            aug_gt_boxes2d = copy.copy(gt_boxes2d)
            aug_gt_boxes2d[:,0] = W - aug_gt_boxes2d[:,0]
            aug_gt_boxes2d[:,2] = W - aug_gt_boxes2d[:,2]
        else:
            aug_gt_boxes2d = gt_boxes2d

        # Flip 3D gt_boxes by flipping the centroids in image space
        if gt_boxes is not None:
            aug_gt_boxes = copy.copy(gt_boxes)
            locations = aug_gt_boxes[:, :3]
            img_pts, img_depth = calib.lidar_to_img(locations)
            img_pts[:, 0] = W - img_pts[:, 0]
            pts_rect = calib.img_to_rect(u=img_pts[:, 0], v=img_pts[:, 1], depth_rect=img_depth)
            pts_lidar = calib.rect_to_lidar(pts_rect)
            aug_gt_boxes[:, :3] = pts_lidar
            aug_gt_boxes[:, 6] = -1 * aug_gt_boxes[:, 6]
        else:
            aug_gt_boxes = gt_boxes


        # Flip points
        aug_points = copy.copy(points)      ### lidar
        points_org = aug_points[:,:3]       ### cords
        points_trans, depth_trans = calib.lidar_to_img(points_org)  ### pc -> img
        points_trans[:, 0] = W - points_trans[:, 0]     ### img u,v flip
        points_trans_rect = calib.img_to_rect(u=points_trans[:, 0], v=points_trans[:, 1], depth_rect=depth_trans)   ### img -> rect
        points_lidar = calib.rect_to_lidar(points_trans_rect)   ### rect -> lidar
        aug_points[:, :3] = points_lidar        ### cords changed , r unchanged

    else:
        aug_image = image
        aug_depth_map = depth_map
        aug_gt_boxes = gt_boxes
        aug_points = points
        aug_gt_boxes2d = gt_boxes2d

    return aug_image, aug_depth_map, aug_gt_boxes, aug_points, aug_gt_boxes2d

def global_translation(gt_boxes, points, std, return_std_noise=False):
    x_trans = np.random.randn(1)*std
    y_trans = np.random.randn(1)*std
    z_trans = np.random.randn(1)*std

    points[:, 0] += x_trans
    points[:, 1] += y_trans
    points[:, 2] += z_trans

    gt_boxes[:, 0] += x_trans
    gt_boxes[:, 1] += y_trans
    gt_boxes[:, 2] += z_trans
    if return_std_noise:
        return gt_boxes, points, np.array([x_trans, y_trans, z_trans]).T
    return gt_boxes, points
