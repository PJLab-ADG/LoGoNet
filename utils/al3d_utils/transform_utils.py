import copy
import numpy as np

import torch


def get_inverse_transform_mat(src_pose):

    reverse_pose = np.zeros((4, 4), dtype=np.float32)
    reverse_pose[:3, :3] = src_pose[:3, :3].T
    reverse_pose[:3, 3:] = -(src_pose[:3, :3].T @ src_pose[:3, 3:])
    reverse_pose[3, 3] = 1

    return reverse_pose

def coordinate_transform(pts, est_bbox, gt_bbox):

    B = pts.shape[0]
    angle = est_bbox[:, 6]

    center_point = est_bbox[:, :3]
    new_points = copy.deepcopy(pts)
    new_points[:, :, :3] -= center_point.reshape(B, 1, -1)

    cosa = torch.cos(-angle)
    sina = torch.sin(-angle)
    zeros = angle.new_zeros(B)
    ones = angle.new_ones(B)
    rot_matrix = torch.stack((
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).reshape(-1, 3, 3).float()
    pts = torch.matmul(pts[:, :, 0:3], rot_matrix)

    new_bbox = copy.deepcopy(gt_bbox)
    new_bbox[:, :3] -= center_point

    new_bbox[:, 6] -= est_bbox[:, 6]
    new_bbox[new_bbox[:, 6] >= np.pi] -= np.pi*2
    new_bbox[new_bbox[:, 6] < -np.pi] += np.pi*2

    return new_points, new_bbox
    
def revert_to_each_frame(data_dict): # data_dict has multi samples, while one sample with multi-frame
    sequence_list = []
    sequence_list_gt = []
    # deal with every sample
    for i, pose in enumerate(data_dict['poses']): # pose: [tracking_sequence_length, 4, 4]
        initial_box = copy.deepcopy(data_dict['batch_init_box'][i])
        initial_box[:3] = 0
        initial_box[6] = 0

        # if reg2, transform box from the first regression box coordinate to the initial box coordinate
        # box_preds_world = box_coordinate_transform(data_dict['pred_boxes'][i], data_dict['batch_reg1_box'][i])

        # transform box from initial box coordinate to world coordinate
        box_preds_world = box_coordinate_transform(data_dict['pred_boxes'][i], data_dict['batch_init_box'][i])
        box_gt_world = box_coordinate_transform(data_dict['gt_boxes'][i], data_dict['batch_init_box'][i])

        # from al3d_dlb.utils import box_vis_for_debug
        # boxes_vis = np.stack((box_preds_world, box_preds_pred), axis=0)
        # box_vis_for_debug.vis_boxes(boxes_vis, None)

        heading_offset = np.arctan2(pose[:,1,0], pose[:,0,0])
        r_t = np.linalg.inv(pose) #[tracking_sequence_length, 4, 4]

        boxes_h = np.hstack((box_preds_world[:3], np.ones((1)))) # 4,
        box_center_in_frames = r_t @ boxes_h.T # [tracking_sequence_length, 4]
        box_center_in_frames = box_center_in_frames[:, :3]
        w_h_l = np.repeat(np.expand_dims(box_preds_world[3:6], axis=0), pose.shape[0], axis=0) # [tracking_sequence_length, 3]
        new_heading = box_preds_world[6] - np.expand_dims(heading_offset, axis=1)
        one_sample_muli_frame_boxes_in_lidar = np.concatenate((box_center_in_frames, w_h_l, new_heading),axis=1) # [tracking_sequence_length, 7]
        sequence_list.append(one_sample_muli_frame_boxes_in_lidar)

        boxes_h_gt = np.hstack((box_gt_world[:3], np.ones((1)))) # 4,
        box_center_in_frames_gt = r_t @ boxes_h_gt.T
        box_center_in_frames_gt = box_center_in_frames_gt[:, :3]
        w_h_l_gt = np.repeat(np.expand_dims(box_gt_world[3:6], axis=0), pose.shape[0], axis=0)
        new_heading_gt = box_gt_world[6] - np.expand_dims(heading_offset, axis=1)
        one_sample_muli_frame_boxes_in_lidar_gt = np.concatenate((box_center_in_frames_gt, w_h_l_gt, new_heading_gt),axis=1) # [tracking_sequence_length, 7]
        sequence_list_gt.append(one_sample_muli_frame_boxes_in_lidar_gt)

        # from al3d_dlb.utils import box_vis_for_debug
        # boxes_vis = np.concatenate((one_sample_muli_frame_boxes_in_lidar, one_sample_muli_frame_boxes_in_lidar_gt),axis=0)
        # box_vis_for_debug.vis_boxes(boxes_vis, None)
    return sequence_list, sequence_list_gt


def revert_to_each_frame_old(data_dict):
    #将结果还原到每一帧上,一个batch只有一个目标，因此预测结果只有一个，但是一个batch有多帧
    pred_dicts = {}
    data_dict['batch_initial_box'] #世界坐标系下参考box的位置和角度
    data_dict['batch_box_preds'] #参考box坐标系下回归的box新位置

    # 要做两次box_coordinate_transform才能回到世界坐标系，一次是从reg1_box坐标系 -> tracl_box坐标系，第二次是track_box坐标系 -> world坐标系
    batch_box_preds_track_box = box_coordinate_transform(data_dict['batch_box_preds'], data_dict['batch_initial_box']) #把预测的box转到世界坐标系下 B,7
    batch_box_preds_world = box_coordinate_transform(data_dict['batch_box_preds'], data_dict['batch_reg1_box']) #把预测的box转到世界坐标系下 B,7
    boxes_h = np.hstack((batch_box_preds_world[:,:3], np.ones((batch_box_preds_world.shape[0], 1)))) #B, 4
    sequence_list = []
    #再用每一帧的pose信息将世界坐标系下的静态框还原到每一帧车的坐标系下
    for i, pose in enumerate(data_dict['poses']): # data_dict['poses']: [32, 796, 4]
        # pts_h = np.hstack((pts[pts[:,1]>10], np.ones((pts[pts[:,1]>10].shape[0], 1))))
        # r_t = np.linalg.inv(pose_list[-1]).T
        # pts_verhicle = pts_h @ r_t
        # heading_offset = math.atan2(poses[i,1,0], poses[i,0,0])
        heading_offset = np.arctan2(pose[:,1,0], pose[:,0,0]) #计算heading角偏移 再减去这个heading角 [sequence length, ]
        r_t = np.linalg.inv(pose).T #[sequence length, 4, 4]
        box_center_verhicle = boxes_h[i] @ r_t # [sequence, 1, 4]
        box_center_verhicle = np.squeeze(box_center_verhicle)[:3] # [sequence, 3]
        w_h_l = batch_box_preds_world[i, 3:6]
        w_h_l = np.repeat(w_h_l, pose.shape[0], axis=0) # [sequence, 3]
        new_heading = batch_box_preds_world[6] - np.expand_dims(heading_offset, axis=1) # [sequence, 1]这儿的加减待确定
        one_sequence_boxes = np.concatenate((box_center_verhicle, w_h_l, new_heading),axis=1)
        sequence_list.append(one_sequence_boxes)
    pred_dicts['sequence_list'] = sequence_list

def box_coordinate_transform(boxes, initial_box):
    # boxes:处在initial_box为原点的坐标系下 [B, 7]
    # initial_box:参考框在世界坐标系下的朝向和位置

    boxes[:3] = boxes[:3] @ np.linalg.inv(rotate_yaw(initial_box[6]).T)
    boxes[:3] = boxes[:3] + initial_box[:3]

    boxes[6] = boxes[6] + initial_box[6]
    # for box in boxes:
    if boxes[6] >= np.pi: boxes[6] -= np.pi*2
    if boxes[6] < -np.pi: boxes[6] += np.pi*2

    return boxes

def rotate_yaw(yaw):
    return np.array([[np.cos(yaw), np.sin(yaw), 0],
                    [-np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]], dtype=np.float32)

def limit_heading(angle):
    angle = angle % (2*np.pi)
    if angle >= np.pi: angle -= np.pi*2
    if angle < -np.pi: angle += np.pi*2
    if angle >= np.pi/2: angle -= np.pi
    if angle < -np.pi/2: angle += np.pi
    return angle
