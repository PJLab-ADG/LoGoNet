import numpy as np
import open3d as o3d
from open3d import geometry
import torch
import sys

def draw_bboxes(bbox3d,
                 pcd=None,
                 bbox_color=(0, 0, 1),
                 points_in_box_color=(1, 0, 0),
                 rot_axis=2,
                 center_mode='lidar_bottom',
                 mode='xyz'):
    """Draw bbox on visualizer and change the color of points inside bbox3d.
    Args:
        bbox3d (numpy.array | torch.tensor, shape=[M, 7]):
            3d bbox (x, y, z, dx, dy, dz, yaw) to visualize.
        vis (:obj:`open3d.visualization.Visualizer`): open3d visualizer.
        points_colors (numpy.array): color of each points.
        pcd (:obj:`open3d.geometry.PointCloud`): point cloud. Default: None.
        bbox_color (tuple[float]): the color of bbox. Default: (0, 1, 0).
        points_in_box_color (tuple[float]):
            the color of points inside bbox3d. Default: (1, 0, 0).
        rot_axis (int): rotation axis of bbox. Default: 2.
        center_mode (bool): indicate the center of bbox is bottom center
            or gravity center. avaliable mode
            ['lidar_bottom', 'camera_bottom']. Default: 'lidar_bottom'.
        mode (str):  indicate type of the input points, avaliable mode
            ['xyz', 'xyzrgb']. Default: 'xyz'.
    """
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0,0,0]) # bbox3d[0, :3].tolist()
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    if isinstance(bbox3d, torch.Tensor):
        bbox3d = bbox3d.cpu().numpy()
    bbox3d = bbox3d.copy()

    in_box_color = np.array(points_in_box_color)
    for i in range(len(bbox3d)):
        center = bbox3d[i, 0:3]
        dim = bbox3d[i, 3:6]
        yaw = np.zeros(3)
        yaw[rot_axis] = bbox3d[i, 6]
        rot_mat = geometry.get_rotation_matrix_from_xyz(yaw)
        # print('rot_mat: ', rot_mat)

        if center_mode == 'lidar_bottom':
            center[rot_axis] += dim[
                rot_axis] / 2  # bottom center to gravity center
        elif center_mode == 'camera_bottom':
            center[rot_axis] -= dim[
                rot_axis] / 2  # bottom center to gravity center
        box3d = geometry.OrientedBoundingBox(center, rot_mat, dim)

        line_set = geometry.LineSet.create_from_oriented_bounding_box(box3d)
        if i > 0:
            bbox_color = (1, 0, 0)
        # if i >= len(bbox3d)//2:
        #     bbox_color = (1, 0, 0)
        
        # if i < len(bbox3d)//3:
        #     bbox_color = (1, 0, 0)
        # elif i >= 2*len(bbox3d)//3:
        #     bbox_color = (0, 1, 0)
        # else:
        #     bbox_color = (0, 0, 1)
        line_set.paint_uniform_color(bbox_color)
        # draw bboxes on visualizer
        vis.add_geometry(line_set)
    vis.add_geometry(pcd)
    vis.add_geometry(mesh_frame)

    # # update points colors
    # if pcd is not None:
    #     # pcd.colors = o3d.utility.Vector3dVector(points_colors)
    #     vis.update_geometry(pcd)
    vis.run()

def draw_points(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=points[0,:3].tolist())
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.add_geometry(mesh_frame)
    vis.run()

def rotate_yaw(yaw):
    # YAW角转旋转矩阵
    return np.array([[np.cos(yaw), np.sin(yaw), 0],
                    [-np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]], dtype=np.float32)

def vis_gt_pred(gt_boxes, box_preds, points):
    # gt_boxes: [batch_size, cc, 7]
    # box_preds: [batch_size, box_num, 7]
    # points: [batch_size, points_num, 3]
    gt_boxes[:, 2] -= 0.5*gt_boxes[:, 5] #box中心点转为底面中心点
    box_preds[:, 2] -= 0.5*box_preds[:, 5] #box中心点转为底面中心点
    pcd = o3d.geometry.PointCloud()
    for i in range(len(gt_boxes)):
        pcd.points = o3d.utility.Vector3dVector(points[i])
        vis_boxes = np.concatenate((np.expand_dims(gt_boxes[i], axis=0), np.expand_dims(box_preds[i], axis=0)), axis =0 )
        draw_bboxes(vis_boxes, pcd) #函数输入box点用底面点表示

def vis_boxes(boxes, points):
    boxes[:, 2] -= 0.5*boxes[:, 5] #box中心点转为底面中心点
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    draw_bboxes(boxes, pcd)

def vis_boxes_with_points_color(boxes, points, fg_label):
    # points: [N, 3]
    # fg_label: [N,1]
    points_colors = np.array([[0.8, 0.2, 0.6]]).repeat(4096, axis = 0)
    points_colors[:, 0] = points_colors[:, 0]*fg_label.squeeze()

    boxes[:, 2] -= 0.5*boxes[:, 5] #box中心点转为底面中心点
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(points_colors)
    draw_bboxes(boxes, pcd)

if __name__ == "__main__":
    points = np.load(sys.argv[1])
    gt_boxes = np.load(sys.argv[2])
    box_preds = np.load(sys.argv[3])
    
    gt_boxes[:, 2] -= 0.5*gt_boxes[:, 5] #box中心点转为底面中心点
    box_preds[:, 2] -= 0.5*box_preds[:, 5] #box中心点转为底面中心点

    vis_gt_pred(gt_boxes, box_preds, points)
