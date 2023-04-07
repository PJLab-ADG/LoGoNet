from re import L
import numpy as np
import pathlib as Path

from ml3d.vis import Visualizer
from ml3d.vis.boundingbox import BoundingBox3D
from ml3d.datasets.base_dataset import BaseDataset

label_to_names = {
            0: 'GT',
            1: 'DETECTION',
            2: 'TRACKING'}


class DataCollect(BaseDataset):
    def __init__(self, dataset_path='Waymo', name='Waymo'):
        super().__init__(dataset_path=dataset_path, name=name)
        
        cfg = self.cfg

        self.name = cfg.name
        self.dataset_path = cfg.dataset_path
        self.num_classes = 3
        self.label_to_names = self.get_label_to_names()

        self.datas = []
        self.labels = []

    def get_infos(self, pts_list, gt_infos=None, det_info=None, track_info=None):
        for i, pts in enumerate(pts_list):
            pts.astype(np.float32)
            self.datas.append(pts)
            label = []
            if gt_infos is not None:
                gt_infos[i].astype(np.float32)
                label.append(gt_infos[i])
            if det_info is not None:
                det_info[i].astype(np.float32)
                label.append(det_info[i])
            if track_info is not None:
                track_info[i].astype(np.float32)
                label.append(track_info[i])
            
            if len(label) != 0:
                label = np.concatenate(label, axis=0)
            self.labels.append(label)
    
    @staticmethod
    def get_label_to_names():
        """Returns a label to names dictonary object.

        Returns:
            A dict where keys are label numbers and
            values are the corresponding names.
        """
        label_to_names = {
            0: 'GT',
            1: 'DETECTION',
            2: 'TRACKING',
        }
        return label_to_names
    
    def is_tested(self, attr):
        """Checks whether a datum has been tested.

        Args:
            attr: The attributes associated with the datum.

        Returns:
            This returns True if the test result has been stored for the datum with the
            specified attribute; else returns False.
        """
        return False

    def save_test_result(self, results, attr):
        """Saves the output of a model.

        Args:
            results: The output of a model for the datum associated with the attribute passed.
            attr: The attributes that correspond to the outputs passed in results.
        """
        return
    
    # @staticmethod
    # def read_lidar(path):
    #     """Reads lidar data from the path provided.

    #     Returns:
    #         A data object with lidar information.
    #     """
    #     assert Path(path).exists()

    #     return np.fromfile(path, dtype=np.float32).reshape(-1, 6)
    
    @staticmethod
    def read_label(labels):
        """Reads labels of bound boxes.

        Returns:
            The data objects with bound boxes information.
        """
        objects = []
        for label in labels:
            center = [float(label[0]), float(label[1]), float(label[2])]
            size = [float(label[4]), float(label[5]), float(label[3])]
            objects.append(Object3D(center, size, label[6], cls=label_to_names[int(label[-2])], text=str(int(label[-1]))))

        return objects
    
    def get_split_list(self):
        """Returns the list of data splits available.

        Args:
            split: A string identifying the dataset split that is usually one of
            'training', 'test', 'validation', or 'all'.

        Returns:
            A dataset split object providing the requested subset of the data.

        Raises:
            ValueError: Indicates that the split name passed is incorrect. The
            split name should be one of 'training', 'test', 'validation', or
            'all'.
        """
        spilt_list = []
        for id in range(len(self.datas)):
            data_dict = {'data':self.datas[id], 'label':self.labels[id]}
            spilt_list.append(data_dict)
        return spilt_list
    
    def __len__(self):
        return len(self.datas)

    def get_split(self, split):
        """Returns a dataset split.

        Args:
            split: A string identifying the dataset split that is usually one of
            'training', 'test', 'validation', or 'all'.

        Returns:
            A dataset split object providing the requested subset of the data.
        """
        return DataSplit(self)

class DataSplit():
    def __init__(self, dataset):
        
        self.data_list = dataset.get_split_list()
        self.dataset = dataset

    def __len__(self):
        return len(self.data_list)

    def get_data(self, idx):
        data_dict = self.data_list[idx]

        pts = data_dict['data']
        label = self.dataset.read_label(data_dict['label'])

        data = {
            'point': pts,
            'feat': None,
            'bounding_boxes': label,
        }
        return data

    def get_attr(self, idx):

        attr = {'name': str(idx)}
        return attr

class Object3D(BoundingBox3D):
    def __init__(self, center, size, yaw, arrow=2.0, 
                    cls=0, socre=1.0, text=None, show_cls=True):
        self.yaw = yaw-np.pi*0.5
        left = [np.cos(self.yaw), np.sin(self.yaw), 0]
        front = [-np.sin(self.yaw), np.cos(self.yaw), 0]
        up = [0, 0, 1]

        super().__init__(center, 
                        front, 
                        up, 
                        left, 
                        size, 
                        cls, 
                        socre,
                        meta=text,
                        show_class=show_cls,
                        show_confidence=False,
                        show_meta=True,
                        identifier=None,
                        arrow_length=arrow)


def sequence_visualize3d(points=None, gt_info=None, detect_info=None, track_info=None):
    data_collect = DataCollect()
    data_collect.get_infos(points, gt_info, detect_info, track_info)

    Visualizer().visualize_dataset(data_collect, split='all')


def single_visualize3d(points_list, bboxes_list, points_names=None, bboxes_names=None):
    
    show_pts_list = []
    show_bboxs_list = []
    if points_names is not None:
        assert len(points_list) == len(points_names)
    if bboxes_names is not None:
        assert len(bboxes_list) == len(bboxes_names)

    for idx, points in enumerate(points_list):
        if points_names is not None:
            pts_name = points_names[idx]
        else: pts_name = 'pts:'+str(idx)
        show_pts_list.append({'name':pts_name, 
                              'points':points.astype(np.float32)})
    
    for idx, bboxs in enumerate(bboxes_list):
        per_frame_bbox = []
        for _, bbox in enumerate(bboxs):

            if bboxes_names is not None:
                bbox_name = bboxes_names[idx]
            else: bbox_name = 'bboxs:'+str(idx)
            center = [float(bbox[0]), float(bbox[1]), float(bbox[2])]
            size = [float(bbox[4]), float(bbox[5]), float(bbox[3])]
            show_bboxs_list.append(Object3D(center, size, bbox[6], cls=label_to_names[int(bbox[-2])], text=str(int(bbox[-1]))))
        # if bboxes_names is not None:
        #     bbox_name = bboxes_names[idx]
        # else: bbox_name = 'bboxs:'+str(idx)
        # center = [float(bbox[0]), float(bbox[1]), float(bbox[2])]
        # size = [float(bbox[4]), float(bbox[5]), float(bbox[3])]
        # show_bboxs_list.append(Object3D(center, size, bbox[6], cls=label_to_names[int(bbox[-2])], text=str(int(label[-1]))))
        # show_bboxs_list.append({'name':bbox_name, 
        #                         'points':bbox.astype(np.float32)})
    
    Visualizer().visualize(data=show_pts_list, bounding_boxes=show_bboxs_list)

# class Visualizer():
   
#     def __init__(self, mode='world'):
#         self.vis = ml3d.vis.Visualizer()
#         self.data = []
#         self.bboxs = []
#         self.world_frame_ceneter = None
#         self.mode = mode

#     def update(self, frame_id, points, infos, world_frame_center=None):
#         if self.world_frame_ceneter is None \
#            and self.mode == 'world' and world_frame_center is not None:
#            self.world_frame_ceneter = world_frame_center
        
#         if self.world_frame_ceneter is not None:
#             infos[:, :3] = infos[:, :3] - self.world_frame_ceneter
#             points = points - self.world_frame_ceneter
        
#         points = points.astype(np.float32)
#         self.data.append({'name':frame_id, 'points':points})
#         bboxs = []
#         for info in infos:
#             bboxs.append(_3Dbbox(info[:3], info[3:6], info[6], cls=0))
#         self.bboxs.append(bboxs)

#     def run(self):
        
#         self.vis.visualize(data = self.data, bounding_boxes=self.bboxs)

    