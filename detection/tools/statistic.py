import io
import os
import pickle

import numpy as np
from petrel_client.client import Client

from al3d_utils.ops.roiaware_pool3d import roiaware_pool3d_utils


def statistic_obj_num():
	client = Client('~/.petreloss.conf')

	info_path = 's3://dataset/waymo/waymo_infos_train.pkl'
	pkl_bytes = client.get(info_path)
	infos = pickle.load(io.BytesIO(pkl_bytes))

	name_num = {
		'Vehicle': 0,
		'Sign': 0,
		'Pedestrian': 0,
		'Cyclist': 0
	}

	frame_num = len(infos)	# 158081
	for info in infos:
		name_list = info['annos']['name']
		for name in name_list:
			if name in name_num: name_num[name] += 1
	
	print(name_num)

def calculate_pts_num():
	points = np.array(
		[1., 1., 1.],
		[1.1, 1.1, 1.1],
		[1.2, 1.2, 1.2],
		[1.3, 1.3, 1.3],
		[1.4, 1.4, 1.4],
		[1.5, 1.5, 1.5]
	)
	boxes = np.array(
		[1, 1, 1, 1, 1, 1, 0]
	)
	box_idxs = roiaware_pool3d_utils.points_in_boxes_gpu(
		torch.from_numpy(points[:, 0:3]).unsqueeze(dim=0).float().cuda(),
		torch.from_numpy(boxes[:, 0:7]).unsqueeze(dim=0).float().cuda()
	).long().squeeze(dim=0).cpu().numpy()

	print("points num: ", points.shape[0])
	# print("points in box num: ", box_idxs==0)


if __name__ == '__main__':
	statistic_obj_num()
