import io
import os
import pickle
import argparse

import numpy as np
from petrel_client.client import Client


def convert_det_to_track_worker(det_res, gt_infos):
    track_res = {}
    for i, res in enumerate(det_res):
        seq_name = res['sequence_name'][8:-19]
        frame_id = res['frame_id']
        
        if seq_name not in track_res:
            track_res[seq_name] = {}
        if str(frame_id) not in track_res[seq_name]:
            track_res[seq_name][str(frame_id)] = {}
        
        track_res[seq_name][str(frame_id)]['pose'] = res['pose']
        track_res[seq_name][str(frame_id)]['boxes_lidar'] = res['boxes_lidar']
        track_res[seq_name][str(frame_id)]['name'] = res['name']
        track_res[seq_name][str(frame_id)]['score'] = res['score']
        track_res[seq_name][str(frame_id)]['sequence_name'] = seq_name
        track_res[seq_name][str(frame_id)]['frame_id'] = frame_id
        track_res[seq_name][str(frame_id)]['timestamp'] = gt_infos[i]['time_stamp']
    return track_res

def conver_det_to_track(res_path, gt_path, save_path):

    print("Start to convert detection result format.")
    # load the detection result, pickle file
    # det_res: list, the length is 39987 for val set
    client = Client('~/.petreloss.conf')
    if 's3' in res_path:
        det_res = pickle.load(io.BytesIO(client.get(res_path)))
    else:
        with open(res_path, 'rb') as f:
            det_res = pickle.load(f)

    # load gt_infos to obtain the timestamp information
    gt_infos = pickle.load(io.BytesIO(client.get(gt_path)))

    track_res = convert_det_to_track_worker(det_res, gt_infos)

    # save the track_res as pickle file for running track
    track_file_name = 'track_' + res_path.split('/')[-1]
    save_path = os.path.join(save_path, track_file_name)
    if 's3' in save_path:
        with io.BytesIO() as f:
            pickle.dump(track_res, f)
            client.put(save_path, f.getvalue())
    else:   
        with open(save_path, 'wb') as f:
            pickle.dump(track_res, f)

    print("Detection Result format convert finished.")


def convert_res_to_eval(res_path, save_path):

    print("Start to convert prediction result format for running al3d_evaluator.")

    if 's3' in res_path:
        client = Client('~/.petreloss.conf')
        res = pickle.load(io.BytesIO(client.get(res_path)))
    else:
        with open(res_path, 'rb') as f:
            res = pickle.load(f)

    eval_res = []
    for seq_name in res.keys():
        for frame_id in res[seq_name].keys():
            eval_res.append(res[seq_name][frame_id])
    
    # save the track_res as pickle file for running track
    eval_file_name = 'eval_' + res_path.split('/')[-1]
    save_path = os.path.join(save_path, eval_file_name)
    if 's3' in save_path:
        with io.BytesIO() as f:
            pickle.dump(eval_res, f)
            client.put(save_path, f.getvalue())
    else:   
        with open(save_path, 'wb') as f:
            pickle.dump(eval_res, f)

    print("Convert prediction result for eval finished.")


def main():
    parser = argparse.ArgumentParser(description='Result Converter.')
    parser.add_argument('--res_path', type=str, default='/mnt/lustre/share_data/PJLab-ADG/waymo/Models/detection/best/result.pkl', help='Path to the prediction result pkl file.')
    parser.add_argument('--gt_path', type=str, default='s3://dataset/waymo/gt_infos/gt.val.pkl', help='Path to ground-truth info pkl file.')
    parser.add_argument('--save_path', type=str, default='./', help='Path to save pickle file for running track module.')
    parser.add_argument('--func', type=str, default='convert_det', help='Please choose convert_det / convert_for_eval')
    args = parser.parse_args()

    if args.func == 'convert_det':
        conver_det_to_track(args.res_path, args.gt_path, args.save_path)
    elif args.fuc == 'convert_for_eval':
        convert_res_to_eval(args.res_path, args.save_path)


if __name__ == '__main__':
    main()
