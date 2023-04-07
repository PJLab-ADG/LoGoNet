import pickle
import time

import cv2
import numpy as np
import torch
import tqdm
import mayavi.mlab as mlab
from tensorboardX import SummaryWriter
from imageio import imwrite

from al3d_det.models import load_data_to_gpu
from . import visualize_utils as V


def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' %
               str(cur_thresh)] += ret_dict.get('roi_%s' %
                                                str(cur_thresh), 0)
        metric['recall_rcnn_%s' %
               str(cur_thresh)] += ret_dict.get('rcnn_%s' %
                                                str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = '(%d, %d) / %d' % \
        (metric['recall_roi_%s' % str(min_thresh)],
         metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])


def eval_one_epoch(
        cfg,
        model,
        dataloader,
        epoch_id,
        logger,
        dist_test=False,
        save_to_file=False,
        result_dir=None,
        save_tb=True):
    '''
    Arguments:
        save_tb: tensorboard visualization
    '''
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    model.eval()

    # The selected samples for visualization
    select_id_list = {'0', '1', '2', '3'}

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(
            total=len(dataloader),
            leave=True,
            desc='eval',
            dynamic_ncols=True)
    start_time = time.time()
    for i, batch_dict in enumerate(dataloader):
        batch_dict['eval_iter'] = i
        load_data_to_gpu(batch_dict)
        with torch.no_grad():
            pred_dicts, ret_dict = model(batch_dict)
        disp_dict = {}

        print(batch_dict['frame_id'][0])

        # visualize the 3D results
        print(batch_dict['gt_boxes'][0][:2])
        V.draw_scenes(
            points=batch_dict['points'][:, 1:], ref_boxes=batch_dict['gt_boxes'][0][:2],
            ref_scores=None, ref_labels=None
        )
        mlab.show(stop=True)
        import pdb; pdb.set_trace()

        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    logger.info('****************Visualization done.*****************')
    return ret_dict


if __name__ == '__main__':
    pass
