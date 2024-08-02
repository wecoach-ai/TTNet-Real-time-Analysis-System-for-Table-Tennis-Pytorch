import time
import sys
import os
import warnings

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
from tqdm import tqdm

sys.path.append('./')

from data_process.ttnet_dataloader import create_test_dataloader
from models.model_utils import create_model, load_pretrained_model, make_data_parallel, get_num_parameters
from utils.misc import AverageMeter
from config.config import parse_configs
from utils.post_processing import get_prediction_ball_pos, get_prediction_seg, prediction_get_events
from utils.metrics import SPCE, PCE


def main():
    configs = parse_configs()

    if configs.gpu_idx is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if configs.dist_url == "env://" and configs.world_size == -1:
        configs.world_size = int(os.environ["WORLD_SIZE"])

    configs.distributed = configs.world_size > 1 or configs.multiprocessing_distributed

    if configs.multiprocessing_distributed:
        configs.world_size = configs.ngpus_per_node * configs.world_size
        mp.spawn(main_worker, nprocs=configs.ngpus_per_node, args=(configs,))
    else:
        main_worker(configs.gpu_idx, configs)


def main_worker(gpu_idx, configs):
    configs.gpu_idx = gpu_idx

    if configs.gpu_idx is not None:
        print(f"Use GPU: {configs.gpu_idx} for training")
        configs.device = torch.device(f'cuda:{configs.gpu_idx}')

    if configs.distributed:
        if configs.dist_url == "env://" and configs.rank == -1:
            configs.rank = int(os.environ["RANK"])
        if configs.multiprocessing_distributed:
            configs.rank = configs.rank * configs.ngpus_per_node + gpu_idx

        dist.init_process_group(backend=configs.dist_backend, init_method=configs.dist_url,
                                world_size=configs.world_size, rank=configs.rank)

    configs.is_master_node = (not configs.distributed) or (
            configs.distributed and (configs.rank % configs.ngpus_per_node == 0))

    # model
    model = create_model(configs)
    model = make_data_parallel(model, configs)

    if configs.is_master_node:
        num_parameters = get_num_parameters(model)
        print(f'number of trained parameters of the model: {num_parameters}')

    if configs.pretrained_path is not None:
        model = load_pretrained_model(model, configs.pretrained_path, gpu_idx, configs.overwrite_global_2_local)
    # Load dataset
    test_loader = create_test_dataloader(configs)
    test(test_loader, model, configs)


def test(test_loader, model, configs):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    iou_seg = AverageMeter('IoU_Seg', ':6.4f')
    mse_global = AverageMeter('MSE_Global', ':6.4f')
    mse_local = AverageMeter('MSE_Local', ':6.4f')
    mse_overall = AverageMeter('MSE_Overall', ':6.4f')
    pce = AverageMeter('PCE', ':6.4f')
    spce = AverageMeter('Smooth_PCE', ':6.4f')
    w_original = 1920.
    h_original = 1080.
    w, h = configs.input_size

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        for batch_idx, (resized_imgs, org_ball_pos_xy, global_ball_pos_xy, target_events, target_seg) in enumerate(
            tqdm(test_loader)):

            print(f'\n===================== batch_idx: {batch_idx} ================================')

            data_time.update(time.time() - start_time)
            batch_size = resized_imgs.size(0)
            target_seg = target_seg.to(configs.device, non_blocking=True)
            resized_imgs = resized_imgs.to(configs.device, non_blocking=True).float()
            # compute output

            pred_ball_global, pred_ball_local, pred_events, pred_seg, local_ball_pos_xy, total_loss, _ = model(
                resized_imgs, org_ball_pos_xy, global_ball_pos_xy, target_events, target_seg)

            org_ball_pos_xy = org_ball_pos_xy.numpy()
            global_ball_pos_xy = global_ball_pos_xy.numpy()
            # Transfer output to cpu
            target_seg = target_seg.cpu().numpy()

            for sample_idx in range(batch_size):
                # Get target
                sample_org_ball_pos_xy = org_ball_pos_xy[sample_idx]
                sample_global_ball_pos_xy = global_ball_pos_xy[sample_idx]  # Target
                # Process the global stage
                sample_pred_ball_global = pred_ball_global[sample_idx]
                sample_prediction_ball_global_xy = get_prediction_ball_pos(sample_pred_ball_global, w,
                                                                           configs.thresh_ball_pos_mask)

                # Calculate the MSE
                if (sample_global_ball_pos_xy[0] > 0) and (sample_global_ball_pos_xy[1] > 0) and (
                        sample_prediction_ball_global_xy[0] > 0) and (sample_prediction_ball_global_xy[1] > 0):
                    mse = (sample_prediction_ball_global_xy[0] - sample_global_ball_pos_xy[0]) ** 2 + \
                          (sample_prediction_ball_global_xy[1] - sample_global_ball_pos_xy[1]) ** 2
                    mse_global.update(mse)

                print(f'\nBall Detection - \t Global stage: \t (x, y) - gt = ({sample_global_ball_pos_xy[0]}, {sample_global_ball_pos_xy[1]}), prediction = ({sample_prediction_ball_global_xy[0]}, {sample_prediction_ball_global_xy[1]})')

                sample_pred_org_x = sample_prediction_ball_global_xy[0] * (w_original / w)
                sample_pred_org_y = sample_prediction_ball_global_xy[1] * (h_original / h)

                # Process local ball stage
                if pred_ball_local is not None:
                    # Get target
                    local_ball_pos_xy = local_ball_pos_xy.cpu().numpy()  # Ground truth of the local stage
                    sample_local_ball_pos_xy = local_ball_pos_xy[sample_idx]  # Target
                    # Process the local stage
                    sample_pred_ball_local = pred_ball_local[sample_idx]
                    sample_prediction_ball_local_xy = get_prediction_ball_pos(sample_pred_ball_local, w,
                                                                              configs.thresh_ball_pos_mask)

                    # Calculate the MSE
                    if (sample_local_ball_pos_xy[0] > 0) and (sample_local_ball_pos_xy[1] > 0):
                        mse = (sample_prediction_ball_local_xy[0] - sample_local_ball_pos_xy[0]) ** 2 + (
                                sample_prediction_ball_local_xy[1] - sample_local_ball_pos_xy[1]) ** 2
                        mse_local.update(mse)
                        sample_pred_org_x += sample_prediction_ball_local_xy[0] - w / 2
                        sample_pred_org_y += sample_prediction_ball_local_xy[1] - h / 2

                    print(f'Ball Detection - \t Local stage: \t (x, y) - gt = ({sample_local_ball_pos_xy[0]}, {sample_local_ball_pos_xy[1]}), prediction = ({sample_prediction_ball_local_xy[0]}, {sample_prediction_ball_local_xy[1]})')

                print(f'Ball Detection - \t Overall: \t (x, y) - org: ({sample_org_ball_pos_xy[0]}, {sample_org_ball_pos_xy[1]}), prediction = ({int(sample_pred_org_x)}, {int(sample_pred_org_y)})')
                mse = (sample_org_ball_pos_xy[0] - sample_pred_org_x) ** 2 + (
                        sample_org_ball_pos_xy[1] - sample_pred_org_y) ** 2
                mse_overall.update(mse)

                # Process event stage
                if pred_events is not None:
                    sample_target_events = target_events[sample_idx].numpy()
                    sample_prediction_events = prediction_get_events(pred_events[sample_idx], configs.event_thresh)
                    print(
                        f'Event Spotting - \t gt = (is bounce: {sample_target_events[0]}, is net: {sample_target_events[1]}), prediction: (is bounce: {pred_events[sample_idx][0]:.4f}, is net: {pred_events[sample_idx][1]:.4f})')
                    # Compute metrics
                    spce.update(SPCE(sample_prediction_events, sample_target_events, thresh=0.5))
                    pce.update(PCE(sample_prediction_events, sample_target_events))

                # Process segmentation stage
                if pred_seg is not None:
                    sample_target_seg = target_seg[sample_idx].transpose(1, 2, 0).astype(np.int)
                    sample_prediction_seg = get_prediction_seg(pred_seg[sample_idx], configs.seg_thresh)

                    # Calculate the IoU
                    iou = 2 * np.sum(sample_target_seg * sample_prediction_seg) / (
                            np.sum(sample_target_seg) + np.sum(sample_prediction_seg) + 1e-9)
                    iou_seg.update(iou)

                    print(f'Segmentation - \t \t IoU = {iou:.4f}')

                    if configs.save_test_output:
                        fig, axes = plt.subplots(nrows=batch_size, ncols=2, figsize=(10, 5))
                        plt.tight_layout()
                        axes.ravel()
                        axes[2 * sample_idx].imshow(sample_target_seg * 255)
                        axes[2 * sample_idx + 1].imshow(sample_prediction_seg * 255)
                        # title
                        target_title = 'target seg'
                        pred_title = 'pred seg'
                        if pred_events is not None:
                            target_title += f', is bounce: {sample_target_events[0]}, is net: {sample_target_events[1]}'
                            pred_title += f', is bounce: {sample_prediction_events[0]}, is net: {sample_prediction_events[1]}'

                        axes[2 * sample_idx].set_title(target_title)
                        axes[2 * sample_idx + 1].set_title(pred_title)

                        plt.savefig(os.path.join(configs.saved_dir,
                                                 f'batch_idx_{batch_idx}_sample_idx_{sample_idx}.jpg'))

            if ((batch_idx + 1) % configs.print_freq) == 0:
                print(
                    f'batch_idx: {batch_idx} - Average iou_seg: {iou_seg.avg:.4f}, mse_global: {mse_global.avg:.1f}, mse_local: {mse_local.avg:.1f}, mse_overall: {mse_overall.avg:.1f}, pce: {pce.avg:.4f} spce: {spce.avg:.4f}')

            batch_time.update(time.time() - start_time)
            start_time = time.time()

    print(
        f'Average iou_seg: {iou_seg.avg:.4f}, mse_global: {mse_global.avg:.1f}, mse_local: {mse_local.avg:.1f}, mse_overall: {mse_overall.avg:.1f}, pce: {pce.avg:.4f} spce: {spce.avg:.4f}')
    print('Done testing')


if __name__ == '__main__':
    main()
