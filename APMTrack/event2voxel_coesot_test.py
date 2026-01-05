import os
import pdb
# import csv
import numpy as np
import cv2
import torch
import pandas as pd
from tqdm import tqdm
import scipy.io
# from spconv.pytorch.utils import PointToVoxel
from dv import AedatFile
import numpy as np
import random


def events_reader_aug(sub_event, h, w, hs, ws, temporal_flip, voxel_channels, device):
    t = torch.tensor(sub_event['timestamp'],  device=device)
    x = torch.tensor(sub_event['x'],  device=device)
    y = torch.tensor(sub_event['y'],  device=device)
    p = torch.tensor(sub_event['polarity'],  device=device)

    if len(t) == 0:
        return torch.zeros((voxel_channels, h, w), dtype=torch.float32, device=device)

    if temporal_flip:
        x = torch.flip(x, dims=[0])
        y = torch.flip(y, dims=[0])
        p = 1.0 - torch.flip(p, dims=[0])

    # 时间归一化
    t0 = t[0]
    t_step = (t[-1] - t0 + 1) / voxel_channels
    t_bin = ((t - t0) / t_step).long().clamp(0, voxel_channels - 1)

    # 双线性插值准备
    x0 = torch.floor(x).long()
    y0 = torch.floor(y).long()
    x1 = x0 + 1
    y1 = y0 + 1

    # 权重
    wx = x - x0.float()
    wy = y - y0.float()
    w00 = (1 - wx) * (1 - wy)
    w01 = (1 - wx) * wy
    w10 = wx * (1 - wy)
    w11 = wx * wy

    # 极性值转为 -1 或 +1
    pv = torch.abs(p * 2 - 1)
    # pv = p * 2 - 1

    voxel = torch.zeros((voxel_channels, h, w), dtype=torch.float32, device=device)

    def accumulate(t_idx, y_idx, x_idx, weight):
        mask = (x_idx >= 0) & (x_idx < w) & (y_idx >= 0) & (y_idx < h)
        voxel.index_put_((t_idx[mask], y_idx[mask], x_idx[mask]), weight[mask] * pv[mask], accumulate=True)

    accumulate(t_bin, y0, x0, w00)
    accumulate(t_bin, y1, x0, w01)
    accumulate(t_bin, y0, x1, w10)
    accumulate(t_bin, y1, x1, w11)

    return voxel

if __name__ == '__main__':
    use_mode = 'frame_exposure_time'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    training_subset = False # False
    data_path = '/wangx/DATA/Dataset/COESOT/test/'
    save_path = '/wangx/DATA/Dataset/COESOT/new_voxel2/test/'

    video_files = sorted(os.listdir(data_path))
    dvs_img_interval = 1

    for videoID in tqdm(range(len(video_files))):
        foldName = video_files[videoID]
        print("==>> foldName: ", foldName)

        mat_save = os.path.join(save_path, foldName, foldName+'_5binvoxel/')
        if not os.path.exists(mat_save):
            os.makedirs(mat_save)
        else:
            continue
        
        aedat4_file = foldName + '.aedat4'
        read_path = os.path.join(data_path, foldName, aedat4_file)
        
        # read aeda4;
        frame_all = []
        frame_exposure_time = []
        frame_interval_time = []
        with AedatFile(read_path) as f:
            # print(f.names)
            for frame in f['frames']:
                frame_all.append(frame.image)
                frame_exposure_time.append([frame.timestamp_start_of_exposure,
                                            frame.timestamp_end_of_exposure])  ## [1607928583397102, 1607928583401102]
                frame_interval_time.append([frame.timestamp_start_of_frame,
                                            frame.timestamp_end_of_frame])  ## [1607928583387944, 1607928583410285]
        if use_mode == 'frame_exposure_time':
            frame_timestamp = frame_exposure_time
        elif use_mode == 'frame_interval_time':
            frame_timestamp = frame_interval_time
        frame_num = len(frame_timestamp)

        if foldName == 'dvSave-2022_03_21_16_11_40':
            event_list = []
            try:
                for event in f['events'].numpy():
                    event_list.append(event)
            except RuntimeError as e:
                # error_flag = True
                print(f"Error reading events: {e}")
                events_back = np.hstack([packet for packet in f['events'].numpy()])
            events = np.hstack(event_list)
            events = np.hstack((events, events_back))
        else:
            events = np.hstack([packet for packet in f['events'].numpy()])

        # t_all = torch.tensor(events['timestamp']).unsqueeze(1).to(device)
        # x_all = torch.tensor(events['x']).unsqueeze(1).to(device)
        # y_all = torch.tensor(events['y']).unsqueeze(1).to(device)
        # p_all = torch.tensor(events['polarity']).unsqueeze(1).to(device)
    
        
        H, W, _ = frame.image.shape
        temporal_flip = (random.random() > 0.5) & training_subset
        for frame_no in range(0, int(frame_num / dvs_img_interval) - 1):
            start_idx = np.where(events['timestamp'] >= frame_timestamp[frame_no][0])[0][0]
            end_idx = np.where(events['timestamp'] >= frame_timestamp[frame_no][1])[0][0]
            sub_event = events[start_idx:end_idx]
            cur_voxel = events_reader_aug(sub_event, H, W, 0, 0, temporal_flip, voxel_channels=5, device=device).detach().cpu().numpy()
            
            # # scipy.io.savemat(mat_save + 'frame{:0>4d}.mat'.format(frame_no), mdict={'features': cur_voxel})

            # event_frame = 255 * np.ones((H, W, 3), dtype=np.uint8)
        
            # on_idx = np.where(sub_event['polarity'] == 1)  ## (array([    3,     4,     5, ..., 10633, 10635, 10636]),)
            # off_idx = np.where(sub_event['polarity'] == 0)  ## (array([    0,     1,     2, ..., 10629, 10632, 10634]),)
            # event_frame[sub_event['y'][on_idx], sub_event['x'][on_idx], :] = [30, 30, 220] * sub_event['polarity'][on_idx][:, None]
            # event_frame[sub_event['y'][off_idx], sub_event['x'][off_idx], :] = [200, 30, 30] * (sub_event['polarity'][off_idx] + 1)[
            #                                                                            :, None]
            # cv2.imwrite(os.path.join(mat_save, 'frame{:04d}'.format((frame_no) * dvs_img_interval) + '.png'), event_frame)
            
            np.savez_compressed(mat_save + f'frame{(frame_no):04d}.npz', features=cur_voxel)