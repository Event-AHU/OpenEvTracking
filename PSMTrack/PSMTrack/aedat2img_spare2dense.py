import os 
import pdb 
import csv 
import numpy as np 
import cv2
import torch
import pandas as pd
from tqdm import tqdm
from PIL import Image
import math
from dv import AedatFile

data_path = '/DATA/Dataset/COESOT/train/'
save_path = '/DATA/Dataset/COESOT/train_sparse2dense/'

if __name__=='__main__':
    device = torch.device("cuda:0")
    use_mode = 'frame_exposure_time'
    os.makedirs(save_path, exist_ok=True)
    fileLIST = sorted(os.listdir(data_path))
    for seq_Name in tqdm(fileLIST):

        if seq_Name in {'train.txt', 'val.txt', 'test.txt', 'list.txt'}:
            continue

        print(seq_Name)

        video_save_path = os.path.join(save_path, seq_Name, seq_Name + '_dvs')

        if not os.path.exists(video_save_path):
            os.makedirs(video_save_path)
        else:
            continue
        
        read_path = os.path.join(data_path, seq_Name, seq_Name + '.aedat4')
        
        # read aeda4;
        frame_all = []
        frame_exposure_time = []
        frame_interval_time = []
        # match_file = "/wangx/DATA/Dataset/FE240/pair_all.txt"
        # pair = {}
        # with open(match_file, 'r') as f:
        #     for line in f.readlines():
        #         file, start_frame = line.split()
        #         pair[file] = int(start_frame) + 1
        # start_frame = pair[seq_Name]
        
        dvs_img_interval = 1
        # img_path = os.path.join(data_path, seq_Name, seq_Name + '_dvs')
        # frame_end = len(os.listdir(img_path))
        
        with AedatFile(read_path) as f:
            for frame in f['frames']:
                frame_all.append(frame.image)
                frame_exposure_time.append([frame.timestamp_start_of_exposure,
                                            frame.timestamp_end_of_exposure])  ## [1607928583397102, 1607928583401102]
                frame_interval_time.append([frame.timestamp_start_of_frame,
                                            frame.timestamp_end_of_frame])  ## [1607928583387944, 1607928583410285]


        frame_timestamp = frame_interval_time
        if  frame_timestamp[0][0] == 0:
            frame_timestamp = frame_exposure_time
            if frame_timestamp[0][0] == 0:
                print("Erro!")
                break
        frame_num = len(frame_timestamp)

        if seq_Name == 'dvSave-2022_03_21_16_11_40':
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
            
        # events = np.hstack([packet for packet in f['events'].numpy()])

        t_all = torch.tensor(events['timestamp']).unsqueeze(1).to(device)
        # x_all = torch.tensor(events['x']).unsqueeze(1).to(device)
        # y_all = torch.tensor(events['y']).unsqueeze(1).to(device)
        # p_all = torch.tensor(events['polarity']).unsqueeze(1).to(device)
            
        ################################################################################
        ###				save the Event Image
        ################################################################################
        # begin_frame = start_frame-1
        # end_frame = frame_end + start_frame-1
        height, width = f['events'].size
        for frame_no in range(0, int(frame_num / dvs_img_interval) - 1):
            mid_event_frame = 255 * np.ones((height, width, 3), dtype=np.uint8)
            sparse_event_frame = 255 * np.ones((height, width, 3), dtype=np.uint8)
            dense_event_frame = 255 * np.ones((height, width, 3), dtype=np.uint8)
            
            start_idx = np.searchsorted(events['timestamp'], frame_timestamp[frame_no][0])
            end_idx   = np.searchsorted(events['timestamp'], frame_timestamp[frame_no][1])
            mid_event = events[start_idx:end_idx]
            
            on_idx = np.where(mid_event['polarity'] == 1)  ## (array([    3,     4,     5, ..., 10633, 10635, 10636]),)
            off_idx = np.where(mid_event['polarity'] == 0)  ## (array([    0,     1,     2, ..., 10629, 10632, 10634]),)
            mid_event_frame[mid_event['y'][on_idx], mid_event['x'][on_idx], :] = [30, 30, 220] * mid_event['polarity'][on_idx][:, None]
            mid_event_frame[mid_event['y'][off_idx], mid_event['x'][off_idx], :] = [200, 30, 30] * (mid_event['polarity'][off_idx] + 1)[:, None]
            cv2.imwrite(os.path.join(video_save_path, 'frame{:04d}_2'.format((frame_no)) + '.png'), mid_event_frame)

            length = end_idx - start_idx
            mid_time_stamp = start_idx + length // 2
            
            #### 稀疏 ####
            spare_start_idx = max(0, mid_time_stamp - length // 4)
            spare_end_idx   = min(t_all.shape[0], mid_time_stamp + length // 4)
            sparse_event = events[spare_start_idx:spare_end_idx]

            on_idx = np.where(sparse_event['polarity'] == 1)
            off_idx = np.where(sparse_event['polarity'] == 0)
            sparse_event_frame[sparse_event['y'][on_idx], sparse_event['x'][on_idx], :] = [30, 30, 220] * sparse_event['polarity'][on_idx][:, None]
            sparse_event_frame[sparse_event['y'][off_idx], sparse_event['x'][off_idx], :] = [200, 30, 30] * (sparse_event['polarity'][off_idx] + 1)[:, None]
            cv2.imwrite(os.path.join(video_save_path, 'frame{:04d}_1'.format(frame_no)+'.png'), sparse_event_frame)

            #### 密集 ####
            dense_start_idx = max(0, mid_time_stamp - length)
            dense_end_idx   = min(t_all.shape[0], mid_time_stamp + length)
            dense_event = events[dense_start_idx:dense_end_idx]

            on_idx = np.where(dense_event['polarity'] == 1)
            off_idx = np.where(dense_event['polarity'] == 0)
            dense_event_frame[dense_event['y'][on_idx], dense_event['x'][on_idx], :] = [30, 30, 220] * dense_event['polarity'][on_idx][:, None]
            dense_event_frame[dense_event['y'][off_idx], dense_event['x'][off_idx], :] = [200, 30, 30] * (dense_event['polarity'][off_idx] + 1)[:, None]
            cv2.imwrite(os.path.join(video_save_path, 'frame{:04d}_3'.format(frame_no)+'.png'), dense_event_frame)