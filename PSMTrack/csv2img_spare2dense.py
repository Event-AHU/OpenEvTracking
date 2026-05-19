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

data_path = '/DATA/dataset/EventVOT_csv/train/'
save_path = '/DATA/dataset/EventVOT/train_sparse2dense/'

if __name__=='__main__':
	os.makedirs(save_path, exist_ok=True)
	fileLIST = sorted(os.listdir(data_path))
	for FileID in tqdm(range(len(fileLIST))):
		csv_Name = fileLIST[FileID]
		print(csv_Name)
		video_save_path = os.path.join(save_path, csv_Name.split('.')[0], 'img')
		if not os.path.exists(video_save_path):
			os.makedirs(video_save_path)
		else:
			continue  
        
		read_path = os.path.join(data_path, csv_Name)
		dt = pd.read_csv(read_path, dtype=np.int32, delimiter=",", usecols=(0, 1, 2, 3) )
		dt = np.array(dt)
		dt = torch.tensor(dt, dtype=torch.int)

		x, y, p, t = torch.chunk(dt, 4, dim=1)
		all_events = torch.cat((x, y, p, t), dim=1)

		all_events = all_events.numpy()
  
		height,width = 720,1280
		time_length = all_events[-1,3] - all_events[0,3]
		# need_frame_num = math.ceil((time_length / 1e6) * frameRATE)
		start_idx = []
		deltaT = int(time_length / 500)
		i = 1
		for j in range(len(all_events)):
			if all_events[j][-1]-all_events[0][-1] > deltaT * i:
				start_idx.append(j)
				i += 1

		################################################################################
		###				save the Event Image
		################################################################################
		count_IMG = 0
		assert len(start_idx)!=0,'{} get 0 img!'.format(csv_Name)
		for imgID in range(len(start_idx)-1):
			event_frame_s = 255 * np.ones((height, width, 3), dtype=np.uint8)
			event_frame_m = 255 * np.ones((height, width, 3), dtype=np.uint8)
			event_frame_d = 255 * np.ones((height, width, 3), dtype=np.uint8)

			start_time_stamp = start_idx[imgID]
			end_time_stamp = start_idx[imgID+1]

			# event = all_events[start_time_stamp:end_time_stamp]

			event_mid = all_events[start_time_stamp:end_time_stamp]

			on_idx = np.where(event_mid[:, 2] == 1)
			off_idx = np.where(event_mid[:, 2] == 0)
			event_frame_m[height - 1 - event_mid[:, 1][on_idx],  event_mid[:, 0][on_idx], :] = [30, 30, 220] * event_mid[:, 2][on_idx][:, None]
			event_frame_m[height - 1 - event_mid[:, 1][off_idx], event_mid[:, 0][off_idx], :] = [200, 30, 30] * (event_mid[:, 2][off_idx]+1)[:, None]
			event_frame_m=cv2.flip(event_frame_m,0)  ##垂直翻转
			cv2.imwrite(os.path.join(video_save_path, '{:04d}_2'.format(count_IMG)+'.png'), event_frame_m)


			length = end_time_stamp - start_time_stamp
			mid_time_stamp = start_time_stamp + length // 2
			
			#### 稀疏 ####
			start_time_stamp_s = max(start_idx[0], mid_time_stamp - length // 4)
			end_time_stamp_s   = min(start_idx[-1], mid_time_stamp + length // 4)
			sparse_event = all_events[start_time_stamp_s:end_time_stamp_s]

			on_idx = np.where(sparse_event[:, 2] == 1)
			off_idx = np.where(sparse_event[:, 2] == 0)
			event_frame_s[height - 1 - sparse_event[:, 1][on_idx],  sparse_event[:, 0][on_idx], :] = [30, 30, 220] * sparse_event[:, 2][on_idx][:, None]
			event_frame_s[height - 1 - sparse_event[:, 1][off_idx], sparse_event[:, 0][off_idx], :] = [200, 30, 30] * (sparse_event[:, 2][off_idx]+1)[:, None]
			event_frame_s=cv2.flip(event_frame_s,0)
			cv2.imwrite(os.path.join(video_save_path, '{:04d}_1'.format(count_IMG)+'.png'), event_frame_s)

			#### 密集 ####
			start_time_stamp_d = max(start_idx[0], mid_time_stamp - length)
			end_time_stamp_d   = min(start_idx[-1], mid_time_stamp + length)
			dense_event = all_events[start_time_stamp_d:end_time_stamp_d]

			on_idx = np.where(dense_event[:, 2] == 1)
			off_idx = np.where(dense_event[:, 2] == 0)
			event_frame_d[height - 1 - dense_event[:, 1][on_idx],  dense_event[:, 0][on_idx], :] = [30, 30, 220] * dense_event[:, 2][on_idx][:, None]
			event_frame_d[height - 1 - dense_event[:, 1][off_idx], dense_event[:, 0][off_idx], :] = [200, 30, 30] * (dense_event[:, 2][off_idx]+1)[:, None]
			event_frame_d=cv2.flip(event_frame_d,0)
			cv2.imwrite(os.path.join(video_save_path, '{:04d}_3'.format(count_IMG)+'.png'), event_frame_d)
			
			count_IMG += 1