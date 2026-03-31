# lib/train/trainers/ltr_trainer.py

import os
import datetime
from collections import OrderedDict

from lib.train.data.wandb_logger import WandbWriter
from lib.train.trainers import BaseTrainer
from lib.train.admin import AverageMeter, StatValue
from lib.train.admin import TensorboardWriter
import torch
import time
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from lib.utils.misc import get_world_size


class LTRTrainer(BaseTrainer):
    def __init__(self, actor, loaders, optimizer, settings,
                 lr_scheduler=None, use_amp=False):
        super().__init__(actor, loaders, optimizer, settings, lr_scheduler)

        self._set_default_settings()
        self.stats = OrderedDict({loader.name: None for loader in self.loaders})

        #  Tensorboard / Wandb：仅 rank 0 初始化 
        self.tensorboard_writer = None
        self.wandb_writer       = None

        if settings.local_rank in [-1, 0]:
            tb_dir = os.path.join(
                self.settings.env.tensorboard_dir,
                self.settings.project_path
            )
            os.makedirs(tb_dir, exist_ok=True)
            self.tensorboard_writer = TensorboardWriter(
                tb_dir, [l.name for l in loaders]
            )

            if settings.use_wandb:
                world_size = get_world_size()
                cur_samples = (self.loaders[0].dataset.samples_per_epoch
                               * max(0, self.epoch - 1))
                interval = world_size * settings.batchsize
                self.wandb_writer = WandbWriter(
                    settings.project_path[6:], {},
                    tb_dir, cur_samples, interval
                )

        self.move_data_to_gpu = getattr(settings, 'move_data_to_gpu', True)
        self.settings = settings
        self.use_amp  = use_amp
        if use_amp:
            self.scaler = GradScaler()

    def _set_default_settings(self):
        defaults = {
            'print_interval': 10,
            'print_stats'   : None,
            'description'   : '',
        }
        for param, val in defaults.items():
            if getattr(self.settings, param, None) is None:
                setattr(self.settings, param, val)

    def cycle_dataset(self, loader):
        """执行一个 epoch 的训练或验证。"""
        self.actor.train(loader.training)
        torch.set_grad_enabled(loader.training)
        self._init_timing()
        for i, data in enumerate(loader, 1):
            self.data_read_done_time = time.time()
            if self.move_data_to_gpu:
                data = data.to(self.device)
            self.data_to_gpu_time = time.time()
            data['epoch']    = self.epoch
            data['settings'] = self.settings
            if not self.use_amp:
                loss, stats = self.actor(data)
            else:
                with autocast():
                    loss, stats = self.actor(data)
            if loader.training:
                self.optimizer.zero_grad()
                if not self.use_amp:
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.actor.net.parameters(),
                        self.settings.grad_clip_norm
                        if self.settings.grad_clip_norm > 0
                        else float('inf')
                    )
                    if not torch.isfinite(grad_norm):
                        # 梯度已 NaN/Inf，跳过本 step，清零梯度防止累积
                        self.optimizer.zero_grad()
                        if self.settings.local_rank in [-1, 0]:
                            print(f"[LTRTrainer] WARNING: grad_norm={grad_norm:.3f} "
                                f"at epoch={self.epoch} iter={i}, skipping optimizer step.")
                    else:
                        self.optimizer.step()
                else:
                    self.scaler.scale(loss).backward()
                    if self.settings.grad_clip_norm > 0:
                        self.scaler.unscale_(self.optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.actor.net.parameters(),
                            self.settings.grad_clip_norm
                        )
                        if not torch.isfinite(grad_norm):
                            self.optimizer.zero_grad()
                            if self.settings.local_rank in [-1, 0]:
                                print(f"[LTRTrainer] WARNING: grad_norm={grad_norm:.3f} "
                                    f"(AMP), skipping step.")
                            self.scaler.update()   # scaler 仍需 update 以维护内部状态
                            continue               # 跳过本 batch 的 stat 统计
        
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            batch_size = data['template_images'].shape[loader.stack_dim]
            self._update_stats(stats, batch_size, loader)
            self._print_stats(i, loader, batch_size)
            if (self.wandb_writer is not None
                    and i % self.settings.print_interval == 0
                    and self.settings.local_rank in [-1, 0]):
                self.wandb_writer.write_log(self.stats, self.epoch)
        # epoch 耗时：仅 rank 0 打印和写文件
        if self.settings.local_rank in [-1, 0]:
            epoch_time = self.prev_time - self.start_time
            time_str = (
                f"Epoch Time: {datetime.timedelta(seconds=epoch_time)}  |  "
                f"Avg DataTime: {self.avg_date_time/self.num_frames*batch_size:.5f}  |  "
                f"Avg GPUTransTime: "
                f"{self.avg_gpu_trans_time/self.num_frames*batch_size:.5f}  |  "
                f"Avg ForwardTime: {self.avg_forward_time/self.num_frames*batch_size:.5f}"
            )
            print(time_str)
            with open(self.settings.log_file, 'a') as f:
                f.write(time_str + '\n')

    def train_epoch(self):
        """每个 epoch 对所有 loader 执行一次 cycle。"""
        for loader in self.loaders:
            if self.epoch % loader.epoch_interval == 0:
                if isinstance(loader.sampler, DistributedSampler):
                    loader.sampler.set_epoch(self.epoch)
                self.cycle_dataset(loader)

        self._stats_new_epoch()

        # Tensorboard：仅 rank 0 写
        if self.settings.local_rank in [-1, 0]:
            self._write_tensorboard()

    def _init_timing(self):
        self.num_frames         = 0
        self.start_time         = time.time()
        self.prev_time          = self.start_time
        self.avg_date_time      = 0
        self.avg_gpu_trans_time = 0
        self.avg_forward_time   = 0

    def _update_stats(self, new_stats: OrderedDict, batch_size, loader):
        if (loader.name not in self.stats
                or self.stats[loader.name] is None):
            self.stats[loader.name] = OrderedDict(
                {name: AverageMeter() for name in new_stats.keys()}
            )

        if loader.training:
            for i, lr in enumerate(self.lr_scheduler.get_last_lr()):
                var_name = f'LearningRate/group{i}'
                if var_name not in self.stats[loader.name]:
                    self.stats[loader.name][var_name] = StatValue()
                self.stats[loader.name][var_name].update(lr)

        for name, val in new_stats.items():
            if name not in self.stats[loader.name]:
                self.stats[loader.name][name] = AverageMeter()
            self.stats[loader.name][name].update(val, batch_size)

    def _print_stats(self, i, loader, batch_size):
        """
        统计打印：仅 rank 0 打印到 stdout 和写文件。
        rank 1+ 完全静默，避免 nohup 重定向到同一文件时出现重复行。
        """
        self.num_frames  += batch_size
        current_time      = time.time()
        batch_fps         = batch_size / (current_time - self.prev_time)
        average_fps       = self.num_frames / (current_time - self.start_time)
        prev_backup       = self.prev_time
        self.prev_time    = current_time
        self.avg_date_time      += self.data_read_done_time - prev_backup
        self.avg_gpu_trans_time += self.data_to_gpu_time - self.data_read_done_time
        self.avg_forward_time   += current_time - self.data_to_gpu_time
        if not (i % self.settings.print_interval == 0 or i == len(loader)):
            return
        if self.settings.local_rank not in [-1, 0]:
            return   # rank 1+ 直接返回，不打印，不写文件
        print_str  = '[%s: %d, %d/%d] ' % (loader.name, self.epoch, i, len(loader))
        print_str += 'FPS: %.1f(%.1f)  ' % (average_fps, batch_fps)
        print_str += 'Data: %.3f  GPU: %.3f  Fwd: %.3f  ' % (
            self.avg_date_time      / self.num_frames * batch_size,
            self.avg_gpu_trans_time / self.num_frames * batch_size,
            self.avg_forward_time   / self.num_frames * batch_size,
        )
        for name, val in self.stats[loader.name].items():
            if (self.settings.print_stats is None
                    or name in self.settings.print_stats):
                if hasattr(val, 'avg'):
                    print_str += '%s: %.5f  ' % (name, val.avg)
        print(print_str)
        with open(self.settings.log_file, 'a') as f:
            f.write(print_str + '\n')

    def _stats_new_epoch(self):
        for loader in self.loaders:
            if loader.training:
                try:
                    lr_list = self.lr_scheduler.get_last_lr()
                except Exception:
                    lr_list = self.lr_scheduler._get_lr(self.epoch)
                for i, lr in enumerate(lr_list):
                    var_name = f'LearningRate/group{i}'
                    if var_name not in self.stats[loader.name]:
                        self.stats[loader.name][var_name] = StatValue()
                    self.stats[loader.name][var_name].update(lr)

        for loader_stats in self.stats.values():
            if loader_stats is None:
                continue
            for stat_value in loader_stats.values():
                if hasattr(stat_value, 'new_epoch'):
                    stat_value.new_epoch()

    def _write_tensorboard(self):
        """仅在 rank 0 调用，无需额外 guard。"""
        if self.epoch == 1:
            self.tensorboard_writer.write_info(
                self.settings.script_name,
                self.settings.description
            )
        self.tensorboard_writer.write_epoch(self.stats, self.epoch)