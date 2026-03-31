# lib/train/trainers/base_trainer.py
import os
import glob
import torch
import traceback
from lib.train.admin import multigpu
from torch.utils.data.distributed import DistributedSampler


# ─────────────────────────────────────────────
# Warmup 包装类：对任意 base_scheduler 叠加线性 warmup
# ─────────────────────────────────────────────
class WarmupSchedulerWrapper:
    """
    在 base_scheduler 之前插入线性 warmup 阶段。

    行为定义：
      epoch ∈ [1, warmup_epoch]    : LR 从 warmup_lr_init 线性升至 base_lr
      epoch ∈ [warmup_epoch+1, ∞)  : 完全由 base_scheduler 控制

    注意：这里的 epoch 编号与 BaseTrainer.epoch 一致，从 1 开始计。
    """

    def __init__(self, base_scheduler, optimizer,
                 warmup_epoch: int, warmup_lr_init: float):
        """
        args:
            base_scheduler  : 原始 scheduler（StepLR / CosineAnnealingLR 等）
            optimizer       : 与 base_scheduler 共享的 optimizer
            warmup_epoch    : warmup 持续的 epoch 数（绝对 epoch 数）
            warmup_lr_init  : warmup 第 1 epoch 开始时的 LR（绝对值）
        """
        assert warmup_epoch >= 0, "warmup_epoch must be non-negative"
        self.base_scheduler  = base_scheduler
        self.optimizer       = optimizer
        self.warmup_epoch    = warmup_epoch
        self.warmup_lr_init  = warmup_lr_init
        self._warmup_finished = (warmup_epoch == 0)

        # 缓存各 param_group 的 base_lr（warmup 目标 LR）
        # 在 __init__ 时 optimizer.param_groups 的 lr 即为 yaml 中设置的初始 LR
        self._base_lrs = [
            group['lr'] for group in optimizer.param_groups
        ]
        # last_epoch 语义：当前已完成的 epoch 编号（与 BaseTrainer.epoch 同步）
        self.last_epoch = 0

    # ── 对外接口：与标准 scheduler 完全兼容 ──────────────────────────────

    def step(self, epoch=None):
        """
        由 BaseTrainer.train() 在每个 epoch 结束后调用。
        epoch 参数含义：刚结束的 epoch 编号（从 1 开始）。
        """
        if epoch is not None:
            self.last_epoch = epoch

        if self.last_epoch <= self.warmup_epoch:
            # warmup 阶段：手动设置各 param_group 的 lr
            self._set_warmup_lr(self.last_epoch)
        else:
            # warmup 结束后：第一次进入时先让 base_scheduler 的
            # last_epoch 对齐（去掉 warmup offset），然后正常 step
            if not self._warmup_finished:
                self._align_base_scheduler()
                self._warmup_finished = True
            self.base_scheduler.step()

    def get_last_lr(self):
        """返回当前各 param_group 的实际 LR，格式与标准 scheduler 一致。"""
        return [group['lr'] for group in self.optimizer.param_groups]

    def state_dict(self):
        return {
            'last_epoch'       : self.last_epoch,
            'warmup_epoch'     : self.warmup_epoch,
            'warmup_lr_init'   : self.warmup_lr_init,
            '_base_lrs'        : self._base_lrs,
            '_warmup_finished' : self._warmup_finished,
            'base_scheduler'   : self.base_scheduler.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.last_epoch       = state_dict['last_epoch']
        self.warmup_epoch     = state_dict['warmup_epoch']
        self.warmup_lr_init   = state_dict['warmup_lr_init']
        self._base_lrs        = state_dict['_base_lrs']
        self._warmup_finished = state_dict['_warmup_finished']
        self.base_scheduler.load_state_dict(state_dict['base_scheduler'])

    # ── 内部方法 ─────────────────────────────────────────────────────────

    def _set_warmup_lr(self, current_epoch):
        """线性插值计算 warmup LR 并写入 optimizer。"""
        # current_epoch=1 → lr=warmup_lr_init
        # current_epoch=warmup_epoch → lr=base_lr（到达目标值）
        if self.warmup_epoch == 0:
            return
        alpha = current_epoch / self.warmup_epoch   # [1/W, 1.0]
        for group, base_lr in zip(self.optimizer.param_groups, self._base_lrs):
            group['lr'] = self.warmup_lr_init + alpha * (base_lr - self.warmup_lr_init)

    def _align_base_scheduler(self):
        """
        warmup 结束后，将 base_scheduler 的 last_epoch 设置为
        (实际 epoch - warmup_epoch)，使其从正确位置继续 decay。

        例：total_epoch=20, warmup=2, LR_DROP_EPOCH=12
            warmup 结束时 last_epoch=3（即第3个epoch开始执行base_scheduler）
            base_scheduler 应该认为自己刚走完第 3-2=1 个 epoch
        """
        offset = self.last_epoch - self.warmup_epoch
        self.base_scheduler.last_epoch = max(0, offset - 1)
        # 强制 base_scheduler 以当前 last_epoch 为准重新计算 lr
        self.base_scheduler.step()


class BaseTrainer:
    """Base trainer class."""

    def __init__(self, actor, loaders, optimizer, settings, lr_scheduler=None):
        self.actor         = actor
        self.optimizer     = optimizer
        self.lr_scheduler  = lr_scheduler
        self.loaders       = loaders

        self.update_settings(settings)

        self.epoch  = 0
        self.stats  = {}
        self.device = getattr(settings, 'device', None)
        if self.device is None:
            self.device = torch.device(
                "cuda:0" if torch.cuda.is_available() and settings.use_gpu else "cpu"
            )
        self.actor.to(self.device)
        self.settings = settings

    def update_settings(self, settings=None):
        if settings is not None:
            self.settings = settings
        if self.settings.env.workspace_dir is not None:
            self.settings.env.workspace_dir = os.path.expanduser(
                self.settings.env.workspace_dir
            )
            if self.settings.save_dir is None:
                self._checkpoint_dir = os.path.join(
                    self.settings.env.workspace_dir, 'checkpoints'
                )
            else:
                self._checkpoint_dir = os.path.join(
                    self.settings.save_dir, 'checkpoints'
                )
            if self.settings.local_rank in [-1, 0]:
                os.makedirs(self._checkpoint_dir, exist_ok=True)
                print("checkpoints will be saved to %s" % self._checkpoint_dir)
        else:
            self._checkpoint_dir = None

    # ════════════════════════════════════════════════════════════════════
    #  train()  ← 核心修改点
    # ════════════════════════════════════════════════════════════════════
    def train(self, max_epochs, load_latest=False, fail_safe=True,
              load_previous_ckpt=False, distill=False, resume=None):
        epoch = -1
        num_tries = 1
        for i in range(num_tries):
            try:
                if resume is not None:
                    if self.settings.local_rank in [-1, 0]:
                        print(f'[BaseTrainer] Resuming from: {resume}')
                    self.load_checkpoint(checkpoint=resume)
                elif load_latest:
                    self.load_checkpoint()
                if load_previous_ckpt:
                    directory = '{}/{}'.format(
                        self._checkpoint_dir,
                        self.settings.project_path_prv
                    )
                    self.load_state_dict(directory)
                if distill:
                    directory_teacher = '{}/{}'.format(
                        self._checkpoint_dir,
                        self.settings.project_path_teacher
                    )
                    self.load_state_dict(directory_teacher, distill=True)

                for epoch in range(self.epoch + 1, max_epochs + 1):
                    self.epoch = epoch
                    self.train_epoch()

                    # ── LR scheduler step ────────────────────────────────
                    if self.lr_scheduler is not None:
                        if isinstance(self.lr_scheduler, WarmupSchedulerWrapper):
                            # WarmupSchedulerWrapper 统一用 epoch 参数驱动
                            self.lr_scheduler.step(epoch=epoch)
                        elif self.settings.scheduler_type != 'cosine':
                            self.lr_scheduler.step()
                        else:
                            self.lr_scheduler.step(epoch - 1)
                    # ─────────────────────────────────────────────────────

                    should_save = (
                        epoch > (max_epochs - 1)
                        or getattr(self.settings, "save_every_epoch", False)
                        or epoch % 1 == 0
                        or epoch > (max_epochs - 3)
                    )
                    if should_save and self._checkpoint_dir:
                        if self.settings.local_rank != -1:
                            import torch.distributed as dist
                            dist.barrier()
                        if self.settings.local_rank in [-1, 0]:
                            self.save_checkpoint()

            except Exception:
                print('Training crashed at epoch {}'.format(epoch))
                if fail_safe:
                    self.epoch -= 1
                    load_latest = True
                    resume = None
                    print(traceback.format_exc())
                    print('Restarting training from last epoch ...')
                else:
                    raise
        print('Finished training!')

    def train_epoch(self):
        raise NotImplementedError

    # ════════════════════════════════════════════════════════════════════
    #  save_checkpoint()  ← 新增 lr_scheduler 序列化
    # ════════════════════════════════════════════════════════════════════
    def save_checkpoint(self):
        net = (self.actor.net.module
               if multigpu.is_multi_gpu(self.actor.net)
               else self.actor.net)
        actor_type = type(self.actor).__name__
        net_type   = type(net).__name__

        # 序列化 lr_scheduler：WarmupSchedulerWrapper 用自定义 state_dict
        if self.lr_scheduler is not None:
            lr_sched_state = self.lr_scheduler.state_dict()
        else:
            lr_sched_state = None

        state = {
            'epoch'        : self.epoch,
            'actor_type'   : actor_type,
            'net_type'     : net_type,
            'net'          : net.state_dict(),
            'net_info'     : getattr(net, 'info', None),
            'constructor'  : getattr(net, 'constructor', None),
            'optimizer'    : self.optimizer.state_dict(),
            'lr_scheduler' : lr_sched_state,          # ← 新增
            'stats'        : self.stats,
            'settings'     : self.settings,
        }
        directory = '{}/{}'.format(
            self._checkpoint_dir, self.settings.project_path
        )
        os.makedirs(directory, exist_ok=True)
        tmp_path  = '{}/{}_ep{:04d}.tmp'.format(directory, net_type, self.epoch)
        file_path = '{}/{}_ep{:04d}.pth.tar'.format(directory, net_type, self.epoch)
        torch.save(state, tmp_path)
        os.rename(tmp_path, file_path)
        print(f'[rank 0] Saved checkpoint: {file_path}')

    # ════════════════════════════════════════════════════════════════════
    #  load_checkpoint()  ← 修改 LR 恢复逻辑
    # ════════════════════════════════════════════════════════════════════
    def load_checkpoint(self, checkpoint=None, fields=None,
                        ignore_fields=None, load_constructor=False):
        net = (self.actor.net.module
               if multigpu.is_multi_gpu(self.actor.net)
               else self.actor.net)
        actor_type = type(self.actor).__name__
        net_type   = type(net).__name__

        # ── checkpoint 路径解析（原逻辑不变）────────────────────────────
        if checkpoint is None:
            checkpoint_list = sorted(glob.glob(
                '{}/{}/{}_ep*.pth.tar'.format(
                    self._checkpoint_dir,
                    self.settings.project_path,
                    net_type
                )
            ))
            if checkpoint_list:
                checkpoint_path = checkpoint_list[-1]
            else:
                print('No matching checkpoint file found')
                return
        elif isinstance(checkpoint, int):
            checkpoint_path = '{}/{}/{}_ep{:04d}.pth.tar'.format(
                self._checkpoint_dir, self.settings.project_path,
                net_type, checkpoint
            )
        elif isinstance(checkpoint, str):
            if os.path.isdir(checkpoint):
                checkpoint_list = sorted(glob.glob(
                    '{}/*_ep*.pth.tar'.format(checkpoint)
                ))
                if checkpoint_list:
                    checkpoint_path = checkpoint_list[-1]
                else:
                    raise Exception(f'No checkpoint found in dir: {checkpoint}')
            else:
                checkpoint_path = os.path.expanduser(checkpoint)
        else:
            raise TypeError('checkpoint must be None, int, or str')

        checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')

        ckpt_net_type = checkpoint_dict.get('net_type', 'unknown')
        if net_type != ckpt_net_type:
            print(f'[load_checkpoint] WARNING: net_type mismatch: '
                  f'current={net_type}, checkpoint={ckpt_net_type}. '
                  f'Loading with strict=False.')

        if fields is None:
            fields = checkpoint_dict.keys()
        if ignore_fields is None:
            ignore_fields = ['settings']
        ignore_fields = list(ignore_fields)
        # lr_scheduler 由下方单独处理，从通用字段加载中排除
        ignore_fields.extend([
            'lr_scheduler', 'constructor', 'net_type',
            'actor_type', 'net_info'
        ])

        for key in fields:
            if key in ignore_fields:
                continue
            if key == 'net':
                missing, unexpected = net.load_state_dict(
                    checkpoint_dict[key], strict=False
                )
                if self.settings.local_rank in [-1, 0]:
                    print(f'[load_checkpoint] net loaded from {checkpoint_path}')
                    print(f'  missing={len(missing)}, unexpected={len(unexpected)}')
                    if missing:
                        print(f'  missing keys (first 5): {missing[:5]}')
            elif key == 'optimizer':
                try:
                    self.optimizer.load_state_dict(checkpoint_dict[key])
                except Exception as e:
                    print(f'[load_checkpoint] WARNING: optimizer load failed: {e}')
                    print('[load_checkpoint] Skipping optimizer state.')
            else:
                setattr(self, key, checkpoint_dict[key])

        if load_constructor and 'constructor' in checkpoint_dict \
                and checkpoint_dict['constructor'] is not None:
            net.constructor = checkpoint_dict['constructor']
        if 'net_info' in checkpoint_dict \
                and checkpoint_dict['net_info'] is not None:
            net.info = checkpoint_dict['net_info']

        # ── LR scheduler 恢复：区分 Warmup / 普通两种情况 ───────────────
        if 'epoch' in fields and self.lr_scheduler is not None:
            self._restore_lr_scheduler(checkpoint_dict)

        for loader in self.loaders:
            if isinstance(loader.sampler, DistributedSampler):
                loader.sampler.set_epoch(self.epoch)

        if self.settings.local_rank in [-1, 0]:
            print(f'[load_checkpoint] Resumed from epoch {self.epoch}')
        return True

    def _restore_lr_scheduler(self, checkpoint_dict):
        """
        从 checkpoint 恢复 lr_scheduler 状态。
        支持三种场景：
          A. 新开训练（checkpoint 无 lr_scheduler 字段）→ 从 epoch 位置对齐
          B. 断点续训同配置（checkpoint 有 lr_scheduler 字段）→ 直接 load_state_dict
          C. 跨配置续训（从旧 SOR checkpoint 初始化 STOR）→ 丢弃旧 scheduler，重新对齐
        """
        ckpt_sched_state = checkpoint_dict.get('lr_scheduler', None)
        resumed_epoch    = self.epoch   # 已被 setattr 更新

        if isinstance(self.lr_scheduler, WarmupSchedulerWrapper):
            warmup_ep = self.lr_scheduler.warmup_epoch

            if ckpt_sched_state is not None and 'warmup_epoch' in ckpt_sched_state:
                # 场景 B：checkpoint 也是 WarmupSchedulerWrapper，直接恢复
                try:
                    self.lr_scheduler.load_state_dict(ckpt_sched_state)
                    if self.settings.local_rank in [-1, 0]:
                        print(f'[LR] Warmup scheduler state restored from checkpoint.')
                except Exception as e:
                    print(f'[LR] WARNING: warmup scheduler load failed ({e}), '
                          f'falling back to epoch alignment.')
                    self._align_warmup_scheduler(resumed_epoch)
            else:
                # 场景 A/C：手动对齐 warmup scheduler 到 resumed_epoch
                self._align_warmup_scheduler(resumed_epoch)
        else:
            # 普通 scheduler（无 warmup）
            if ckpt_sched_state is not None:
                try:
                    self.lr_scheduler.load_state_dict(ckpt_sched_state)
                except Exception as e:
                    print(f'[LR] WARNING: scheduler load failed ({e}), '
                          f'aligning by epoch.')
                    self.lr_scheduler.last_epoch = resumed_epoch - 1
                    self.lr_scheduler.step()
            else:
                # 旧格式 checkpoint，手动对齐
                self.lr_scheduler.last_epoch = resumed_epoch - 1
                self.lr_scheduler.step()

        if self.settings.local_rank in [-1, 0]:
            try:
                current_lrs = self.lr_scheduler.get_last_lr()
            except Exception:
                current_lrs = [g['lr'] for g in self.optimizer.param_groups]
            print(f'[LR] After resume (epoch={resumed_epoch}): {current_lrs}')

    def _align_warmup_scheduler(self, resumed_epoch: int):
        """
        将 WarmupSchedulerWrapper 对齐到 resumed_epoch，
        使下一个 epoch（resumed_epoch+1）的 LR 正确。

        逻辑：
          - 若 resumed_epoch <= warmup_epoch：设置 warmup lr
          - 若 resumed_epoch >  warmup_epoch：对齐 base_scheduler
        """
        ws = self.lr_scheduler
        ws.last_epoch = resumed_epoch

        if resumed_epoch <= ws.warmup_epoch:
            # 仍在 warmup 区间：设置当前 warmup lr
            ws._set_warmup_lr(resumed_epoch)
            ws._warmup_finished = False
            if self.settings.local_rank in [-1, 0]:
                print(f'[LR] Aligned to warmup phase: epoch={resumed_epoch}/'
                      f'{ws.warmup_epoch}')
        else:
            # warmup 已结束：将 base_scheduler 对齐到正确位置
            ws._warmup_finished = True
            offset = resumed_epoch - ws.warmup_epoch
            ws.base_scheduler.last_epoch = offset - 1
            ws.base_scheduler.step()
            if self.settings.local_rank in [-1, 0]:
                print(f'[LR] Aligned to post-warmup phase: '
                      f'base_scheduler.last_epoch={offset - 1}')

    # ════════════════════════════════════════════════════════════════════
    #  load_state_dict()  ← 原逻辑不变
    # ════════════════════════════════════════════════════════════════════
    def load_state_dict(self, checkpoint=None, distill=False):
        if distill:
            net = (self.actor.net_teacher.module
                   if multigpu.is_multi_gpu(self.actor.net_teacher)
                   else self.actor.net_teacher)
        else:
            net = (self.actor.net.module
                   if multigpu.is_multi_gpu(self.actor.net)
                   else self.actor.net)
        net_type = type(net).__name__
        if isinstance(checkpoint, str):
            if os.path.isdir(checkpoint):
                checkpoint_list = sorted(
                    glob.glob('{}/*_ep*.pth.tar'.format(checkpoint))
                )
                if checkpoint_list:
                    checkpoint_path = checkpoint_list[-1]
                else:
                    raise Exception('No checkpoint found')
            else:
                checkpoint_path = os.path.expanduser(checkpoint)
        else:
            raise TypeError
        print("Loading pretrained model from ", checkpoint_path)
        checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
        assert net_type == checkpoint_dict['net_type'], \
            'Network is not of correct type.'
        missing_k, unexpected_k = net.load_state_dict(
            checkpoint_dict["net"], strict=False
        )
        print("previous checkpoint is loaded.")
        print("missing keys: ", missing_k)
        print("unexpected keys:", unexpected_k)
        return True