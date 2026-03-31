# tracking/analysis_results.py
# python tracking/analysis_results.py --dataset fe108 --parameter_name ceutrack_fe108_sor --epoch 10
import _init_paths
import matplotlib
matplotlib.use('Agg')          # 服务器无显示器时用 Agg 后端
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 8]
import argparse

from lib.test.analysis.plot_results import (
    plot_results, print_results, print_per_sequence_results
)
from lib.test.evaluation import get_dataset, trackerlist


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate CEUTrack results on FE108/VisEvent/COESOT.'
    )
    parser.add_argument(
        '--dataset', type=str, default='fe108',
        choices=['fe108', 'visevent', 'coesot'],
        help='评测数据集名称'
    )
    parser.add_argument(
        '--parameter_name', type=str, default='ceutrack_fe108',
        help='与 tracking/test.sh -c 保持一致的 config 名'
    )
    parser.add_argument(
        '--epoch', type=int, default=40,
        help='评测的 epoch（对应 run_id，决定结果目录后缀 _040）'
    )
    parser.add_argument(
        '--force_eval', action='store_true',
        help='强制重新计算（忽略缓存的 eval_data.pkl）'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    dataset_name   = args.dataset           # 'fe108'
    parameter_name = args.parameter_name    # 'ceutrack_fe108'
    epoch          = args.epoch             # 40

    # run_id=epoch → results_dir = .../ceutrack_fe108_040
    trackers = trackerlist(
        name='ceutrack',
        parameter_name=parameter_name,
        dataset_name=dataset_name,
        run_ids=epoch,                      # ← 关键：传入 epoch 编号
        display_name=f'CEUTrack-FE108-ep{epoch:03d}'
    )

    dataset = get_dataset(dataset_name)

    # 打印指标表格（AUC / OP50 / OP75 / Precision / Norm-Precision）
    print_results(
        trackers,
        dataset,
        report_name=dataset_name,           # eval_data.pkl 保存目录名
        merge_results=True,
        plot_types=('success', 'norm_prec', 'prec'),
        force_evaluation=args.force_eval    # True 时忽略缓存
    )