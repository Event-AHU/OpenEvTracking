import _init_paths
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 8]
import argparse

from lib.test.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from lib.test.evaluation import get_dataset, trackerlist


if __name__ == '__main__':
    trackers = []
    dataset_name = 'fe108'
    parameter_name = f'{dataset_name}'

    """aprtrack"""
    trackers.extend(trackerlist(name='aprtrack', parameter_name=parameter_name, dataset_name=dataset_name, run_ids=None, display_name='APRTrack'))
    
    dataset = get_dataset(dataset_name)
    print_results(trackers, dataset, dataset_name, merge_results=True, plot_types=('success', 'norm_prec', 'prec'))