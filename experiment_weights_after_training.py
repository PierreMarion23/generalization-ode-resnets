import distutils.spawn
import os
from typing import Optional

from matplotlib import rc
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from torch.multiprocessing import Pool

import config
import training

sns.set(font_scale=1.5)

if distutils.spawn.find_executable('latex'):
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)


def run_experiment(exp_config: dict, filepath: Optional[str] = 'figures'):
    """Train a model and plot a random weight as a function
    of the layer index.

    :param exp_config: configuration of the experiment
    :param filepath: path to the folder where the figures should be saved
    :return:
    """

    with Pool(processes=exp_config['n_workers']) as pool:
        pool.map(training.fit, [exp_config]*exp_config['n_iter'])


if __name__ == '__main__':
    exp_config = config.weights_after_training
    filepath = 'figures/weights_after_training'
    os.makedirs(filepath, exist_ok=True)
    run_experiment(exp_config, filepath)


# add error bars
# change the regularity value
# classification vs regression
# change activation function
# try without training init and final layers
