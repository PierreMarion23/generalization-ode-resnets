import distutils.spawn
import os
from typing import Optional

from matplotlib import rc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from torch.multiprocessing import Pool

import config
import training

sns.set(font_scale=1.5)

if distutils.spawn.find_executable('latex'):
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)


def plot(results, filepath):
    os.makedirs(filepath, exist_ok=True)

    df = pd.DataFrame.from_dict(results)
    df['log_train_loss'] = np.log(df['train_loss'])
    df['log_test_loss'] = np.log(df['test_loss'])

    plt.figure()
    sns.lineplot(data=df, x='epoch', y='max_norm', label='Max norm')
    plt.ylabel('')
    ax2 = plt.twinx()
    sns.lineplot(data=df, ax=ax2, x='epoch', y='log_train_loss', color='y', label='Train loss')
    sns.lineplot(data=df[df['epoch']>0], ax=ax2, x='epoch', y='log_test_loss', color='r', label='Test loss')
    plt.ylabel('')
    plt.legend()
    plt.savefig(os.path.join(filepath, 'max-norm-loss.pdf'), bbox_inches='tight')
    plt.close()

    plt.figure()
    sns.lineplot(data=df, x='epoch', y='lipschitz_norm', label='Lipschitz norm')
    plt.ylabel('')
    ax2 = plt.twinx()
    sns.lineplot(data=df, ax=ax2, x='epoch', y='log_train_loss', color='y', label='Train loss')
    sns.lineplot(data=df[df['epoch']>0], ax=ax2, x='epoch', y='log_test_loss', color='r', label='Test loss')
    plt.ylabel('')
    plt.legend()
    plt.savefig(os.path.join(filepath, 'lipschitz-norm-loss.pdf'), bbox_inches='tight')
    plt.close()

    plt.figure()
    sns.lineplot(data=df, x='epoch', y='max_norm', label='Max norm')
    plt.ylabel('')
    ax2 = plt.twinx()
    sns.lineplot(data=df[df['epoch']>0], ax=ax2, x='epoch', y='test_accuracy', color='r', label='Test accuracy')
    plt.ylabel('')
    plt.legend()
    plt.savefig(os.path.join(filepath, 'max-norm-accuracy.pdf'), bbox_inches='tight')
    plt.close()

    plt.figure()
    sns.lineplot(data=df, x='epoch', y='lipschitz_norm', label='Lipschitz norm')
    plt.ylabel('')
    ax2 = plt.twinx()
    sns.lineplot(data=df[df['epoch']>0], ax=ax2, x='epoch', y='test_accuracy', color='r', label='Test accuracy')
    plt.ylabel('')
    plt.legend()
    plt.savefig(os.path.join(filepath, 'lipschitz-norm-accuracy.pdf'), bbox_inches='tight')
    plt.close()


def run_experiment(exp_config: dict, filepath: Optional[str] = 'figures'):
    """Train a model and plot a random weight as a function
    of the layer index.

    :param exp_config: configuration of the experiment
    :param filepath: path to the folder where the figures should be saved
    :return:
    """

    for _ in range(exp_config['n_iter']):
        training.fit(exp_config, verbose=True)
    results = training.get_results(exp_config['name'])
    plot(results, os.path.join(filepath, exp_config['name']))


if __name__ == '__main__':
    filepath = 'figures'
    run_experiment(config.weights_after_training_mse_30_epochs_no_train_init_final, filepath)
