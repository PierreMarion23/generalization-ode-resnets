import distutils.spawn
import os

from matplotlib import rc
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import config
import training

sns.set(font_scale=1.5)

if distutils.spawn.find_executable('latex'):
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)


def plot(filepath):
    """Plot the figures of the paper.

    :param filepath: path of the save folder.
    :return:
    """

    os.makedirs(filepath, exist_ok=True)
    
    results_mnist = training.get_results('weights-after-training-30-epochs')
    results_mnist_no_train_init_final = training.get_results('weights-after-training-30-epochs-no-train-init-final')
    df = pd.DataFrame.from_dict(results_mnist)
    df_no = pd.DataFrame.from_dict(results_mnist_no_train_init_final)
    df['gen_gap'] = df['test_loss'] - df['train_loss']
    df_no['gen_gap'] = df_no['test_loss'] - df_no['train_loss']

    plt.figure()
    # We only plot after training has started (not for epoch==0 <-> before training).
    sns.scatterplot(data=df[df['epoch']>0], x='lipschitz_norm', y='gen_gap', marker='<', s=80, label='Trained projections')
    sns.scatterplot(data=df_no[df_no['epoch']>0], x='lipschitz_norm', y='gen_gap', s=55, label='Random projections')
    plt.xlabel(r'$\sup_{0 \leq k \leq L-1}(\|W_{k+1} - W_k\|_\infty)$')
    plt.ylabel('Generalization gap')
    plt.legend()
    plt.savefig(os.path.join(filepath, 'figure-generalization-lipschitz.png'), bbox_inches='tight')
    plt.close()

    df_zero_pen = pd.DataFrame.from_dict(training.get_results('weights-after-training-50-epochs-no-train-init-final'))
    df_pen_01 = pd.DataFrame.from_dict(training.get_results('penalized-lip-0.1-max-0-50-epochs-no-train-init-final'))
    df_pen_001 = pd.DataFrame.from_dict(training.get_results('penalized-lip-0.01-max-0-50-epochs-no-train-init-final'))
    total_df = pd.concat([df_zero_pen, df_pen_01, df_pen_001])
    total_df['gen_gap'] = total_df['test_loss'] - total_df['train_loss']

    plt.figure()
    sns.boxplot(data=total_df[total_df['epoch']==50], x='lambda', y='gen_gap', color='mediumseagreen')
    plt.xlabel(r'$\lambda$')
    plt.ylabel('Generalization gap')
    plt.savefig(os.path.join(filepath, 'figure-generalization-lambda.png'), bbox_inches='tight')
    plt.close()


def run_experiment(exp_config: dict):
    """Train a model according to a given configuration.

    :param exp_config: configuration of the experiment
    :return:
    """

    for k in range(exp_config['n_iter']):
        print('EXPERIMENT {}'.format(exp_config['name']))
        print('ITERATION {}'.format(k))
        training.fit(exp_config, verbose=True)


if __name__ == '__main__':
    run_experiment(config.weights_after_training_30_epochs)
    run_experiment(config.weights_after_training_30_epochs_no_train_init_final)
    run_experiment(config.weights_after_training_50_epochs_no_train_init_final)
    run_experiment(config.penalized_lip_01_max_0_50epochs_no_train_init_final)
    run_experiment(config.penalized_lip_001_max_0_50epochs_no_train_init_final)
    plot('figures')