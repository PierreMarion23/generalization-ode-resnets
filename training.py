import copy
import glob
import os
import pickle
import time
from typing import Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning import loggers
import torch
from torch.multiprocessing import Pool

import data
import models
import utils


class SaveWeightMetrics(Callback):
    def __init__(self):
        self.epoch_counter = 0

    def compute_max_norm(self, weights):
        return torch.max(torch.Tensor([torch.sum(torch.abs(weights[k])) for k in range(len(weights))]))

    def compute_lipschitz_norm(self, weights):
        return torch.max(torch.stack([weights[k+1] - weights[k] for k in range(len(weights)-1)]))

    def on_train_epoch_end(self, trainer, resnet):
        self.epoch_counter += 1
        max_norm = self.compute_max_norm([layer.weight for layer in resnet.outer_weights])
        resnet.logger.log_metrics(
            {"max_norm": max_norm, "epoch": self.epoch_counter})
        lipschitz_norm = self.compute_lipschitz_norm([layer.weight for layer in resnet.outer_weights])
        resnet.logger.log_metrics(
            {"lipschitz_norm": lipschitz_norm, "epoch": self.epoch_counter})



def get_results(exp_name: str) -> list:
    """Read the results saved after execution of the training file.

    :param exp_name: name of the configuration
    :return: list of results
    """
    results = {'accuracy': [], 'rbf_bandwidth': [], 'lr': [], 'dataset': [], 'activation': [], 'width': [], 'depth': [], 'train_init_last': [], 'epoch': [], 'max_norm': [], 'lipschitz_norm': []}
    for directory in glob.glob(os.path.join('results', exp_name, '*')):
        with open(os.path.join(directory, 'metrics.csv'), 'r') as f:
            csv_log = pd.read_csv(f)
        with open(os.path.join(directory, 'config.pkl'), 'rb') as f:
            config = pickle.load(f)
        with open(os.path.join(directory, 'metrics.pkl'), 'rb') as f:
            metrics = pickle.load(f)
        for k in range(config['epochs']):
            results['rbf_bandwidth'].append(
                config['model-config']['rbf_bandwidth'])
            results['lr'].append(config['model-config']['lr'])
            results['dataset'].append(config['dataset'])
            results['activation'].append(config['model-config']['activation'])
            results['width'].append(config['model-config']['width'])
            results['depth'].append(config['model-config']['depth'])
            results['train_init_last'].append(config['model-config']['train_init_last'])
            results['accuracy'].append(metrics['test_accuracy'])
            results['max_norm'].append(float(csv_log[csv_log['max_norm'].notnull()][csv_log['epoch']==k]['max_norm']))
            results['lipschitz_norm'].append(float(csv_log[csv_log['lipschitz_norm'].notnull()][csv_log['epoch']==k]['lipschitz_norm']))

    return results


def fit(config_dict: dict, verbose: bool = False) -> pl.LightningModule:
    """Train a ResNet following the configuration.

    :param config_dict: configuration of the network and dataset
    :param verbose: print information about traning
    :return: the trained model
    """
    name = config_dict['name'].replace('dataset', config_dict['dataset'])

    train_dl, test_dl, first_coord, nb_classes = data.load_dataset(
        config_dict['dataset'], vectorize=True)

    model = models.ResNet(
        first_coord=first_coord, final_width=nb_classes,
        **config_dict['model-config'])

    gpu = 1 if torch.cuda.is_available() else 0
    device = torch.device("cuda") if torch.cuda.is_available() \
        else torch.device("cpu")
    
    results_dir = f'{os.getcwd()}/results/{name}/{str(time.time())}'
    os.makedirs(results_dir, exist_ok=True)

    logger = loggers.CSVLogger(save_dir=results_dir, name='', version='')
    trainer = pl.Trainer(
            gpus=gpu,
            max_epochs=config_dict['epochs'],
            logger=logger,
            callbacks=[SaveWeightMetrics()],
            enable_checkpointing=False,
            enable_progress_bar=verbose,
            enable_model_summary=verbose
        )
    trainer.fit(model, train_dl)

    print('Training finished')
    true_targets, predictions = utils.get_true_targets_predictions(
        test_dl, model, device)
    accuracy = np.mean(np.array(true_targets) == np.array(predictions))
    loss = utils.get_eval_loss(test_dl, model, device)

    metrics = {'test_accuracy': accuracy, 'test_loss': loss}
    if verbose:
        print(f'Test accuracy: {accuracy}')
    trainer.save_checkpoint(f'{results_dir}/model.ckpt')

    with open(f'{results_dir}/metrics.pkl', 'wb') as f:
        pickle.dump(metrics, f)

    with open(f'{results_dir}/config.pkl', 'wb') as f:
        pickle.dump(config_dict, f)

    return model


def fit_parallel(exp_config: dict,
                 grid_lr: list,
                 grid_regularity: list,
                 grid_scaling: list,
                 resume_experiment: Optional[bool] = False):
    """Train in parallel ResNet with different learning rate, scaling, and
    initialization.

    :param config: configuration of the network and dataset
    :param grid_lr: grid of learning rates
    :param grid_regularity: grid of initialization regularities
    :param grid_scaling: grid of scaling values
    :param resume_experiment: if True, will look in the results folder if
    the grid was partially explored and skip the runs which were already
    performed
    :return:
    """
    if resume_experiment:
        previous_results = get_results(exp_config['name'].replace('dataset', 'MNIST'))
        found_experiments = [(previous_results['lr'][k], 
                             previous_results['regularity'][k],
                             previous_results['scaling'][k]) 
                             for k in range(len(previous_results['lr']))
                            ]
    else:
        found_experiments = []
    list_configs = []
    for lr in grid_lr:
        for reg in grid_regularity:
            for scaling in grid_scaling:
                if (lr, reg, scaling) not in found_experiments:
                    exp_config['model-config']['lr'] = lr
                    exp_config['model-config']['regularity']['value'] = reg
                    exp_config['model-config']['scaling'] = scaling
                    list_configs.append(copy.deepcopy(exp_config))
    with Pool(processes=exp_config['n_workers']) as pool:
        pool.map(fit, list_configs)
