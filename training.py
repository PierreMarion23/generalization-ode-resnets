import glob
import os
import pickle
import time

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning import loggers
import torch

import data
import models
import utils


class SaveWeightMetrics(Callback):
    """Save metrics during training."""

    def __init__(self, test_dl, device):
        self.epoch_counter = -1
        self.first_batch = True
        self.test_dl = test_dl
        self.device = device

    def compute_lipschitz_norm(self, weights):
        return torch.max(torch.stack([weights[k+1] - weights[k] for k in range(len(weights)-1)]))
    
    def save_metrics(self, resnet):
        lipschitz_norm = self.compute_lipschitz_norm([layer.weight for layer in resnet.outer_weights])
        resnet.logger.log_metrics(
            {"lipschitz_norm": lipschitz_norm, "epoch": self.epoch_counter})
        true_targets, predictions = utils.get_true_targets_predictions(self.test_dl, resnet, self.device)
        accuracy = np.mean(np.array(true_targets) == np.array(predictions))
        resnet.logger.log_metrics({"test/accuracy": accuracy, "epoch": self.epoch_counter})
        loss = utils.get_eval_loss(self.test_dl, resnet, self.device)
        resnet.logger.log_metrics({"test/loss": loss, "epoch": self.epoch_counter})
        
    def on_train_batch_end(self, trainer, resnet, outputs, batch, batch_idx):
        """Compute train loss before training (loss of the first batch of the first epoch)."""
        if self.first_batch:
            resnet.logger.log_metrics({"train/loss": outputs['loss'], "epoch": -1})
            self.first_batch = False
        
    def on_train_epoch_start(self, trainer, resnet):
        """Compute metrics before training (epoch=-1)."""
        if self.epoch_counter == -1:
            self.save_metrics(resnet)

    def on_train_epoch_end(self, trainer, resnet):
        """Compute metrics after each epoch."""
        self.epoch_counter += 1
        self.save_metrics(resnet)


def get_results(exp_name: str) -> list:
    """Read the results saved after execution of the training file.

    :param exp_name: name of the configuration
    :return: list of results
    """
    results = {'rbf_bandwidth': [], 'lr': [], 'dataset': [], 'activation': [], 
               'width': [], 'depth': [], 'train_init_final': [], 'lambda': [], 
               'epoch': [], 'lipschitz_norm': [], 'train_loss': [], 
               'test_loss': [], 'test_accuracy': []}
    # for loop over all runs.
    for directory in glob.glob(os.path.join('results', exp_name, '*')):
        with open(os.path.join(directory, 'metrics.csv'), 'r') as f:
            csv_log = pd.read_csv(f)
        with open(os.path.join(directory, 'config.pkl'), 'rb') as f:
            config = pickle.load(f)
        for epoch in range(-1, config['epochs']):   # epoch=-1 <-> before training.
            results['rbf_bandwidth'].append(
                config['model-config']['rbf_bandwidth'])
            results['lr'].append(config['model-config']['lr'])
            results['dataset'].append(config['dataset'])
            results['activation'].append(config['model-config']['activation'])
            results['width'].append(config['model-config']['width'])
            results['depth'].append(config['model-config']['depth'])
            results['train_init_final'].append(config['model-config']['train_init_final'])
            if 'lambda_lip' in config['model-config']:
                results['lambda'].append(config['model-config']['lambda_lip'])
            else:
                results['lambda'].append(0)
            results['epoch'].append(epoch + 1)   # index offset.
            results['lipschitz_norm'].append(
                float(csv_log[csv_log['lipschitz_norm'].notnull() & (csv_log['epoch']==epoch)]['lipschitz_norm']))
            results['train_loss'].append(
                float(csv_log[csv_log['train/loss'].notnull() & (csv_log['epoch']==epoch)]['train/loss']))
            results['test_loss'].append(
                    float(csv_log[csv_log['test/loss'].notnull() & (csv_log['epoch']==epoch)]['test/loss']))
            results['test_accuracy'].append(
                    float(csv_log[csv_log['test/accuracy'].notnull() & (csv_log['epoch']==epoch)]['test/accuracy']))

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
            callbacks=[SaveWeightMetrics(test_dl, device)],
            enable_checkpointing=False,
            enable_progress_bar=verbose,
            enable_model_summary=verbose
        )
    trainer.fit(model, train_dl)

    print('Training finished')
    
    trainer.save_checkpoint(f'{results_dir}/model.ckpt')

    with open(f'{results_dir}/config.pkl', 'wb') as f:
        pickle.dump(config_dict, f)

    return model
