weights_after_training_baseline = {
    'name': 'weights-after-training-baseline',
    'dataset': 'MNIST',
    'model-config': {
        'width': 30,
        'depth': 1000,
        'activation': 'ReLU',              # 'ReLU' or 'Tanh'
        'lr': 0.01,
        'rbf_bandwidth': 0.1,
        'train_init_final': True,
        'loss': 'CrossEntropyLoss'         # 'CrossEntropyLoss' or 'MSELoss'
    },
    'epochs': 10,
    'n_workers': 5,
    'n_iter': 2,
}


weights_after_training_30_epochs = {
    'name': 'weights-after-training-30-epochs',
    'dataset': 'MNIST',
    'model-config': {
        'width': 30,
        'depth': 1000,
        'activation': 'ReLU',              # 'ReLU' or 'Tanh'
        'lr': 0.01,
        'rbf_bandwidth': 0.1,
        'train_init_final': True,
        'loss': 'CrossEntropyLoss'         # 'CrossEntropyLoss' or 'MSELoss'
    },
    'epochs': 30,
    'n_workers': 5,
    'n_iter': 2,
}


weights_after_training_large_lr = {
    'name': 'weights-after-training-large-lr',
    'dataset': 'MNIST',
    'model-config': {
        'width': 30,
        'depth': 1000,
        'activation': 'ReLU',              # 'ReLU' or 'Tanh'
        'lr': 0.1,
        'rbf_bandwidth': 0.1,
        'train_init_final': True,
        'loss': 'CrossEntropyLoss'         # 'CrossEntropyLoss' or 'MSELoss'
    },
    'epochs': 10,
    'n_workers': 5,
    'n_iter': 2,
}


weights_after_training_large_bandwidth = {
    'name': 'weights-after-training-large-bandwidth',
    'dataset': 'MNIST',
    'model-config': {
        'width': 30,
        'depth': 1000,
        'activation': 'ReLU',              # 'ReLU' or 'Tanh'
        'lr': 0.01,
        'rbf_bandwidth': 1,
        'train_init_final': True,
        'loss': 'CrossEntropyLoss'         # 'CrossEntropyLoss' or 'MSELoss'
    },
    'epochs': 10,
    'n_workers': 5,
    'n_iter': 2,
}


weights_after_training_small_bandwidth = {
    'name': 'weights-after-training-small-bandwidth',
    'dataset': 'MNIST',
    'model-config': {
        'width': 30,
        'depth': 1000,
        'activation': 'ReLU',              # 'ReLU' or 'Tanh'
        'lr': 0.01,
        'rbf_bandwidth': 0.02,
        'train_init_final': True,
        'loss': 'CrossEntropyLoss'         # 'CrossEntropyLoss' or 'MSELoss'
    },
    'epochs': 10,
    'n_workers': 5,
    'n_iter': 2,
}


weights_after_training_tanh_higher_lr = {
    'name': 'weights-after-training-tanh-higher-lr',
    'dataset': 'MNIST',
    'model-config': {
        'width': 30,
        'depth': 1000,
        'activation': 'Tanh',              # 'ReLU' or 'Tanh'
        'lr': 0.03,
        'rbf_bandwidth': 0.1,
        'train_init_final': True,
        'loss': 'CrossEntropyLoss'         # 'CrossEntropyLoss' or 'MSELoss'
    },
    'epochs': 10,
    'n_workers': 5,
    'n_iter': 2,
}


weights_after_training_no_train_init_final = {
    'name': 'weights-after-training-no-train-init-final',
    'dataset': 'MNIST',
    'model-config': {
        'width': 30,
        'depth': 1000,
        'activation': 'ReLU',              # 'ReLU' or 'Tanh'
        'lr': 0.01,
        'rbf_bandwidth': 0.1,
        'train_init_final': False,
        'loss': 'CrossEntropyLoss'         # 'CrossEntropyLoss' or 'MSELoss'
    },
    'epochs': 10,
    'n_workers': 5,
    'n_iter': 2,
}


weights_after_training_no_train_init_final_larger_lr = {
    'name': 'weights-after-training-no-train-init-final-larger-lr',
    'dataset': 'MNIST',
    'model-config': {
        'width': 30,
        'depth': 1000,
        'activation': 'ReLU',              # 'ReLU' or 'Tanh'
        'lr': 0.1,
        'rbf_bandwidth': 0.1,
        'train_init_final': False,
        'loss': 'CrossEntropyLoss'         # 'CrossEntropyLoss' or 'MSELoss'
    },
    'epochs': 10,
    'n_workers': 5,
    'n_iter': 2,
}


weights_after_training_mse = {
    'name': 'weights-after-training-mse',
    'dataset': 'MNIST',
    'model-config': {
        'width': 30,
        'depth': 1000,
        'activation': 'ReLU',              # 'ReLU' or 'Tanh'
        'lr': 0.01,
        'rbf_bandwidth': 0.1,
        'train_init_final': True,
        'loss': 'MSELoss'         # 'CrossEntropyLoss' or 'MSELoss'
    },
    'epochs': 10,
    'n_workers': 5,
    'n_iter': 2,
}


weights_after_training_mse_large_lr = {
    'name': 'weights-after-training-mse-large-lr',
    'dataset': 'MNIST',
    'model-config': {
        'width': 30,
        'depth': 1000,
        'activation': 'ReLU',              # 'ReLU' or 'Tanh'
        'lr': 0.1,
        'rbf_bandwidth': 0.1,
        'train_init_final': True,
        'loss': 'MSELoss'         # 'CrossEntropyLoss' or 'MSELoss'
    },
    'epochs': 10,
    'n_workers': 5,
    'n_iter': 2,
}


weights_after_training_mse_30_epochs = {
    'name': 'weights-after-training-mse-30-epochs',
    'dataset': 'MNIST',
    'model-config': {
        'width': 30,
        'depth': 1000,
        'activation': 'ReLU',              # 'ReLU' or 'Tanh'
        'lr': 0.01,
        'rbf_bandwidth': 0.1,
        'train_init_final': True,
        'loss': 'MSELoss'         # 'CrossEntropyLoss' or 'MSELoss'
    },
    'epochs': 30,
    'n_workers': 5,
    'n_iter': 2,
}


weights_after_training_mse_30_epochs_no_train_init_final = {
    'name': 'weights-after-training-mse-30-epochs-no-train-init-final',
    'dataset': 'MNIST',
    'model-config': {
        'width': 30,
        'depth': 1000,
        'activation': 'ReLU',              # 'ReLU' or 'Tanh'
        'lr': 0.01,
        'rbf_bandwidth': 0.1,
        'train_init_final': False,
        'loss': 'MSELoss'         # 'CrossEntropyLoss' or 'MSELoss'
    },
    'epochs': 30,
    'n_workers': 5,
    'n_iter': 2,
}