weights_after_training_30_epochs = {
    'name': 'weights-after-training-30-epochs',
    'dataset': 'MNIST',
    'model-config': {
        'width': 30,
        'depth': 1000,
        'activation': 'ReLU',              # 'ReLU' or 'Tanh'
        'lr': 0.02,
        'rbf_bandwidth': 0.1,
        'train_init_final': True,
        'loss': 'CrossEntropyLoss'         # 'CrossEntropyLoss' or 'MSELoss'
    },
    'epochs': 30,
    'n_iter': 10,
}


weights_after_training_30_epochs_no_train_init_final = {
    'name': 'weights-after-training-30-epochs-no-train-init-final',
    'dataset': 'MNIST',
    'model-config': {
        'width': 30,
        'depth': 1000,
        'activation': 'ReLU',              # 'ReLU' or 'Tanh'
        'lr': 0.02,
        'rbf_bandwidth': 0.1,
        'train_init_final': False,
        'loss': 'CrossEntropyLoss'         # 'CrossEntropyLoss' or 'MSELoss'
    },
    'epochs': 30,
    'n_iter': 10,
}


weights_after_training_50_epochs_no_train_init_final = {
    'name': 'weights-after-training-50-epochs-no-train-init-final',
    'dataset': 'MNIST',
    'model-config': {
        'width': 30,
        'depth': 1000,
        'activation': 'ReLU',              # 'ReLU' or 'Tanh'
        'lr': 0.02,
        'rbf_bandwidth': 0.1,
        'train_init_final': False,
        'loss': 'CrossEntropyLoss'         # 'CrossEntropyLoss' or 'MSELoss'
    },
    'epochs': 50,
    'n_iter': 20,
}


penalized_lip_01_max_0_50epochs_no_train_init_final = {
    'name': 'penalized-lip-0.1-max-0-50-epochs-no-train-init-final',
    'dataset': 'MNIST',
    'model-config': {
        'width': 30,
        'depth': 1000,
        'activation': 'ReLU',              # 'ReLU' or 'Tanh'
        'lr': 0.02,
        'rbf_bandwidth': 0.1,
        'train_init_final': False,
        'loss': 'CrossEntropyLoss',         # 'CrossEntropyLoss' or 'MSELoss'
        'lambda_lip': 0.1
    },
    'epochs': 50,
    'n_iter': 20,
}


penalized_lip_001_max_0_50epochs_no_train_init_final = {
    'name': 'penalized-lip-0.01-max-0-50-epochs-no-train-init-final',
    'dataset': 'MNIST',
    'model-config': {
        'width': 30,
        'depth': 1000,
        'activation': 'ReLU',              # 'ReLU' or 'Tanh'
        'lr': 0.02,
        'rbf_bandwidth': 0.1,
        'train_init_final': False,
        'loss': 'CrossEntropyLoss',         # 'CrossEntropyLoss' or 'MSELoss'
        'lambda_lip': 0.01
    },
    'epochs': 50,
    'n_iter': 20,
}


penalization_two_max_01 = {
    'name': 'penalization-two-max-01',
    'dataset': 'MNIST',
    'model-config': {
        'width': 30,
        'depth': 1000,
        'activation': 'ReLU',              # 'ReLU' or 'Tanh'
        'lr': 0.02,
        'rbf_bandwidth': 0.1,
        'train_init_final': False,
        'loss': 'CrossEntropyLoss',         # 'CrossEntropyLoss' or 'MSELoss'
        'lambda_two_max': 0.1
    },
    'epochs': 50,
    'n_iter': 20,
}


penalization_two_max_1 = {
    'name': 'penalization-two-max-1',
    'dataset': 'MNIST',
    'model-config': {
        'width': 30,
        'depth': 1000,
        'activation': 'ReLU',              # 'ReLU' or 'Tanh'
        'lr': 0.02,
        'rbf_bandwidth': 0.1,
        'train_init_final': False,
        'loss': 'CrossEntropyLoss',         # 'CrossEntropyLoss' or 'MSELoss'
        'lambda_two_max': 1.0
    },
    'epochs': 50,
    'n_iter': 20,
}


penalization_max_max_1 = {
    'name': 'penalization-max-max-1',
    'dataset': 'MNIST',
    'model-config': {
        'width': 30,
        'depth': 1000,
        'activation': 'ReLU',              # 'ReLU' or 'Tanh'
        'lr': 0.02,
        'rbf_bandwidth': 0.1,
        'train_init_final': False,
        'loss': 'CrossEntropyLoss',         # 'CrossEntropyLoss' or 'MSELoss'
        'lambda_max_max': 1.0
    },
    'epochs': 50,
    'n_iter': 20,
}


penalization_max_max_10 = {
    'name': 'penalization-max-max-10',
    'dataset': 'MNIST',
    'model-config': {
        'width': 30,
        'depth': 1000,
        'activation': 'ReLU',              # 'ReLU' or 'Tanh'
        'lr': 0.02,
        'rbf_bandwidth': 0.1,
        'train_init_final': False,
        'loss': 'CrossEntropyLoss',         # 'CrossEntropyLoss' or 'MSELoss'
        'lambda_max_max': 10.0
    },
    'epochs': 50,
    'n_iter': 20,
}


weights_after_training_50_epochs_no_train_init_final_tied = {
    'name': 'weights-tied',
    'dataset': 'MNIST',
    'model-config': {
        'width': 30,
        'depth': 1000,
        'activation': 'ReLU',              # 'ReLU' or 'Tanh'
        'lr': 0.02,
        'rbf_bandwidth': 0.1,
        'train_init_final': False,
        'loss': 'CrossEntropyLoss'         # 'CrossEntropyLoss' or 'MSELoss'
    },
    'epochs': 50,
    'n_iter': 20,
}
