weights_after_training = {
    'name': 'weights-after-training2',
    'dataset': 'MNIST',
    'model-config': {
        'width': 30,
        'depth': 1000,
        'activation': 'ReLU',              # 'ReLU' or 'Tanh'
        'lr': 0.01,
        'rbf_bandwidth': 0.1,
        'train_init_last': True,
    },
    'epochs': 10,
    'n_workers': 5,
    'n_iter': 20,
}
