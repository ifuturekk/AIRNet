import os
import numpy as np
import pandas as pd
import itertools
import torch

from configs import Config
from dataloader.dataloader import dataloader
from models import modules
from trainer import TSINet3trainer, evaluator

import warnings

warnings.filterwarnings('ignore')
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['OMP_NUM_THREADS'] = '128'

configs = Config()

w1, w2, w3, w4 = [1.0, 0.1, 0.0], [1.0, 0.1, 0.0], [1.0, 0.1, 0.0], [1.0, 0.1, 0.0]
# weights_list = [(0.0, 0.0, 0.0)] + list(itertools.product(w1, w2, w3))
'''weights_list = [[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.1, 0.1], [1.0, 1.0, 0.01, 0.01]] \
               + [[0.0, 1.0, 0.1, 0.1], [1.0, 1.0, 0.1, 0.0], [0.0, 1.0, 0.01, 0.01], [1.0, 1.0, 0.01, 0.0]] \
               + [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.1, 0.0], [0.0, 0.0, 0.0, 0.1]]'''
weights_list = [[0.1, 1.0, 0.1, 0.1]]
# weights_list = ['awl']
batch_sizes = [64, 128, 256, 512, 1024]

print('Experiments begin...')
results = pd.DataFrame()
for task in configs.tasks:
    train_dl, val_dl, test_dl = dataloader(configs, task[0], task[1])  # , images=True
    weights = weights_list[0]
    for batch_size, num in list(itertools.product(batch_sizes, range(configs.repeat_num))):
        configs.batch_size = batch_size
        encoder = modules.ResNet(configs)
        classifier = modules.MLP2(32, 16, 3)
        model_TSINet3 = modules.TSINet3(encoder, classifier,
                                        modules.MLP2GRL(32, 16, len(task[0])),
                                        modules.MLP1(32, 16))

        optimizer = torch.optim.Adam(model_TSINet3.parameters(), 1e-2)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 500, 0.5)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 70], gamma=0.1)

        print('Current task: ', task, '| Weights: ', weights, '| Repeat number: ', num)
        TSINet3trainer(model_TSINet3, train_dl, test_dl, weights, optimizer, scheduler, configs, task[1], num)

        print('Begin testing...')
        checkpoint_best_val = torch.load(os.path.join(configs.exp_log_dir, f'TSINet3_best_val_{task[1]}_{num}.pt'))
        print('The model with best ==val== is obtained at epoch: ', checkpoint_best_val['epoch'])
        encoder.load_state_dict(checkpoint_best_val['encoder_dict'])
        classifier.load_state_dict(checkpoint_best_val['classifier_dict'])
        model_best_val = modules.TSINet(encoder, classifier)
        accuracy, recall, precision, f1 = evaluator(model_best_val, test_dl, configs, False)
        result = {'Source domains': task[0], 'Target domain': task[1], 'Number': num, 'Accuracy': np.round(accuracy, 4),
                  'Precision': np.round(precision, 4), 'Recall': np.round(recall, 4), 'F1 Score': np.round(f1, 4),
                  'flag': 'best val', 'weights': weights, 'epoch': checkpoint_best_val['epoch'], 'bs': batch_size}
        print(result)
        results = pd.concat([results, pd.DataFrame([result])], axis=0)

        checkpoint_best_loss = torch.load(os.path.join(configs.exp_log_dir, f'TSINet3_best_loss_{task[1]}_{num}.pt'))
        print('The model with best ==loss== is obtained at epoch: ', checkpoint_best_loss['epoch'])
        encoder.load_state_dict(checkpoint_best_loss['encoder_dict'])
        classifier.load_state_dict(checkpoint_best_loss['classifier_dict'])
        model_best_loss = modules.TSINet(encoder, classifier)
        accuracy, recall, precision, f1 = evaluator(model_best_loss, test_dl, configs, False)
        result = {'Source domains': task[0], 'Target domain': task[1], 'Number': num, 'Accuracy': np.round(accuracy, 4),
                  'Precision': np.round(precision, 4), 'Recall': np.round(recall, 4), 'F1 Score': np.round(f1, 4),
                  'flag': 'best loss', 'weights': weights, 'epoch': checkpoint_best_loss['epoch'], 'bs': batch_size}
        print(result)
        results = pd.concat([results, pd.DataFrame([result])], axis=0)

        checkpoint_last = torch.load(os.path.join(configs.exp_log_dir, f'TSINet3_last_{task[1]}_{num}.pt'))
        print('The model obtained at the last epoch: ', checkpoint_last['epoch'])
        encoder.load_state_dict(checkpoint_last['encoder_dict'])
        classifier.load_state_dict(checkpoint_last['classifier_dict'])
        model_best_loss = modules.TSINet(encoder, classifier)
        accuracy, recall, precision, f1 = evaluator(model_best_loss, test_dl, configs, False)
        result = {'Source domains': task[0], 'Target domain': task[1], 'Number': num, 'Accuracy': np.round(accuracy, 4),
                  'Precision': np.round(precision, 4), 'Recall': np.round(recall, 4), 'F1 Score': np.round(f1, 4),
                  'flag': 'last', 'weights': weights, 'epoch': checkpoint_last['epoch'], 'bs': batch_size}
        print(result)
        results = pd.concat([results, pd.DataFrame([result])], axis=0)

        # Save results once per experiment to prevent run interruptions
        results.to_excel(f'results_cwt_cutmix_DCI_sensitivity.xlsx', index=False)

