import os
import time
from time import strftime, gmtime
from tqdm import tqdm
import sklearn.metrics as metrics

import torch

from dataloader import cutmix, mixup
from models import losses
from utils import AutomaticWeightedLoss, domain_label_reset, entropy_filtering


def trainer(model, optimizer, scheduler, loss_fn, train_dl, val_dl, configs):
    start = time.time()
    best_f1 = 0
    best_loss = 1e2
    for epoch in tqdm(range(configs.train_epoch)):
        model.to(configs.train_device)
        model.train()
        total_loss = []
        for batch_idx, (inputs, labels, _) in enumerate(train_dl):
            inputs, labels = inputs.float().to(configs.train_device), labels.long().to(configs.train_device)
            optimizer.zero_grad()
            outputs = model(inputs)
            # print(inputs.shape, outputs.shape, labels.shape)

            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss.append(loss.item())
        scheduler.step()
        ave_loss = torch.tensor(total_loss).mean()
        accuracy, recall, precision, f1 = evaluator(model, val_dl, configs)
        if f1 > best_f1:
            print('F1 score has been updated!')
            print(f'Accuracy: {accuracy: 2.5f}\t',
                  f'Recall: {recall: 2.5f}\t',
                  f'Precision: {precision: 2.5f}\t',
                  f'F1 score: {f1: 2.5f}\t')
            best_f1 = f1
            checkpoint_val = {'epoch': epoch,
                              'model_dict': model.state_dict()}
        if ave_loss < best_loss:
            print('Best loss has been updated!')
            print(f'Accuracy: {accuracy: 2.5f}\t',
                  f'Recall: {recall: 2.5f}\t',
                  f'Precision: {precision: 2.5f}\t',
                  f'F1 score: {f1: 2.5f}\t')
            best_loss = ave_loss
            checkpoint_loss = {'epoch': epoch,
                               'model_dict': model.state_dict()}

        if epoch % 10 == 0 or epoch == configs.train_epoch - 1:
            print(f'Epoch: {epoch}\t | \tTotal training loss: {ave_loss: 2.6f}')

    checkpoint_last = {'epoch': configs.train_epoch,
                       'model_dict': model.state_dict()}

    os.makedirs(os.path.join(configs.exp_log_dir), exist_ok=True)
    torch.save(checkpoint_val, os.path.join(configs.exp_log_dir, 'model_best_val.pt'))
    torch.save(checkpoint_loss, os.path.join(configs.exp_log_dir, 'model_best_loss.pt'))
    torch.save(checkpoint_last, os.path.join(configs.exp_log_dir, 'model_last.pt'))

    end = time.time()
    total_time = end - start
    print('Total training time: {}'.format(strftime('%H:%M:%S', gmtime(total_time))))


def evaluator(model, eval_dl, configs, mo=False):
    model.to(configs.eval_device)
    model.eval()
    with torch.no_grad():
        true_labels, pred_labels = torch.tensor([]), torch.tensor([])
        for batch_idx, (inputs, labels, _) in enumerate(eval_dl):
            inputs, labels = inputs.float().to(configs.eval_device), labels.long().to(configs.eval_device)
            if mo:
                outputs, _, _ = model(inputs)
            else:
                outputs = model(inputs)

            _, predicted = torch.max(outputs.data, dim=1)

            true_labels = torch.cat((true_labels, labels), dim=0)
            pred_labels = torch.cat((pred_labels, predicted), dim=0)

        true_labels, pred_labels = true_labels.cpu().numpy(), pred_labels.cpu().numpy()
        cm = metrics.confusion_matrix(true_labels, pred_labels)
        print(cm)
        val_accuracy = metrics.accuracy_score(true_labels, pred_labels)
        val_recall = metrics.recall_score(true_labels, pred_labels, average='macro')
        val_precision = metrics.precision_score(true_labels, pred_labels, average='macro')
        val_f1 = metrics.f1_score(true_labels, pred_labels, average='macro')

    return val_accuracy, val_recall, val_precision, val_f1


def TSINet3trainer(model, train_dl, val_dl, weights, optimizer, scheduler, configs, target, num):
    cf_loss_fn = torch.nn.CrossEntropyLoss()
    dcf_loss_fn = torch.nn.CrossEntropyLoss()
    ct_loss_fn = losses.SupConLoss(configs.train_device)
    awloss = AutomaticWeightedLoss(5)
    start = time.time()
    best_f1, best_loss = 0, 1e2
    for epoch in tqdm(range(configs.train_epoch + configs.pre_train_epoch)):
        model.to(configs.train_device)
        model.train()
        total_loss = []
        total_cf_loss, total_dcf_loss, total_ct_loss, total_mixed_cf_loss, total_mixed_dcf_loss = [], [], [], [], []
        for batch_idx, (inputs, labels, d_labels) in enumerate(train_dl):
            # print(labels)
            inputs, labels = inputs.float().to(configs.train_device), labels.long().to(configs.train_device)

            optimizer.zero_grad()
            classes, domains, features = model(inputs)
            cf_loss = cf_loss_fn(classes, labels)
            loss = cf_loss
            if (epoch >= configs.pre_train_epoch) and any(weights):
                d_labels = domain_label_reset(d_labels)
                # print(d_labels)
                d_labels = d_labels.long().to(configs.train_device)

                filter_idx = entropy_filtering(classes, labels)
                # print('raw: ', inputs.shape[0])
                inputs, labels, d_labels = inputs[filter_idx], labels[filter_idx], d_labels[filter_idx]
                # print('after filtering: ', inputs.shape[0])
                domains, features = domains[filter_idx], features[filter_idx]

                aug_inputs = cutmix.cutmix_group((inputs, labels), configs.cutmix_ab)
                mixed_inputs, mixed_targets, mixed_d_targets = cutmix.cutmix((inputs, labels, d_labels), configs.cutmix_ab)

                _, _, aug_features = model(aug_inputs)
                all_features = torch.cat([features.unsqueeze(1), aug_features.unsqueeze(1)], dim=1)
                ct_loss = ct_loss_fn(all_features, labels)

                mixed_classes, mixed_domains, _ = model(mixed_inputs)
                # print(domains, d_labels)
                dcf_loss = dcf_loss_fn(domains, d_labels)
                mixed_loss_fn = cutmix.CutMixCriterion()
                mixed_cf_loss = mixed_loss_fn(mixed_classes, mixed_targets)
                mixed_dcf_loss = mixed_loss_fn(mixed_domains, mixed_d_targets)

                # print(cf_loss, dcf_loss, ct_loss, mixed_cf_loss, mixed_dcf_loss)
                # loss = awloss(cf_loss, dcf_loss, ct_loss, mixed_cf_loss, mixed_dcf_loss)
                # loss = cf_loss + dcf_loss + ct_loss + mixed_cf_loss + mixed_dcf_loss
                loss = cf_loss + weights[0] * mixed_cf_loss + weights[1] * ct_loss + weights[2] * dcf_loss + weights[3] * mixed_dcf_loss
                total_dcf_loss.append(dcf_loss.item())
                total_ct_loss.append(ct_loss.item())
                total_mixed_cf_loss.append(mixed_cf_loss.item())
                total_mixed_dcf_loss.append(mixed_dcf_loss.item())

            loss.backward()
            optimizer.step()

            total_cf_loss.append(cf_loss.item())
            total_loss.append(loss.item())
        scheduler.step()
        ave_cf_loss = torch.tensor(total_cf_loss).mean()
        ave_dcf_loss = torch.tensor(total_dcf_loss).mean()
        ave_ct_loss = torch.tensor(total_ct_loss).mean()
        ave_mixed_cf_loss = torch.tensor(total_mixed_cf_loss).mean()
        ave_mixed_dcf_loss = torch.tensor(total_mixed_dcf_loss).mean()
        ave_loss = torch.tensor(total_loss).mean()
        accuracy, recall, precision, f1 = evaluator(model, val_dl, configs, True)
        if f1 > best_f1 and epoch >= configs.pre_train_epoch:
            print('F1 score has been updated!')
            print(f'Accuracy: {accuracy: 2.5f}\t',
                  f'Recall: {recall: 2.5f}\t',
                  f'Precision: {precision: 2.5f}\t',
                  f'F1 score: {f1: 2.5f}\t')
            best_f1 = f1
            checkpoint_val = {'epoch': epoch,
                              'encoder_dict': model.encoder.state_dict(),
                              'classifier_dict': model.classifier.state_dict()}
        if ave_cf_loss < best_loss and epoch >= configs.pre_train_epoch:
            print('Best loss has been updated!')
            print(f'Accuracy: {accuracy: 2.5f}\t',
                  f'Recall: {recall: 2.5f}\t',
                  f'Precision: {precision: 2.5f}\t',
                  f'F1 score: {f1: 2.5f}\t')
            best_loss = ave_cf_loss
            checkpoint_loss = {'epoch': epoch,
                               'encoder_dict': model.encoder.state_dict(),
                               'classifier_dict': model.classifier.state_dict()}

        if epoch % 1 == 0 or epoch == configs.train_epoch - 1:
            print(f'Epoch: {epoch}\t|\tTotal: {ave_loss: 2.6f}\t|\tcf: {ave_cf_loss: 2.6f}'
                  f'\t|\tdcf: {ave_dcf_loss: 2.6f}\t|\tct: {ave_ct_loss: 2.6f}'
                  f'\t|\tMixed cf:{ave_mixed_cf_loss: 2.6f}\t|\tMixed dcf: {ave_mixed_dcf_loss: 2.6f}')

    checkpoint_last = {'epoch': epoch,
                       'encoder_dict': model.encoder.state_dict(),
                       'classifier_dict': model.classifier.state_dict()}

    os.makedirs(os.path.join(configs.exp_log_dir), exist_ok=True)
    torch.save(checkpoint_val, os.path.join(configs.exp_log_dir, f'TSINet3_best_val_{target}_{num}.pt'))
    torch.save(checkpoint_loss, os.path.join(configs.exp_log_dir, f'TSINet3_best_loss_{target}_{num}.pt'))
    torch.save(checkpoint_last, os.path.join(configs.exp_log_dir, f'TSINet3_last_{target}_{num}.pt'))

    end = time.time()
    total_time = end - start
    print('Total training time: {}'.format(strftime('%H:%M:%S', gmtime(total_time))))

