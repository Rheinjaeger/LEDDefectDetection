from resnet import SupConResNet,resnet50, LinearClassifier, SupConHead
from torch.utils.data import DataLoader
from dataset_SupContrast import LED
import torch
from torch import optim
from losses import SupConLoss

from tqdm import tqdm
import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

import random

def set_random_seed(seed=0):
    # seed setting
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    set_random_seed(42)
    led_dataset = LED()
    weights=[100 if label!=5 else 1 for _,label in led_dataset.Combine_Train]
    from torch.utils.data.sampler import  WeightedRandomSampler
    sampler = WeightedRandomSampler(weights,num_samples=1500,replacement=True)

    Combine_Trainloader = DataLoader(led_dataset.Combine_Train,batch_size=16,sampler=sampler,num_workers=0)
    Combine_Testloader = DataLoader(led_dataset.Combine_Test,batch_size=64,num_workers=0)

    model_local = SupConResNet(name='resnet50')
    head_local = SupConHead(name='resnet50')
    Classifier_local = LinearClassifier(name='resnet50', num_classes=2)

    model_local=model_local.cuda()
    head_local = head_local.cuda()
    Classifier_local = Classifier_local.cuda()

    optimizer_local = optim.SGD(model_local.parameters(),lr=0.001,momentum=0.9,weight_decay=1e-4,nesterov=True)
    optimizer_head_local = optim.SGD(head_local.parameters(),lr=0.001,momentum=0.9,weight_decay=1e-4,nesterov=True)
    optimizer_cls_local = optim.SGD(Classifier_local.parameters(),lr=0.001,momentum=0.9,weight_decay=1e-4,nesterov=True)
    criterion = SupConLoss(temperature=0.1)
    loss_fn = torch.nn.CrossEntropyLoss()

    plt.rcParams['font.sans-serif'] = ['SimHei']
    a=1
    count_float = 0
    p_best = 0
    f1_best = 0
    recall_best = 0
    g_mean_best = 0
    batch_loss = []

    try:
        for epoch in range(10):
            if epoch < a:
                model_local = model_local.train()
                head_local = head_local.train()
                Classifier_local = Classifier_local.train()
            else:
                model_local = model_local.train()
                Classifier_local = Classifier_local.train()
                head_local.requires_grad_(False)
                optimizer_head_local.zero_grad(set_to_none=True)

            train_dataloader = tqdm(Combine_Trainloader, "Train epoch %d" % epoch)
            outputs_epoch_local = []
            labels_epoch_local = []
            for i,(imge,label) in enumerate(train_dataloader):
                img = torch.cat([imge[2], imge[2]], dim=0)
                bsz = label.shape[0]
                img=img.to('cuda')
                label=label.to('cuda')

                features_local = model_local(img.to('cuda'))
                label_loacl = label.clone()
                label_loacl[label == 5] = 0  # Normal is 0
                label_loacl[label != 5] = 1
                f1_local, f2_local = torch.split(features_local, [bsz, bsz], dim=0)
                outputs_local = Classifier_local(f1_local)
                if epoch < a:
                    Contrast_features_local = head_local(features_local)
                    f1_local, f2_local = torch.split(Contrast_features_local, [bsz, bsz], dim=0)
                    Contrast_features_local = torch.cat([f1_local.unsqueeze(1), f2_local.unsqueeze(1)], dim=1)
                    loss_con_local = criterion(Contrast_features_local, label_loacl)
                    loss_local =loss_fn(outputs_local,label_loacl)
                    loss = loss_local+loss_con_local
                else:
                    loss =loss_fn(outputs_local,label_loacl)

                _, pred_local = torch.max(outputs_local.cpu().data, 1)
                outputs_epoch_local.extend([i for i in pred_local.numpy()])
                labels_epoch_local.extend([i for i in label_loacl.cpu().numpy()])
                batch_loss.append(loss.cpu().detach().numpy())

                if epoch < a:
                    optimizer_local.zero_grad()
                    optimizer_head_local.zero_grad()
                    optimizer_cls_local.zero_grad()
                else:
                    optimizer_local.zero_grad()
                    optimizer_cls_local.zero_grad()
                loss.backward()

                if epoch < a:
                    optimizer_local.step()
                    optimizer_head_local.step()
                    optimizer_cls_local.step()
                else:
                    optimizer_local.step()
                    optimizer_cls_local.step()

                train_dataloader.postfix = 'acc_local %0.2f%%' % (
                    metrics.accuracy_score(labels_epoch_local, outputs_epoch_local)*100)

            with torch.no_grad():
                model_local = model_local.eval()
                Classifier_local = Classifier_local.eval()
                head_local = head_local.eval()
                batch_loss = []
                outputs_epoch_local = []
                labels_epoch_local = []
                # outputs_epoch_normal = []
                test_dataloader = tqdm(Combine_Testloader, "Test")
                for i,(img,label) in enumerate(test_dataloader):
                    img = img.to('cuda')
                    label = label.to('cuda')
                    label_loacl = label.clone()
                    label_loacl[label == 5] = 0  # Normal is 0
                    label_loacl[label != 5] = 1
                    features_local = model_local(img)
                    outputs_local = Classifier_local(features_local)
                    if count_float==0: # Calculate time complexity
                        count_float=1
                        from thop import profile
                        flops_3, params_3 = profile(model_local, (img,))
                        flops_4, params_4 = profile(Classifier_local, (features_local,))
                        flops =  flops_3 + flops_4
                        params = params_3 + params_4
                        print('flops: ', flops, 'params: ', params)

                    _, pred_local = torch.max(outputs_local.cpu().data, 1)


                    outputs_epoch_local.extend([i for i in pred_local.numpy()])
                    labels_epoch_local.extend([i for i in label_loacl.cpu().numpy()])

                p_local,recall_local, F1_local,  support_micro_local = metrics.precision_recall_fscore_support( labels_epoch_local,outputs_epoch_local)
                F1_local = 2 * p_local.mean() * recall_local.mean() / (p_local.mean() + recall_local.mean())
                g_mean_local = np.sqrt(p_local.mean() * recall_local.mean())


                print(
                    '----------------------------------------------------------------epoch %d---------------------------------------------------------' % epoch)
                print("Metrics")
                print('acc:            ','%0.2f' % (
                metrics.accuracy_score(outputs_epoch_local,labels_epoch_local)*100))
                print('mean_F1:        ', '%0.2f' % (F1_local * 100))
                print('mean_precision: ', '%0.2f' % (p_local.mean() * 100))
                print('mean_Recall:    ', '%0.2f' % (recall_local.mean() * 100))
                print('g_mean:         ', '%0.2f' % (g_mean_local * 100))
                if F1_local.mean() > f1_best:
                    epoch_best = epoch
                    f1_best = F1_local
                    p_best = p_local.mean()
                    recall_best = recall_local.mean()
                    g_mean_best = g_mean_local

                    # Save best confusion matrix data
                    best_cm = confusion_matrix(labels_epoch_local, outputs_epoch_local)

                batch_loss.append(loss.cpu().detach().numpy())
                test_dataloader.postfix = 'loss %0.2f'
    except KeyboardInterrupt:
        print('flops: ', flops, 'params: ', params)
        print(
            '----------------------------------------------------------------best epoch %d---------------------------------------------------------' %epoch_best)
        print("Metrics           normal")
        print('mean_F1:        ', '%0.2f' %  f1_best)
        print('mean_precision: ', '%0.2f' % p_best)
        print('mean_Recall:    ', '%0.2f' % recall_best)
        print('g_mean:         ', '%0.2f' % g_mean_best)
    print('flops: ', flops, 'params: ', params)
    print(
        '----------------------------------------------------------------best epoch %d---------------------------------------------------------' % epoch_best)
    print("Metrics")
    print('mean_F1:        ', '%0.2f' % f1_best)
    print('mean_precision: ', '%0.2f' % p_best)
    print('mean_Recall:    ', '%0.2f' % recall_best)
    print('g_mean:         ', '%0.2f' % g_mean_best)

    # Show the best confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=best_cm, display_labels=['Normal', 'Defect'])
    fig, ax = plt.subplots(figsize=(6, 6), facecolor='#212121')
    ax.set_facecolor('#212121')
    disp.plot(ax=ax, cmap=plt.cm.Blues, colorbar=False)
    plt.title(f"Best Confusion Matrix", color='white')
    plt.xticks(color='white')
    plt.yticks(color='white')
    plt.xlabel("Predicted label", color='white')
    plt.ylabel("True label", color='white')
    plt.grid(False)
    plt.show()

if __name__ == '__main__':
    main()
