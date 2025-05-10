from torch.utils.data import Dataset
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
from torchvision.datasets import ImageFolder
from torchvision.datasets import folder
from sklearn.model_selection import train_test_split
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform
    def __call__(self, x):
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        normalize = transforms.Normalize(mean=mean, std=std)
        x_local =transforms.Compose([transforms.Resize((254, 87)),
                transforms.ToTensor(),
                normalize])(x)
        return [self.transform(x), self.transform(x),x_local]

class LED_Folder():
    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = folder.default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        dataset=ImageFolder(root, transform=transform,
                                         target_transform=target_transform,
                                         loader=loader)
        dataset_test=ImageFolder(root+"Test", transform=transform,
                                         target_transform=target_transform,
                                         loader=loader)
        self.imgs_train,  self.targets_train = dataset.imgs, dataset.targets
        self.imgs_test, self.targets_test = dataset_test.imgs, dataset_test.targets
        self.classes_to_idx = dataset.class_to_idx
        self.classes = dataset.classes
        if True:
            self.classes_to_idx_test = dataset_test.class_to_idx
            self.classes_test = dataset_test.classes
            self.targets=dataset.targets
        # self.imgs_train, self.imgs_test, self.targets_train, self.targets_test = train_test_split(
        #     dataset.imgs, dataset.targets, test_size=0.95, random_state=63
        # )
class LED_CombineDataset():
    def __init__(self,LED_7020,LED_Q60B,train_transform=None,val_transform=None):
        self.classes_to_idx = LED_7020.classes_to_idx
        self.LED_7020 = LED_7020
        self.LED_Q60B = LED_Q60B
        for k in LED_Q60B.classes_to_idx:
            if k not in self.classes_to_idx:
                self.classes_to_idx[k] = len(self.classes_to_idx)
        self.id_to_classes = {v: k for k, v in self.classes_to_idx.items()}
        self.classes = [k for k in self.classes_to_idx]
        self.label_tran(self.LED_Q60B)
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        normalize = transforms.Normalize(mean=mean, std=std)
        if not train_transform:
            train_transform = transforms.Compose([
                transforms.Resize((254, 87)),
                # transforms.RandomResizedCrop(size=(254, 87), scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                # transforms.RandomApply([
                #     transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                # ], p=0.8),
                # transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                normalize,
            ])
            train_transform = TwoCropTransform(train_transform)
        if not val_transform:
            val_transform = transforms.Compose([
                transforms.Resize((254, 87)),
                transforms.ToTensor(),
                normalize,
            ])
        self.CombineTrain = to_Dataset(self.LED_7020.imgs_train+self.LED_Q60B.imgs_train,train_transform)
        self.CombineTest = to_Dataset(self.LED_7020.imgs_test+self.LED_Q60B.imgs_test,val_transform)
        self.LED_7020.imgs_train = to_Dataset(self.LED_7020.imgs_train,train_transform)
        self.LED_7020.imgs_test = to_Dataset(self.LED_7020.imgs_test,val_transform)
        self.LED_Q60B.imgs_train = to_Dataset(self.LED_Q60B.imgs_train,train_transform)
        self.LED_Q60B.imgs_test = to_Dataset(self.LED_Q60B.imgs_test,val_transform)
        self.targets_train = self.LED_7020.targets_train+self.LED_Q60B.targets_train
        self.targets_test = self.LED_7020.targets_test+self.LED_Q60B.targets_test
    def label_tran(self,dataset):
        id_to_classes = {v: k for k, v in dataset.classes_to_idx.items()}
        dataset.imgs_train = [(i,self.classes_to_idx[id_to_classes[j]]) for i,j in dataset.imgs_train]
        dataset.targets_train = [self.classes_to_idx[id_to_classes[i]] for i in dataset.targets_train]
        dataset.imgs_test = [(i, self.classes_to_idx[id_to_classes[j]]) for i, j in dataset.imgs_test]
        dataset.targets_test = [self.classes_to_idx[id_to_classes[i]] for i in dataset.targets_test]
class to_Dataset(Dataset):
    def __init__(self,samples,transforms=None,target_transform=None):
        super(to_Dataset, self).__init__()
        self.samples = samples
        self.transform = transforms
        self.target_transform = target_transform
        self.loader = folder.default_loader
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target
    def __len__(self):
        return len(self.samples)

class LED():
    def __init__(self,train_transform=None,val_transform=None):
        LED_7020 = LED_Folder('data/7020')
        LED_Q60B = LED_Folder('data/Q60B')
        self.combinedataset = LED_CombineDataset(LED_7020,LED_Q60B,train_transform,val_transform)
        self.classes = self.combinedataset.classes
        self.id_to_classes = self.combinedataset.id_to_classes
        self.Combine_Train = self.combinedataset.CombineTrain
        self.Combine_Test = self.combinedataset.CombineTest
        self.LED_7020_Test = self.combinedataset.LED_7020.imgs_test
        self.LED_Q60B_Test = self.combinedataset.LED_Q60B.imgs_test
        self.LED_7020_Train = self.combinedataset.LED_7020.imgs_train
        self.LED_Q60B_Train = self.combinedataset.LED_Q60B.imgs_train

if __name__ == "__main__":
    led = LED()
    all_label = []
    classes = led.classes
    all_count = [0 for i in range(len(classes))]
    LED_7020_count = [0 for i in range(len(classes))]
    LED_Q60B_count = [0 for i in range(len(classes))]
    for (img,label) in led.Combine_Train:
        all_count[label] += 1
    for (img,label) in led.Combine_Test:
        all_count[label] += 1
    for i in led.combinedataset.LED_7020.targets_train:
        LED_7020_count[i] += 1
    for i in led.combinedataset.LED_7020.targets_test:
        LED_7020_count[i] += 1
    for i in led.combinedataset.LED_Q60B.targets_train:
        LED_Q60B_count[i] += 1
    for i in led.combinedataset.LED_Q60B.targets_test:
        LED_Q60B_count[i] += 1
    plt.rcParams['font.sans-serif'] = ['SimHei']
    fig, ax = plt.subplots()
    # ax=fig.add_subplot(3, 1, 1)
    classes = np.array(classes)
    all_count = np.array(all_count)
    sns.barplot(x=classes,y=all_count)
    plt.xticks(rotation=90)
    plt.title("all")
    # ax = fig.add_subplot(3, 1, 2)
    fig, ax = plt.subplots()
    LED_7020_count = np.array(LED_7020_count)
    sns.barplot(x=classes,y=LED_7020_count)
    plt.xticks(rotation=90)
    plt.title("LED_7020")
    # ax = fig.add_subplot(3, 1, 3)
    fig, ax = plt.subplots()
    LED_Q60B_count = np.array(LED_Q60B_count)
    sns.barplot(x=classes,y=LED_Q60B_count)
    plt.xticks(rotation=90)
    plt.title("LED_Q60B")
    plt.show()