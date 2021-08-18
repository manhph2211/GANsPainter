import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
import glob
import random
import os
from torch.utils.data import Dataset
from PIL import Image
from utils import plot, read_json
torch.manual_seed(0)


class ImageDataset(Dataset):
    def __init__(self, data_json, transform=None, mode='full'):
        self.transform = transform
        self.data = read_json(data_json)[mode]
        self.files_A = sorted(self.data[0])
        self.files_B = sorted(self.data[1])
        if len(self.files_A) > len(self.files_B):
            self.files_A, self.files_B = self.files_B, self.files_A
        self.new_perm()
        self.mode = mode
        assert len(self.files_A) > 0, "Make sure you downloaded the horse2zebra images!"

    def new_perm(self):
        self.randperm = torch.randperm(len(self.files_B))[:len(self.files_A)]

    def __getitem__(self, index):

        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))
        item_B = self.transform(Image.open(self.files_B[self.randperm[index]]))
        if index == len(self) - 1:
            self.new_perm()

        return (item_A - 0.5) * 2, (item_B - 0.5) * 2

    def __len__(self):
        return min(len(self.files_A), len(self.files_B))


if __name__ == '__main__':

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    dataset = ImageDataset('../data/data.json',transform)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        num_workers=2
    )

    monet, photo = iter(dataloader).next()
    monet, photo = monet[0]/ 2 + 0.5 , photo[0]/2 + 0.5 
    monet, photo = monet.permute(1,2,0) , photo.permute(1,2,0) 
    
    plot(monet,photo)



    
