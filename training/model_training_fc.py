import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from io import BytesIO
import json, time
import os
import copy
from PIL import Image

import torch
import torchvision
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader



##### DATA TRANSFORMS #####

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'val': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}



##### DATA LOADERS #####

class CustomDataset(Dataset):
    def __init__(self, csv_file, train_test, label_name,transform=None):
        df = pd.read_csv(csv_file, index_col=0)
        df['path'] = df['file'].apply(lambda x: os.path.join(os.getcwd(),'data',x))
        
        if train_test == 'train':
            df = df[df['train']==True]
        elif train_test == 'val':
            df = df[df['train']==False]
        
        class_names = sorted(list(df.roomType.unique()))
        self.paths = df.path.values
        self.labels = df[label_name].replace({x: class_names.index(x) for x in class_names}).values
        self.transform = transform
    def __getitem__(self, index):
        # we want to be index like dataset[index]
        # to get the index-th batch
        img = Image.open(self.paths[index]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[index]
    
    def __len__(self):
        # to retrieve the total samples by doing len(dataset)
        return len(self.paths)


labels = pd.read_csv('labels.csv',index_col=0)

image_datasets = {x: CustomDataset('labels.csv', train_test=x, label_name='roomType',transform=data_transforms[x]) for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=4) for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = sorted(list(labels.roomType.unique()))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


##### TRAIN FUNCTION #####

def train(class_names, epochs, dataloaders):

    start_time = time.time()
    train_losses = []
    val_losses = []

    model = models.resnet152(pretrained=True) #resnet152
    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512), 
        nn.ReLU(), 
        nn.Dropout(0.4),
        nn.Linear(512, 128),
        nn.ReLU(), 
        nn.Dropout(0.4),
        nn.Linear(128, len(class_names)), 
        nn.LogSoftmax(dim=1))
   
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    #optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 8 epochs
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_model = model.state_dict()
    best_acc = 0.0
    best_loss = 1000
    
    for epoch in range(epochs):
        
        print(f'Starting epoch {epoch+1}/{epochs}')
        #Train set
        
        model.train()

        train_loss = 0.0
        running_corrects = 0
        
        batch = 0
        
        for inputs, labels in dataloaders['train']:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            if batch%250==0:
                print(f'batch {batch}')
            
            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)


            loss_train = criterion(outputs, labels)
            loss_train.backward()
            optimizer.step()
            
            train_loss += loss_train.item()
            batch +=1
            
        train_losses.append(train_loss/dataset_sizes['train'])
        exp_lr_scheduler.step()

        #Val set

        model.eval()

        val_loss = 0.0
        val_corrects = 0

        with torch.no_grad():
            
            for inputs, labels in dataloaders['val']:
                
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss_val = criterion(outputs, labels)

                val_loss += loss_val.item() 
                val_corrects += torch.sum(preds == labels.data)
            
        val_losses.append(val_loss)

        val_loss =  val_loss / dataset_sizes['val']
        val_acc = val_corrects / dataset_sizes['val']
        
        print(f'Epoch : {epoch+1} \t Train loss: {train_loss/dataset_sizes["train"]} \t Validation loss: {val_loss} \t Acc: {val_acc}')

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model, "./models/best_loss_model.pth")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model, "./models/best_acc_model.pth")



    time_elapsed = time.time() - start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    #model.load_state_dict(best_model_wts)
    return model

epochs = 100
train(class_names, epochs, dataloaders)




