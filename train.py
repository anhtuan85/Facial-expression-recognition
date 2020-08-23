import torch
from torchvision import transforms
from vgg import VGG
from datasets import FER2013
import numpy as np
import torch.nn as nn
import copy
import time
from utils import save_checkpoint, set_lr, clip_gradient
import torch.optim as optim
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ap = argparse.ArgumentParser()
ap.add_argument("--dataset_root", default= "fer2013/fer2013.csv", help = "path to dataset")
ap.add_argument("--model_name", default= "VGG19",type= str, help = "name model")
ap.add_argument("--checkpoint", default= None, help = "path to the checkpoint")
ap.add_argument("--bs", default= 128, type= int, help= "batch size for training")
ap.add_argument("--num_workers", default= 4, type= int, help = "Number of workers")
ap.add_argument("--lr", "--learning-rate", default= 0.01, type= float, help= "Learning rate")
ap.add_argument("--epochs", default= 200, type= int, help = "number of epochs to train")
ap.add_argument("--grad_clip", default = True, help= "Gradient clip for large batch_size")
ap.add_argument("--lr_decay_start", default= 80, type= int, help= "epoch learning rate decay")
ap.add_argument("--lr_decay_every", default= 5, type= int, help = "#epochs lr decay every")
ap.add_argument("--lr_decay_rate", default= 0.9, type= float, help = "lr decay rate")
ap.add_argument("--adjust_optim", default = None, help = "Adjust optimizer for checkpoint model")
args = ap.parse_args()

def train_model(model, dataloaders, criterion, optimizer, start_epoch, num_epochs= args.epochs):
    '''
    Train model
    model: Model
    dataloaders: dataloader dict: {train: , val: }
    criterion: Loss function
    optimizer: Optimizer for training
    num_epochs: Number of epochs to train

    Out: Best model, val_acc_history
    '''
    since = time.time()
    val_acc_history = []
    lr = args.lr
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc= 0.0
    learning_rate_decay_start = args.lr_decay_start
    learning_rate_decay_every = args.lr_decay_every
    learning_rate_decay_rate = args.lr_decay_rate
    for epoch in range(start_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print("-"*10)
        if epoch > learning_rate_decay_start and learning_rate_decay_every > 0:
            frac = (epoch - learning_rate_decay_start) // learning_rate_decay_every
            decay_factor = learning_rate_decay_rate ** frac
            current_lr =  lr * decay_factor
            set_lr(optimizer, current_lr)
            print("Learning rate: ", current_lr)
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()
            
            running_loss= 0.0
            running_corrects = 0
            for inputs, labels in dataloaders[phase]:
                t = inputs.size(0)
                if phase == "val":
                    bs, ncrops, c, h, w = np.shape(inputs)
                    inputs = inputs.view(-1, c, h, w)    #(bs*n_crops, c, h, w)
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    if phase == "val":
                        outputs = outputs.view(bs, ncrops, -1).mean(1)
                    loss = criterion(outputs, labels)
                    
                    _, preds = torch.max(outputs, 1)
                    if phase == 'train':
                        loss.backward()
                        clip_gradient(optimizer, 0.1)
                        optimizer.step()
                running_loss += loss.item() * t
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / (dataloader_length[phase])
            epoch_acc = running_corrects.double() / (dataloader_length[phase])
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)
        save_checkpoint(epoch, best_model_wts, optimizer)
        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60,time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    
    model.load_state_dict(best_model_wts)
    return model, val_acc_history
    
crop_size = 44
transform_train = transforms.Compose([
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()])

transform_test = transforms.Compose([
        transforms.TenCrop(crop_size),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]))
        ])

if args.checkpoint is None:
    start_epoch = 0
    model = VGG(args.model_name)
    optimizer = optim.SGD(model.parameters(), lr= args.lr, momentum= 0.9, 
                          weight_decay= 5e-4)
else:
    checkpoint = torch.load(args.checkpoint)
    start_epoch = checkpoint['epoch'] + 1
    print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
    model = VGG(args.model_name)
    model.load_state_dict(checkpoint["model_weights"])
    optimizer = checkpoint["optimizer"]
    if args.adjust_optim is not None:
        print("Adjust optimizer....")
        optimizer = optim.SGD(model.parameters(), lr= args.lr, momentum= 0.9,
                              weight_decay= 5e-4)

data = FER2013(args.dataset_root, split= "TRAIN", transform= transform_train)
valid_data = FER2013(args.dataset_root, split= "PUBLIC_TEST", transform= transform_test)
train_loader = torch.utils.data.DataLoader(data, batch_size= args.bs, shuffle= True,
                                           num_workers= args.num_workers)
validation_loader = torch.utils.data.DataLoader(valid_data, batch_size= args.bs, 
                                                num_workers= args.num_workers)
dataloader_dict = {"train": train_loader, "val": validation_loader}
dataloader_length = {"train": len(train_loader.dataset), "val": len(validation_loader.dataset)}
criterion= nn.CrossEntropyLoss()
model = model.to(device)

model, acc_val_history = train_model(model, dataloader_dict, criterion, optimizer,
                                     start_epoch, num_epochs= args.epochs)