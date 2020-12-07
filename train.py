
import sys
sys.path.append("./model/")
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from model.SSD300 import SSD300, MultiBoxLoss

from datasets import CrowdDataset
from utils import *
from model.metrics.metric import Metrics
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 2
batch_size = 8  # batch size
iterations = 145000  # number of iterations to train
workers = 4  # number of workers for loading data in the DataLoader
print_freq = 10  # print training status every __ batches
lr =  0.001 # learning rate
decay_lr_at = [96500, 120000]  # decay learning rate after these many iterations
decay_lr_to = 0.1  # decay learning rate to this fraction of the existing learning rate
momentum = 0.9  # momentum
weight_decay = 0.0005
cudnn.benchmark = True

def main():
    global start_epoch, label_map, epoch, checkpoint, decay_lr_at
    #Init model or load checkpoint
   
    start_epoch= 0
    model = SSD300(num_classes)
    biases = list()
    not_biases = list()
    for param_name, param in model.named_parameters():
            if param.requires_grad:
                if param_name.endswith(".bias"):
                    biases.append(param)
                else:
                    not_biases.append(param)
    optimizer = optim.SGD(params= [{'params': biases,"lr": 2* lr}, {"params": not_biases}],
                              lr = lr, momentum = momentum, weight_decay = weight_decay)
    #Move to default device
    model = model.to(device)
    criterion = MultiBoxLoss(model.default_boxes).to(device)

    root = './'
    train_dataset = CrowdDataset(root,'new_train.csv',transform=torchvision.transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size= batch_size, 
                                               shuffle= True,num_workers=0,collate_fn=combine)
    epochs = iterations // (len(train_dataset) // batch_size)
    decay_lr_at = [it // (len(train_dataset) // batch_size) for it in decay_lr_at]
    
    for epoch in range(start_epoch, epochs):
        if epoch in decay_lr_at:
            print("Decay learning rate...")
            adjust_lr(optimizer, decay_lr_to)
        train(train_loader = train_loader, model = model, criterion= criterion,
              optimizer = optimizer, epoch = epoch)
        save_checkpoint(epoch, model, optimizer)

def train(train_loader, model, criterion, optimizer, epoch):
    '''
        One epoch's training
    '''
    model.train()
    losses = Metrics()
    
    for i, (images, boxes, labels) in enumerate(train_loader):
        
        images = images.to(device)  # (batch_size (N), 3, 300, 300)

        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]

        #Foward pass
        locs_pred, cls_pred = model(images)
        #loss
        loss = criterion(locs_pred, cls_pred, boxes, labels)
        
        #Backward pass
        optimizer.zero_grad()
        loss.backward()
            
        optimizer.step()
        
        losses.update(loss.item(), images.size(0))
        
        # if i % print_freq == 0:
        print('Epoch: [{0}][{1}/{2}]\t' 'Loss {loss.val:.4f} ( Average Loss per epoch: {loss.avg:.4f})\t'.format(epoch, i, len(train_loader), loss=losses))
    del locs_pred, cls_pred, images, boxes, labels

if __name__ == '__main__':
    main()