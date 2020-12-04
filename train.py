
import sys
sys.path.append("./model/")
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from model.SSD300 import SSD300
from model.vgg import VGG16BaseNet, AuxiliaryNet, PredictionNet
from model.mutibox_loss import MultiBoxLoss
from datasets import CrowdDataset
from utils import *
from model.metrics.metric import Metrics
import argparse
import torchvision 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''

ap = argparse.ArgumentParser()
ap.add_argument("--dataset_root", default= "./JSONdata/", help= "Dataroot directory path")
ap.add_argument("--batch_size", default= 8, type= int, help= "Batch size for training")
ap.add_argument("--num_workers", default= 4, type= int, help = "Number of workers")
ap.add_argument("--lr", "--learning-rate", default= 1e-3, type= float, help= "Learning rate")
ap.add_argument("--cuda", default= True, type= str2bool, help = "Use CUDA to train model")
ap.add_argument("--momentum", default= 0.9, type= float, help = "Momentum value for optim")
ap.add_argument("--weight_decay", default= 5e-4, type= float, help = "Weight decay for SGD")
ap.add_argument("--checkpoint", default = None, help = "path to model checkpoint")
ap.add_argument("--iterations", default= 145000, type= int, help = "number of iterations to train")
ap.add_argument("--grad_clip", default = None, help= "Gradient clip for large batch_size")
ap.add_argument("--adjust_optim", default = None, help = "Adjust optimizer for checkpoint model")
args = ap.parse_args()
'''
#Data parameters

num_classes = 2
batch_size = 8  # batch size
iterations = 145000  # number of iterations to train
workers = 4  # number of workers for loading data in the DataLoader
print_freq = 100  # print training status every __ batches
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
    
    '''
    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1   
        print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
        if args.adjust_optim is not None:
            print("Adjust optimizer....")
            print(args.lr)
            biases = list()
            not_biases = list()
            for param_name, param in model.named_parameters():
                if param.requires_grad:
                    if param_name.endswith(".bias"):
                        biases.append(param)
                    else:
                        not_biases.append(param)
            optimizer = optim.SGD(params= [{'params': biases,"lr": 2* lr}, {"params": not_biases}],lr = lr, momentum = momentum, weight_decay = weight_decay)
    '''
    #Move to default device
    model = model.to(device)
    criterion = MultiBoxLoss(model.default_boxes).to(device)
    # root='drive/MyDrive/CSE 5523 Project/'
    # dataset=CrowdDataset(root,'train2.csv',transform=torchvision.transforms.ToTensor())
    # torch.manual_seed(1)
    # indices = torch.randperm(len(dataset)).tolist()
    # dataset_train = dataset
    root = './'
    #data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=4, shuffle=False,collate_fn=lambda x:list(zip(*x)))
    train_dataset = CrowdDataset(root,'train.csv',transform=torchvision.transforms.ToTensor())
    test_dataset = CrowdDataset(root,'test.csv',transform=torchvision.transforms.ToTensor())
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size= batch_size, 
    #                                            shuffle= True, collate_fn= combine,
    #                                            num_workers = workers, pin_memory = True)
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size= batch_size, 
    #                                            shuffle= True, num_workers = workers, pin_memory = True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size= batch_size, 
                                               shuffle= True,num_workers=0,collate_fn=combine)
    epochs = iterations // (len(train_dataset) // batch_size)
    decay_lr_at = [it // (len(train_dataset) // batch_size) for it in decay_lr_at]
    
    for epoch in range(start_epoch, epochs):
        if epoch in decay_lr_at:
            print("Decay learning rate...")
            adjust_lr(optimizer, decay_lr_to)
        #One 's training
        train(train_loader = train_loader, model = model, criterion= criterion,
              optimizer = optimizer, epoch = epoch)
        
        #Save
        save_checkpoint(epoch, model, optimizer)
def train(train_loader, model, criterion, optimizer, epoch):
    '''
        One epoch's training
    '''
    model.train()
    losses = Metrics()
    
    for i, (images, boxes, labels) in enumerate(train_loader):
        
        images = images.to(device)  # (batch_size (N), 3, 300, 300)
        # images = torch.tensor(images).to(device)
        # print(images.shape)
        # [image.to(device) for image in images]
        # images = list(image.to(device) for image in images)
        # # images = np.array(image.to(device) for image in images).convert('RGB')
        # images = torch.FloatTensor(images)
        # images = torch.FloatTensor(image.to(device) for image in images)

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