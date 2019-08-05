import argparse
import utils
import model.crnn as crnn

import random
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import torch
import string
import numpy as np
import torch.backends.cudnn as cudnn
import torch.optim as optim
import random
import torchvision.models as models
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from torch_baidu_ctc import CTCLoss

from PIL import Image, ImageOps
# from invert import Invert

import warnings
import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    
#### argument parsing ####
parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--epoch', type=int, default=30, help='number of epochs to train for')
parser.add_argument('--dataPath', required=True, help='path to training dataset')
parser.add_argument('--savePath', required=True, help='path to save trained weights')
parser.add_argument('--preTrainedPath', type=str, default=None,
                    help='path to pre-trained weights (incremental learning)')
parser.add_argument('--seed', type=int, default=8888, help='reproduce experiement')
parser.add_argument('--worker', type=int, default=0, 
                    help='number of cores for data loading')
# parser.add_argument('--imgW', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-1)
parser.add_argument('--maxLength', type=int, default=9, 
                    help='maximum license plate character length in data')
opt = parser.parse_args()
print(opt)


#### set up constants and experiment settings ####
BATCH_SIZE = opt.batchSize
EPOCH = opt.epoch
PATH_TRAIN = opt.dataPath
PRE_TRAINED_PATH = opt.preTrainedPath
IMGH = 32

## Feature: Char Resizing ##
# 1. IMGW (const): Uncomment formula
# 2. LPDataSet __getitem__ (class method): Uncomment PIL resize
# 3. train_transformer (torchvision): Uncomment transform.Resize

# Note:crnn output length = img_width / 4 + 1
# Assumption: 4 cuts per character
# Calculation: 4 * maxCharLen = img_width / 4 + 1
#             => img_width = 16 * maxCharLen - 4 
# IMGW = opt.maxLength * 16 - 4
IMGW = 100

cudnn.benchmark = True

random.seed(opt.seed)
np.random.seed(opt.seed)
torch.manual_seed(opt.seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if not os.path.exists(opt.savePath):
    os.makedirs(opt.savePath)
    
#### data preparation & loading ####
train_transformer = transforms.Compose([
    transforms.Grayscale(),  
#     transforms.RandomApply([Invert()], p=0.3), # taxi license plate
    transforms.Resize((IMGH,IMGW)),
    transforms.ToTensor()])  # transform it into a torch tensor

class LPDataset(Dataset):
    """
    A standard PyTorch definition of Dataset which defines the functions 
    __len__ and __getitem__.
    """
    def __init__(self, path, cv_idx, transform):
        """
        Store the filenames of the jpgs to use. 
        Specifies transforms to apply on images.

        Args:
            path: (string) directory containing the dataset
            cv_idx: cross validation indices (training / validation sets)
            transform: (torchvision.transforms) transformation to apply on image
        """
        self.path = path
        temp = os.listdir(path)
        self.dirs = [temp[i] for i in cv_idx]
    
        filenames = [os.path.splitext(directory)[0] for directory in self.dirs]
        
        self.labels = [file_name.split('_')[-1] for file_name in filenames]
        self.transform = transform

    def __len__(self):
        # return size of dataset
        return len(self.dirs)

    def __getitem__(self, idx):
        """
        Fetch index idx image and labels from dataset. 
        Perform transforms on image.

        Args:
            idx: (int) index in [0, 1, ..., size_of_dataset-1]

        Returns:
            image: (Tensor) transformed image
            label: corresponding label of image
        """
        image = Image.open(os.path.join(self.path, self.dirs[idx]))  # PIL image
#         image = image.resize((16 * len(self.labels[idx]) - 4, IMGH))

#         if image.size[0] < IMGW:
#             image = ImageOps.expand(image, (0, 0, (IMGW - image.size[0]), 0), 
#                                   fill='black')
#         elif image.size[0] > IMGW:
#             raise Exception("Invalid --maxLength")
        
        image = self.transform(image)
        return image, self.labels[idx]
    
n = range(len(os.listdir(PATH_TRAIN)))
train_idx, val_idx = train_test_split(n, train_size=0.8, test_size=0.2, random_state=opt.seed)

# train data
print("Checkpoint: Loading data")
train_loader = DataLoader(LPDataset(PATH_TRAIN, train_idx, train_transformer), 
                          batch_size=BATCH_SIZE, num_workers = opt.worker, 
                          shuffle=True, pin_memory=True)
print("Checkpoint: Data loaded")

# validation data
val_set = LPDataset(PATH_TRAIN, val_idx, train_transformer)


#### setup crnn model hyperparameters ####
classes = string.ascii_uppercase+string.digits
nclass = len(classes) + 1
nc = 1 # number of channels 1=grayscale

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# CRNN(imgH, nc, nclass, num_hidden(LSTM))
crnn = crnn.CRNN(IMGH, nc, nclass, 256).to(device)
print("Checkpoint: Model loaded")

if torch.cuda.device_count() > 1:
    print("Running parallel on", torch.cuda.device_count(), "GPUs..")
    crnn = torch.nn.DataParallel(crnn, range(1))

if PRE_TRAINED_PATH is not None:    
    crnn.load_state_dict(torch.load(PRE_TRAINED_PATH, map_location=device))
else:
    crnn.apply(weights_init)

    
#### image and text (convert to tensor) ####
image = torch.FloatTensor(BATCH_SIZE, 1, IMGH, IMGH).to(device)
text = torch.IntTensor(BATCH_SIZE * 5)
length = torch.IntTensor(BATCH_SIZE)

image = Variable(image)
test = Variable(text)
length = Variable(length)

#### decoder, loss function, batch loss ####
converter = utils.strLabelConverter(classes)
loss_avg = utils.averager()
criterion = CTCLoss().to(device)

#### learning rate, lr scheduler, lr optimiser ####
LR = opt.lr
optimizer = optim.Adadelta(crnn.parameters(), lr=LR)
# T_max = len(train_loader) * EPOCH
# lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=LR/10)


def trainBatch(net, criterion, optimizer):
    data = train_iter.next()
    cpu_images, cpu_texts = data
    batch_size = cpu_images.size(0)
    utils.loadData(image, cpu_images)
    t, l = converter.encode(cpu_texts)
    utils.loadData(text, t)
    utils.loadData(length, l)
    
    preds = crnn(image)
    preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
    
    cost = criterion(preds, text, preds_size, length) / batch_size
    
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
#     lr_scheduler.step()
    
    return cost

def validation(net, dataset, criterion, max_iter=100):
    """
    To compute the validation loss from a given validation dataset
    
    net: neural network architecture
    dataset: validation set
    criterion: loss function
    max_iter: maximum number of mini_batches
    
    return: validation loss
    """
    
    for p in crnn.parameters():
        p.requires_grad = False
    
    # configure the mode: evaluation (model.train() & model.eval() behaves differently)
    net.eval()
    data_loader = DataLoader(
        dataset, shuffle=True, batch_size=BATCH_SIZE)
    val_iter = iter(data_loader)

    i = 0
    loss_avg = utils.averager()
    
    # measure accuracy
    y_true = []
    y_pred = []
    
    max_iter = min(max_iter, len(data_loader))
    for i in range(max_iter):
        data = val_iter.next()
        i += 1
        cpu_images, cpu_texts = data
        batch_size = cpu_images.size(0)
        utils.loadData(image, cpu_images)
        t, l = converter.encode(cpu_texts)
        utils.loadData(text, t)
        utils.loadData(length, l)

        preds = crnn(image)
        preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
        cost = criterion(preds, text, preds_size, length) / batch_size
        loss_avg.add(cost)
        
        ## make predictions & check accuracy
        _, preds = preds.max(2, keepdim=True)
        preds = preds.squeeze(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
        
        for pred, target in zip(sim_preds, cpu_texts):
            y_true.append(target)
            y_pred.append(pred.upper())
        
#         preds_np = preds.data.cpu().numpy().squeeze()
#         preds_np = np.hstack([preds_np[:, 1:], preds_np[:, [0]]])
#         y_pred.append(utils.ctcBestPath(preds_sm, classes))
#         y_true.append(cpu_texts)
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Sample improvement: gTruth:", y_true[0], "pred:", y_pred[0])
    
    return loss_avg.val()


#### training begins ####

# 25000 * 0.8 (# of data) // 64 (bs) ~= 310 (iterations) 
if __name__ == "__main__":
    save_iter = len(os.listdir(PATH_TRAIN)) * 0.8 // BATCH_SIZE
    PRINT_ITER = save_iter
    
    print("Checkpoint: Start training")
    for epoch in range(EPOCH):
        train_iter = iter(train_loader)
        i = 0

        while i < len(train_loader):
            for p in crnn.parameters():
                p.requires_grad = True
            crnn.train()

            cost = trainBatch(crnn, criterion, optimizer)
            loss_avg.add(cost)

            i += 1
            if i % PRINT_ITER == 0:
                # print training loss and validation loss
                print('[%d/%d][%d/%d] Train Loss: %f  Validation Loss: %f' %
                      (epoch + 1, EPOCH, i + 1, len(train_loader), loss_avg.val(), 
                       validation(crnn, val_set, criterion)))
                loss_avg.reset()

            if i % save_iter == 0 :
                
                try:
                    state_dict = crnn.module.state_dict()
                except AttributeError:
                    state_dict = crnn.state_dict()
                    
                torch.save(state_dict,
                           os.path.join(opt.savePath,
                                        'netCRNN_{}_{}.pth'.format(epoch + 1, i + 1)))