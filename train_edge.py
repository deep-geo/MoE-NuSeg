# Only train the model to segment edges

# Standard Libraries
import os
import sys
import time
import random
import argparse
import logging
import copy
from datetime import datetime
from os.path import join as pjoin

# Numerical and Scientific Libraries
import numpy as np
import math
from scipy import ndimage

# PyTorch Core and Utilities
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import CosineAnnealingLR

# Visualization
import matplotlib.pyplot as plt

# Evaluation and Metrics
from miseval import evaluate

# Model Utilities and Libraries
from timm.models.layers import DropPath
from einops import rearrange

# Custom Modules
from utils import *
from models.transnuseg_MoE_p1 import TransNuSeg as MoE_p1

# Logging and Tracking
import wandb
import warnings

warnings.filterwarnings("ignore", category=UserWarning, message="torch.meshgrid") # Suppress the specific UserWarning related to torch.meshgrid
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

base_data_dir = "/root/autodl-tmp/data/"
HISTOLOGY_DATA_PATH = os.path.join(base_data_dir, 'histology') #HISTOLOGY_DATA_PATH  Containing two folders named data and test
RADIOLOGY_DATA_PATH = os.path.join(base_data_dir,'fluorescence') # Containing two folders named data and test
THYROID_DATA_PATH = os.path.join(base_data_dir,'thyroid') # Containing two folders named data and test
LIZARD_DATA_PATH = os.path.join(base_data_dir,'TNBC')

def main():
    '''
    model_type:  default: transnuseg
    alpha: ratio of the loss of nuclei mask loss, dafault=0.3
    beta: ratio of the loss of normal edge segmentation, dafault=0.35
    gamma: ratio of the loss of cluster edge segmentation, dafault=0.35
    random_seed: set the random seed for splitting dataset
    dataset: Radiology(grayscale) or Histology(rgb), default=Histology
    num_epoch: number of epoches
    lr: learning rate
    model_path: if used pretrained model, put the path to the pretrained model here
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type',required=True, default="transnuseg", help="declare the model type to use, currently only support input being transnuseg") 
    parser.add_argument("--random_seed",required=True, help="random seed")
    parser.add_argument("--batch_size",required=True, help="batch size")
    parser.add_argument("--dataset",required=True,default="Histology", help="Histology, Radiology")
    parser.add_argument("--num_epoch",required=True,help='number of epoches')
    parser.add_argument("--lr",required=True,help="learning rate")
    parser.add_argument("--model_path",default=None,help="the path to the pretrained model")

    args = parser.parse_args()
    
    model_type = args.model_type
    dataset = args.dataset

    batch_size=int(args.batch_size)
    random_seed = int(args.random_seed)
    num_epoch = int(args.num_epoch)
    base_lr = float(args.lr)
    
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True  # Ensure deterministic operations
    torch.backends.cudnn.benchmark = False    # Disable auto-tuning for reproducibility
    
    IMG_SIZE = 512
    wandb.init(project = dataset, name = "MoE p1 seed" + str(random_seed) )

    if dataset == "Radiology":
        data_path = RADIOLOGY_DATA_PATH
        from dataset_radio import MyDataset
    elif dataset == "Histology":
        data_path = HISTOLOGY_DATA_PATH
        from dataset import MyDataset
    elif dataset == "Thyroid":
        data_path = THYROID_DATA_PATH
    elif dataset == "Lizard":
        data_path = LIZARD_DATA_PATH
    else:
        print("Wrong Dataset type")
        return 0
    print(f"dataloader {dataset} imported")

    model = MoE_p1(img_size=IMG_SIZE)
    model.to(device)
    
    now = datetime.now()
    create_dir('./log')
    logging.basicConfig(filename='./log/log_{}_{}_{}.txt'.format(model_type,dataset,str(now)), level=logging.INFO,
                            format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(f"Random seed:{random_seed}, Batch size:{batch_size}, epoch num:{num_epoch}")
    
    total_data = MyDataset(dir_path=data_path)
    train_set_size = int(len(total_data) * 0.8)
    val_set_size = int(len(total_data) * 0.1)
    test_set_size = len(total_data) - train_set_size - val_set_size
    logging.info(f"train size {train_set_size}, val size {val_set_size}, test size {test_set_size}")
    
    train_set, val_set, test_set = random_split(total_data, [train_set_size, val_set_size, test_set_size],generator=torch.Generator().manual_seed(random_seed))

    trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    valloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=True)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=True)
 
    dataloaders = {"train":trainloader,"val":valloader, "test":testloader}
    dataset_sizes = {"train":len(trainloader), "val":len(valloader), "test":len(testloader)}
    logging.info(f"# of batchs: train: {dataset_sizes['train']}, val: {dataset_sizes['val']}, test: {dataset_sizes['test']}")
    
    train_loss = []
    val_loss = []
    test_loss = []
    
    num_classes = 2
    
    ce_loss1 = CrossEntropyLoss()

    dice_loss1 = DiceLoss(num_classes)
    
    HDLoss_nor = HausdorffLoss()
    bf1_nor = BoundaryF1Score(tolerance=1)
    dice_nor = DiceCoefficientBoundary(tolerance=1)
    pr_nor = ContourPrecisionRecall(tolerance=1)
    
    softmax = torch.nn.Softmax(dim=1)

    optimizer = optim.Adam(model.parameters(), lr=base_lr)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=500, eta_min=1e-8) #2000

    best_loss = 10000
    best_epoch = 0
    
    for epoch in range(num_epoch):
        # early stop, if the loss does not decrease for 50 epochs
        if epoch > best_epoch + 50:
            break
        for phase in ['train','val']:
            running_loss = 0
            s = time.time()  # start time for this epoch
            
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   

            for i, d in enumerate(dataloaders[phase]):

                img, _, _,normal_edge_mask,_ = d
                
                img = img.float().to(device)   
               
                normal_edge_mask = normal_edge_mask.to(device)
           
                _, output1, _ = model(img) 

                ce_loss_nor = ce_loss1(output1, normal_edge_mask.long())
                dice_loss_nor = dice_loss1(output1, normal_edge_mask.float(), softmax=True)
          
                loss  = 0.4*ce_loss_nor + 0.6*dice_loss_nor

                if phase == 'val':
                    wandb.log({
                        "CE Loss Edge": ce_loss_nor,
                        "Dice Loss Edge": dice_loss_nor,
                    })

                    BF1_nor = bf1_nor(output1, normal_edge_mask.long())
                    BDSC_nor = dice_nor(output1, normal_edge_mask.long())
                    BPrec_nor, BRecall_nor = pr_nor(output1, normal_edge_mask.long())
                    HDF_loss_nor = HDLoss_nor(output1, normal_edge_mask)
                
                    wandb.log({
                        "HDF_loss_nor": HDF_loss_nor.item(),
                        "BF1_nor": BF1_nor.item(),
                        "BDSC_nor": BDSC_nor.item(),
                        "BPrec_nor": BPrec_nor.item(),
                        "BRecall_nor": BRecall_nor.item(),
                        "Total Loss":loss
                    })

                    if epoch%30 == 1:
                        log_predictions_to_wandb(img, output1, normal_edge_mask, output1, normal_edge_mask, output1, normal_edge_mask, epoch, prefix='comparison')

                running_loss+=loss.item()
    
                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    lr_scheduler.step()
                
            e = time.time()
            batch_loss = running_loss / dataset_sizes[phase]
           
            print(f"{phase} Epoch: {epoch+1}, loss: {batch_loss}, lr: {optimizer.param_groups[0]['lr']}, time: {e-s}")
           
            if phase == 'val':
                test_loss.append(batch_loss)
                wandb.log({"Seg Loss per batch": batch_loss})
               
                if batch_loss < best_loss:
                    best_loss = batch_loss
                    best_epoch = epoch+1
                    best_model_wts = copy.deepcopy(model.state_dict())
                    logging.info("Best val seg loss {} save at epoch {}".format(best_loss,epoch+1))
    
    create_dir('./saved')
    
    formatted_time = now.strftime("%m%d_%H%M")  # YYYYMMDD_HHMM format
    save_path = f"./saved/model_{dataset}_{model_type}_seed{random_seed}_epoch{best_epoch}_loss_{best_loss:.5f}_{formatted_time}.pt"
    torch.save(best_model_wts, save_path)
    logging.info(f'Model saved at: {save_path}')
 
    ########################## Testing ##############################  
    #test_main('p1',save_path)
    model.load_state_dict(best_model_wts)
    model.eval()

    dices_pred1, bf1s_pred1, bdscs_pred1, bprecs_pred1, brecalls_pred1 = [], [], [], [], []

    with torch.no_grad():
        for i, d in enumerate(testloader):
            img, _, _, nor_edge_mask, _ = d

            img = img.float()    
            img = img.to(device)
            
            _,pred1,_= model(img)          
    
            # **Secondary prediction (pred1) evaluation**
            # Metrics for normal edge segmentation (pred1)
            pred1_probs = torch.sigmoid(pred1)  # Sigmoid for binary prediction
            pred1_classes = (pred1_probs > 0.5).long()

            HDF_loss_nor = HDLoss_nor(pred1, nor_edge_mask.float())
            BF1_nor = bf1_nor(pred1, nor_edge_mask)
            BDSC_nor = dice_nor(pred1, nor_edge_mask)
            BPrec_nor, BRecall_nor = pr_nor(pred1, nor_edge_mask)

            dices_pred1.append(HDF_loss_nor.item())
            bf1s_pred1.append(BF1_nor.item())
            bdscs_pred1.append(BDSC_nor.item())
            bprecs_pred1.append(BPrec_nor.item())
            brecalls_pred1.append(BRecall_nor.item())
            
    
    average_HDF_nor = np.mean(dices_pred1)
    average_BF1_nor = np.mean(bf1s_pred1)* 100.0
    average_BDSC_nor = np.mean(bdscs_pred1)* 100.0
    average_BPrec_nor = np.mean(bprecs_pred1)* 100.0
    average_BRecall_nor = np.mean(brecalls_pred1)* 100.0
    
    wandb.log({ 
    "Result_Boundary_Normal/HDF_Loss": average_HDF_nor,
    "Result_Boundary_Normal/F1": average_BF1_nor,
    "Result_Boundary_Normal/DSC": average_BDSC_nor,
    "Result_Boundary_Normal/Precision": average_BPrec_nor,
    "Result_Boundary_Normal/Recall": average_BRecall_nor   
    })
    wandb.finish()

    
if __name__=='__main__':
    main()