# Swin_UNet
import copy
import logging
import math
from os.path import join as pjoin
import cv2
import torch
import torch.nn as nn
import numpy as np
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from torchvision import transforms
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import os
from torch.utils.data import Dataset,DataLoader,TensorDataset
import torch.nn.functional as F
#from torch.nn.modules.loss import CrossEntropyLoss

import torch.optim as optim
import torch.utils.data as data 
#import scipy.io as sio
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchsummary import summary

import matplotlib.pyplot as plt
import random
import time
import sys
from datetime import datetime
import argparse

from dataset import MyDataset
from utils import *

from models.vision_transformer import SwinUnet as ViT_seg
from models.swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys as Swin_UNet

from sklearn.metrics import f1_score, accuracy_score
from itertools import chain

import wandb
import warnings
from miseval import evaluate

warnings.filterwarnings("ignore", category=UserWarning, message="torch.meshgrid") # Suppress the specific UserWarning related to torch.meshgrid
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

base_data_dir = "/root/autodl-tmp/data"
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
    parser.add_argument("--alpha",required=True,default=0.3, help="coeffiecient of the weight of nuclei mask loss")
    parser.add_argument("--beta",required=True,default=0.35, help="coeffiecient of the weight of normal edge loss")
    parser.add_argument("--gamma",required=True,default=0.35, help="coeffiecient of the weight of cluster edge loss")
    parser.add_argument("--random_seed",required=True, help="random seed")
    parser.add_argument("--batch_size",required=True, help="batch size")
    parser.add_argument("--dataset",required=True,default="Histology", help="Histology, Radiology")
    parser.add_argument("--num_epoch",required=True,help='number of epoches')
    parser.add_argument("--lr",required=True,help="learning rate")
    parser.add_argument("--model_path",default=None,help="the path to the pretrained model")

    args = parser.parse_args()
    
    model_type = args.model_type
    dataset = args.dataset

    alpha = float(args.alpha)
    beta = float(args.beta)
    gamma = float(args.gamma)
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
    wandb.init(project = dataset, name = "Swin_UNet" + str(random_seed) )

    if dataset == "Radiology":
        channel = 1
        data_path = RADIOLOGY_DATA_PATH
    elif dataset == "Histology":
        channel = 3
        data_path = HISTOLOGY_DATA_PATH
    elif dataset == "Thyroid":
        channel = 3
        data_path = THYROID_DATA_PATH
    elif dataset == "Lizard":
        channel = 3
        data_path = LIZARD_DATA_PATH
    else:
        print("Wrong Dataset type")
        return 0
    
    if os.path.exists(data_path):
        print(f"data_path {data_path} exists")
    else:
        print(f"data_path {data_path} does not exists")

    ############ Parameters for Swin Unet#################
    patch_size = 4  
    in_chans = 3     # 1 for grayscale, 3 for RGB
    num_classes = 2  # Binary segmentation
    window_size = 16

    # Initialize the Swin U-Net model
    model = Swin_UNet(img_size=IMG_SIZE, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes, window_size=window_size)
              
    model.to(device)

    now = datetime.now()
    create_dir('./log')
    logging.basicConfig(filename='./log/log_{}_{}_{}.txt'.format(model_type,dataset,str(now)), level=logging.INFO,
                            format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info("Dataset:{}, Random Seed:{}, Batch size :{}, epoch num:{}, alph:{}, beta:{}, gamma:{}".format(dataset, random_seed, batch_size,num_epoch,alpha,beta,gamma))

    
    total_data = MyDataset(dir_path=data_path, seed=random_seed)
    train_set_size = int(len(total_data) * 0.7)
    val_set_size = int(len(total_data) * 0.15)
    test_set_size = len(total_data) - train_set_size - val_set_size
    logging.info(f"train size {train_set_size}, val size {val_set_size}, test size {test_set_size}")
    
    #train_set, test_set = data.random_split(total_data, [train_set_size, test_set_size],generator=torch.Generator().manual_seed(random_seed))
    train_set, val_set, test_set = data.random_split(total_data, [train_set_size, val_set_size, test_set_size],generator=torch.Generator().manual_seed(random_seed))

    trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    valloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=True)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=True)
 
    dataloaders = {"train":trainloader,"val":valloader, "test":testloader}
    dataset_sizes = {"train":len(trainloader), "val":len(valloader), "test":len(testloader)}
    logging.info(f"batch size train: {dataset_sizes['train']}, batch size val: {dataset_sizes['val']}, batch size test: {dataset_sizes['test']}")
        
    
    train_loss = []
    val_loss = []
    test_loss = []
    
    num_classes = 2
    
    ce_loss1 = CrossEntropyLoss()

    #ce_loss1 = BCE_Loss_logits()
    #ce_loss1 = FocalLoss()
    #ce_loss1 = SST_loss()
   
    dice_loss1 = DiceLoss(num_classes)

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=1e-4)
    
    best_loss = 10000
    best_epoch = 0
    
    for epoch in range(num_epoch):
        # early stop, if the loss does not decrease for 50 epochs
        if epoch > best_epoch + 80:
            break
        for phase in ['train','val']:
            running_loss = 0
            running_loss_seg = 0
            s = time.time()  # start time for this epoch
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   

            for i, d in enumerate(dataloaders[phase]):

                img, instance_seg_mask, semantic_seg_mask,normal_edge_mask,cluster_edge_mask = d
             
                img = img.float()    
                img = img.to(device)
               
                semantic_seg_mask = semantic_seg_mask.to(device).long()
                
                # Convert to one-hot encoding
                target_one_hot = F.one_hot(semantic_seg_mask.long(), num_classes=num_classes)  # Shape: [batch_size, height, width, num_classes]

                # Permute dimensions to match model output shape
                target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()  # Shape: [batch_size, num_classes, height, width]

                output1 = model(img) 

                ce_l = ce_loss1(output1, target_one_hot)
                dice_l = dice_loss1(output1, semantic_seg_mask.float(), softmax=True)
                
                loss_seg = 0.4 * ce_l + 0.6 * dice_l
                
                if phase == 'val':
                    wandb.log({"CE Loss Seg":ce_l, "Dice Loss Seg":dice_l, "Total Loss":loss_seg})

                running_loss_seg += loss_seg.item() ## Loss for nuclei segmantation
                if phase == 'train':
                    optimizer.zero_grad()
                    loss_seg.backward()
                    optimizer.step()
                
            e = time.time()
            
            batch_loss_seg = running_loss_seg / dataset_sizes[phase]

            #print(f"{phase} Epoch: {epoch+1}, loss: {epoch_loss}, lr: {optimizer.param_groups[0]['lr']}, time: {e-s}")
            print(f"{phase} Epoch: {epoch+1}, loss: {batch_loss_seg}, lr: {optimizer.param_groups[0]['lr']}, time: {e-s}")
           
            if phase == 'train':
                train_loss.append(batch_loss_seg)
                #wandb.log({"epoch time/train": e-s})
            else:
                test_loss.append(batch_loss_seg)
                wandb.log({"Seg Loss per batch": batch_loss_seg})

            if phase == 'val' and batch_loss_seg < best_loss:
                best_loss = batch_loss_seg
                best_epoch = epoch+1
                best_model_wts = copy.deepcopy(model.state_dict())
                logging.info("Best val loss {} save at epoch {}".format(best_loss,epoch+1))

    create_dir('./saved')    
    formatted_time = now.strftime("%m%d_%H%M")  # MMDD_HHMM format
    save_path = f"./saved/model_{dataset}_{model_type}_seed{random_seed}_epoch{best_epoch}_loss_{best_loss:.5f}_{formatted_time}.pt"
    torch.save(best_model_wts, save_path)
    logging.info(f'Model saved at: {save_path}')
########################## Testing ##############################
    model.load_state_dict(best_model_wts)
    model.eval()
    
    dices, f1s, ious, accuracies, precs, senss = [], [], [], [], [], []
    #dices_pred1, bf1s_pred1, bdscs_pred1, bprecs_pred1, brecalls_pred1, bIoU_pred1 = [], [], [], [], [], []

    with torch.no_grad():
        for i, d in enumerate(testloader):
            img, _, semantic_seg_mask,_,_ = d

            img = img.float()    
            img = img.to(device)
            
            preds = model(img)
            
            preds_softmax = torch.softmax(preds, dim=1)
            preds_classes = torch.argmax(preds_softmax, dim=1)
                       
            preds_flat = preds_classes.view(-1)
            labels_flat = semantic_seg_mask.view(-1)
            
            np_preds = preds_flat.cpu().numpy()
            np_labels = labels_flat.cpu().numpy()
            
            dice = evaluate(np_labels, np_preds, metric="DSC") 
            iou = evaluate(np_labels, np_preds, metric="IoU")  
            acc = evaluate(np_labels, np_preds, metric="ACC")  
            prec = evaluate(np_labels, np_preds, metric="PREC")  
            sens = evaluate(np_labels, np_preds, metric="SENS")  
            f1 = 2*prec*sens/(prec+sens)
            
            dices.append(dice)
            f1s.append(f1) 
            ious.append(iou)
            accuracies.append(acc)
            precs.append(prec)
            senss.append(sens)
            
            # **Secondary prediction (pred1) evaluation**
            # Metrics for normal edge segmentation (pred1)
#             pred1_probs = torch.sigmoid(pred1)  # Sigmoid for binary prediction
#             pred1_classes = (pred1_probs > 0.5).long()

#             HDF_loss_nor = HDLoss_nor(pred1_probs, normal_edge_mask.float())
#             BF1_nor = bf1_nor(pred1_classes, normal_edge_mask)
#             BDSC_nor = dice_nor(pred1_classes, normal_edge_mask)
#             BPrec_nor, BRecall_nor = pr_nor(pred1_classes, normal_edge_mask)
#             BIoU_nor = iou_nor(pred1_classes, normal_edge_mask)
            
#             dices_pred1.append(HDF_loss_nor.item())
#             bf1s_pred1.append(BF1_nor.item())
#             bdscs_pred1.append(BDSC_nor.item())
#             bprecs_pred1.append(BPrec_nor.item())
#             brecalls_pred1.append(BRecall_nor.item())
#             bIoU_pred1.append(BIoU_nor.item())           
            
    average_dice = np.mean(dices)* 100.0
    average_f1 = np.mean(f1)* 100.0
    average_iou = np.mean(ious)* 100.0
    average_accuracy = np.mean(accuracies)* 100.0
    average_prec = np.mean(precs)* 100.0
    average_sens = np.mean(sens)* 100.0
    
    # average_HDF_nor = np.mean(dices_pred1)
    # average_BF1_nor = np.mean(bf1s_pred1)* 100.0
    # average_BDSC_nor = np.mean(bdscs_pred1)* 100.0
    # average_BPrec_nor = np.mean(bprecs_pred1)* 100.0
    # average_BRecall_nor = np.mean(brecalls_pred1)* 100.0
    # average_BIoU_nor = np.mean(bIoU_pred1)* 100.0
    
    wandb.log({
    "Result/Dice": average_dice,
    "Result/F1": average_f1,
    "Result/IoU_JI": average_iou,
    "Result/Accuracy": average_accuracy,
    "Result/Sensitivity": average_sens,
    "Result/Precision": average_prec,
    # "Result_Boundary_Normal/HDF_Loss": average_HDF_nor,
    # "Result_Boundary_Normal/F1": average_BF1_nor,
    # "Result_Boundary_Normal/DSC": average_BDSC_nor,
    # "Result_Boundary_Normal/IoU": average_BIoU_nor,
    # "Result_Boundary_Normal/Precision": average_BPrec_nor,
    # "Result_Boundary_Normal/Recall": average_BRecall_nor,
    })
    wandb.finish()

    print(f"average_Dice: {average_dice}\n"
          f"average_F1: {average_f1}\n"
          f"average_IoU_JI: {average_iou}\n"
          f"average_accuracy: {average_accuracy}\n"
          f"average_Precision: {average_prec}\n"
          f"average_sensitivity: {average_sens}\n")
    
    parts = save_path.split('/')
    model_part = next((part for part in parts if part.startswith('model')), None) # extract the weight file name instead of whole path
    log_entry = f"{dataset}, 'p2', {random_seed}, {average_dice}, {average_iou}, {average_f1}, {average_accuracy}, {average_prec},{average_sens}, {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, {model_part}\n"

    # Append the log entry to the file
    with open("./log/results.txt", "a") as file:
        file.write(log_entry)  
           
if __name__=='__main__':
    main()

    
#     model.load_state_dict(best_model_wts)
#     model.eval()
    
#     dices, f1s, ious, accuracies, precs, senss = [], [], [], [], [], []

#     with torch.no_grad():
#         for i, d in enumerate(testloader):
#             img, _, semantic_seg_mask,_,_ = d

#             img = img.float()    
#             img = img.to(device)
            
#             preds = model(img)
            
            
#             preds_softmax = torch.softmax(preds, dim=1)
#             preds_classes = torch.argmax(preds_softmax, dim=1)
            
#             preds_flat = preds_classes.view(-1)
#             labels_flat = semantic_seg_mask.view(-1)
            
#             np_preds = preds_flat.cpu().numpy()
#             np_labels = labels_flat.cpu().numpy()
            
#             dice = evaluate(np_labels, np_preds, metric="DSC") 
#             iou = evaluate(np_labels, np_preds, metric="IoU")  
#             acc = evaluate(np_labels, np_preds, metric="ACC")  
#             prec = evaluate(np_labels, np_preds, metric="PREC")  
#             sens = evaluate(np_labels, np_preds, metric="SENS")  
#             f1 = 2*prec*sens/(prec+sens)
            
#             dices.append(dice)
#             f1s.append(f1) 
#             ious.append(iou)
#             accuracies.append(acc)
#             precs.append(prec)
#             senss.append(sens)
            
#     average_dice = np.mean(dices)
#     average_f1 = np.mean(f1)
#     average_iou = np.mean(ious)
#     average_accuracy = np.mean(accuracies)
#     average_prec = np.mean(precs)
#     average_sens = np.mean(sens)
    
#     print("Final metrics being logged...")
#     wandb.log({
#     "Test/Dice": average_dice * 100.0,
#     "Test/F1": average_f1 * 100.0,
#     "Test/IoU_JI": average_iou * 100.0,
#     "Test/Accuracy": average_accuracy * 100.0,
#     "Test/Sensitivity": average_sens * 100.0,
#     "Test/Precision": average_prec * 100.0
#     })
#     wandb.finish()
 
