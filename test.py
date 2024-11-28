# test for fix the metrics calculation of test phase

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
from dataset_radio import MyDataset
from models.transnuseg_MoE_p2_prior import TransNuSeg as MoE_p2

# Logging and Tracking
import wandb
import warnings

warnings.filterwarnings("ignore", category=UserWarning, message="torch.meshgrid") # Suppress the specific UserWarning related to torch.meshgrid

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

base_data_dir = "/root/autodl-tmp/data"
HISTOLOGY_DATA_PATH = os.path.join(base_data_dir, 'histology') # Containing two folders named data and test
RADIOLOGY_DATA_PATH = os.path.join(base_data_dir,'fluorescence') # Containing two folders named data and test
THYROID_DATA_PATH = os.path.join(base_data_dir,'thyroid') # Containing two folders named data and test
LIZARD_DATA_PATH = os.path.join(base_data_dir,'lizard') #Containing two folders named data and test
checkpoint_path = "/root/autodl-tmp/MoE-NuSeg/saved/model_Histology_MoE_p2_seed42_epoch84_loss_0.13084_1125_1249.pt"


def main():

    dataset = "Histology"

    alpha = 0.3
    beta = 0.35
    gamma = 0.35
    batch_size = 2
    random_seed = 42
    
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True  # Ensure deterministic operations
    torch.backends.cudnn.benchmark = False    # Disable auto-tuning for reproducibility

    IMG_SIZE = 512
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
 
    wandb.init(project = dataset, name = "MoE p2 Test seed" + str(random_seed))
    model = MoE_p2(img_size=IMG_SIZE)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint, strict=False)
    print(f"model {checkpoint_path} loaded")
    model = model.to(device)

    now = datetime.now()

    print(f"Seed:{random_seed}, Batch size:{batch_size}")
    
    total_data = MyDataset(dir_path=data_path, seed=random_seed)
    train_set_size = int(len(total_data) * 0.8)
    val_set_size = int(len(total_data) * 0.1)
    test_set_size = len(total_data) - train_set_size - val_set_size
    print(f"train size {train_set_size}, val size {val_set_size}, test size {test_set_size}")
    
    train_set, val_set, test_set = random_split(total_data, [train_set_size, val_set_size, test_set_size],generator=torch.Generator().manual_seed(random_seed))

    trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    valloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=True)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=True)
    
    dataloaders = {"train":trainloader,"val":valloader, "test":testloader}
    dataset_sizes = {"train":len(trainloader), "val":len(valloader), "test":len(testloader)}
    logging.info(f"batch size train: {dataset_sizes['train']}, batch size val: {dataset_sizes['val']}, batch size test: {dataset_sizes['test']}")
        
    
    train_loss = []
    val_loss = []
    test_loss = []
    
    num_classes=2
    
    ce_loss1 = CrossEntropyLoss()
    ce_loss2 = SST_loss(gamma0=0.125, gamma1=0.0)
    ce_loss3 = SST_loss(gamma0=0.25, gamma1=0.0)
    
    dice_loss1 = DiceLoss(num_classes)
    dice_loss2 = DiceLoss(num_classes)
    dice_loss3 = DiceLoss(num_classes)
    
    HDLoss_nor = HausdorffLoss()
    bf1_nor = BoundaryF1Score(tolerance=1)
    dice_nor = DiceCoefficientBoundary(tolerance=1)
    pr_nor = ContourPrecisionRecall(tolerance=1)
    
    HDLoss_clu = HausdorffLoss()
    bf1_clu = BoundaryF1Score(tolerance=1)
    dice_clu = DiceCoefficientBoundary(tolerance=1)
    pr_clu = ContourPrecisionRecall(tolerance=1)
    
    ########################## Testing ############################
    model.eval()

    dices, f1s, ious, accuracies, precs, senss, specs = [], [], [], [], [], [], []
    dices_pred1, bf1s_pred1, bdscs_pred1, bprecs_pred1, brecalls_pred1 = [], [], [], [], []
    dices_pred2, bf1s_pred2, bdscs_pred2, bprecs_pred2, brecalls_pred2 = [], [], [], [], []


    with torch.no_grad():
        for i, d in enumerate(testloader):
            img, _, semantic_seg_mask,nor_edge_mask,clu_edge_mask = d

            img = img.float()    
            img = img.to(device)

            preds,pred1,pred2= model(img)


            preds_softmax = torch.softmax(preds, dim=1)
            preds_classes = torch.argmax(preds_softmax, dim=1)

            pred1_softmax = torch.softmax(pred1, dim=1)
            pred1_classes = torch.argmax(pred1_softmax, dim=1)

            pred2_softmax = torch.softmax(pred2, dim=1)
            pred2_classes = torch.argmax(pred2_softmax, dim=1)

            preds_flat = preds_classes.view(-1)
            labels_flat = semantic_seg_mask.view(-1)

            np_preds = preds_flat.cpu().numpy()
            np_labels = labels_flat.cpu().numpy()

            dice = evaluate(np_labels, np_preds, metric="DSC") 
            iou = evaluate(np_labels, np_preds, metric="IoU")  
            acc = evaluate(np_labels, np_preds, metric="ACC")  
            prec = evaluate(np_labels, np_preds, metric="PREC")  
            sens = evaluate(np_labels, np_preds, metric="SENS")  
            spec = evaluate(np_labels, np_preds, metric="SPEC")  
            f1 = 2*prec*sens/(prec+sens)

            dices.append(dice)
            f1s.append(f1) 
            ious.append(iou)
            accuracies.append(acc)
            precs.append(prec)
            senss.append(sens)
            specs.append(spec)

            # **Secondary prediction (pred1) evaluation**
            # Metrics for normal edge segmentation (pred1)
            HDF_loss_nor = HDLoss_nor(pred1, nor_edge_mask.float())

            BF1_nor = bf1_nor(pred1, nor_edge_mask)
            BDSC_nor = dice_nor(pred1, nor_edge_mask)
            BPrec_nor, BRecall_nor = pr_nor(pred1, nor_edge_mask)

            dices_pred1.append(HDF_loss_nor.item())
            bf1s_pred1.append(BF1_nor.item())
            bdscs_pred1.append(BDSC_nor.item())
            bprecs_pred1.append(BPrec_nor.item())
            brecalls_pred1.append(BRecall_nor.item())

            # **Thirdary prediction (pred2) evaluation**
            # Metrics for normal edge segmentation (pred2)
            HDF_loss_clu = HDLoss_clu(pred2, clu_edge_mask.float())

            BF1_clu = bf1_clu(pred2, clu_edge_mask)
            BDSC_clu = dice_clu(pred2, clu_edge_mask)
            BPrec_clu, BRecall_clu = pr_clu(pred2, clu_edge_mask)

            dices_pred2.append(HDF_loss_clu.item())
            bf1s_pred2.append(BF1_clu.item())
            bdscs_pred2.append(BDSC_clu.item())
            bprecs_pred2.append(BPrec_clu.item())
            brecalls_pred2.append(BRecall_clu.item())

    average_dice = np.mean(dices)* 100.0
    average_f1 = np.mean(f1)* 100.0
    average_iou = np.mean(ious)* 100.0
    average_accuracy = np.mean(accuracies)* 100.0
    average_prec = np.mean(precs)* 100.0
    average_sens = np.mean(sens)* 100.0
    average_spec = np.mean(spec)* 100.0

    average_HDF_nor = np.mean(dices_pred1)
    average_BF1_nor = np.mean(bf1s_pred1)* 100.0
    average_BDSC_nor = np.mean(bdscs_pred1)* 100.0
    average_BPrec_nor = np.mean(bprecs_pred1)* 100.0
    average_BRecall_nor = np.mean(brecalls_pred1)* 100.0

    average_HDF_clu = np.mean(dices_pred2)
    average_BF1_clu = np.mean(bf1s_pred2)* 100.0
    average_BDSC_clu = np.mean(bdscs_pred2)* 100.0
    average_BPrec_clu = np.mean(bprecs_pred2)* 100.0
    average_BRecall_clu = np.mean(brecalls_pred2)* 100.0

    wandb.log({
    "Result/Dice": average_dice,
    "Result/F1": average_f1,
    "Result/IoU_JI": average_iou,
    "Result/Accuracy": average_accuracy,
    "Result/Sensitivity": average_sens,
    "Result/Specificity": average_spec,
    "Result/Precision": average_prec,
    "Result_Boundary_Normal/HDF_Loss": average_HDF_nor,
    "Result_Boundary_Normal/F1": average_BF1_nor,
    "Result_Boundary_Normal/DSC": average_BDSC_nor,
    "Result_Boundary_Normal/Precision": average_BPrec_nor,
    "Result_Boundary_Normal/Recall": average_BRecall_nor,
    "Result_Boundary_Cluster/HDF_Loss": average_HDF_clu,
    "Result_Boundary_Cluster/F1": average_BF1_clu,
    "Result_Boundary_Cluster/DSC": average_BDSC_clu,
    "Result_Boundary_Cluster/Precision": average_BPrec_clu,
    "Result_Boundary_Cluster/Recall": average_BRecall_clu
    })
    wandb.finish()

    print(f"average_Dice: {average_dice}\n"
          f"average_F1: {average_f1}\n"
          f"average_IoU_JI: {average_iou}\n"
          f"average_accuracy: {average_accuracy}\n"
          f"average_Precision: {average_prec}\n"
          f"average_sensitivity: {average_sens}\n")

if __name__=='__main__':
    main()