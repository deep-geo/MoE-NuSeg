# MoE-NuSeg (phase 2) : train the gating network together with 3 experts

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

checkpoint_path = "/root/autodl-tmp/MoE-NuSeg/saved/model_Radiology_MoE_p1_seed42_epoch147_loss_0.08226_1125_1452.pt"

def main():
    '''
    model_type:  default: transnuseg
    alpha: ratio of the loss of nuclei mask loss, dafault=0.3
    beta: ratio of the loss of normal edge segmentation, dafault=0.35
    gamma: ratio of the loss of cluster edge segmentation, dafault=0.35
    sharing_ratio: ratio of sharing  reviewedportion of decoders, default=0.5
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
    dataset = args.dataset
    
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
 
    wandb.init(project = dataset, name = "MoE p2 seed" + str(random_seed))

    model = MoE_p2(img_size=IMG_SIZE)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint, strict=False)
    print(f"=========Phase 2 training: Phase 1 model was loaded, Retrain all==========")
    print(f"Loaded Model: {checkpoint_path}")
    
    if os.path.exists(data_path):
        print(f"Dataset: {dataset}; data_path: {data_path} exists")
    else:
        print(f"data_path {data_path} does not exists")
    
        
#     # Freeze all parameters of the model
#     for param in model.parameters():
#         param.requires_grad = False
        
#     # Unfreeze the gating layer's pFarameters
#     for layer in model.layers_up:
#         if hasattr(layer, 'blocks'):  # Check if the layer has the 'blocks' attribute
#             for block in layer.blocks:
#                 for param in block.gating_network.parameters():
#                     param.requires_grad = True
#                 for expert in block.experts:
#                     for param in expert.parameters():
#                         param.requires_grad = True

            
    model.to(device)

    now = datetime.now()
    create_dir('./log')
    logging.basicConfig(filename='./log/log_{}_{}_{}.txt'.format(model_type,dataset,str(now)), level=logging.INFO,
                            format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(f"Seed:{random_seed}, Batch size:{batch_size}, epoch num:{num_epoch}")
    
    total_data = MyDataset(dir_path=data_path, seed=random_seed)
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
    logging.info(f"batch size train: {dataset_sizes['train']}, batch size val: {dataset_sizes['val']}, batch size test: {dataset_sizes['test']}")
        
    
    train_loss = []
    val_loss = []
    test_loss = []
    
    num_classes=2
    
    ce_loss1 = CrossEntropyLoss()
    ce_loss2 = CrossEntropyLoss()
    ce_loss3 = CrossEntropyLoss()
    
    dice_loss1 = DiceLoss(num_classes)
    dice_loss2 = DiceLoss(num_classes)
    dice_loss3 = DiceLoss(num_classes)
    
    HDLoss_nor = HausdorffLoss()
    bf1_nor = BoundaryF1Score(tolerance=2)
    dice_nor = DiceCoefficientBoundary(tolerance=2)
    pr_nor = ContourPrecisionRecall(tolerance=2)
    
    HDLoss_clu = HausdorffLoss()
    bf1_clu = BoundaryF1Score(tolerance=2)
    dice_clu = DiceCoefficientBoundary(tolerance=2)
    pr_clu = ContourPrecisionRecall(tolerance=2)

    optimizer = optim.Adam(model.parameters(), lr=base_lr)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=400, eta_min=1e-8)
 
    best_loss = 100
    best_epoch = 0
    
    for epoch in range(num_epoch):
        # early stop, if the loss does not decrease for 50 epochs
        if epoch > best_epoch + 50:
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

                img, _, semantic_seg_mask,normal_edge_mask,cluster_edge_mask = d
             
                img = img.float()    
                img = img.to(device)
               
                semantic_seg_mask = semantic_seg_mask.to(device)
                normal_edge_mask = normal_edge_mask.to(device)
                cluster_edge_mask = cluster_edge_mask.to(device)
                
                #print('semantic_seg_mask shape ',semantic_seg_mask.shape)
                #print('input img shape: ', img.shape)

                output1,output2,output3 = model(img)
                #print('output1 shape ', output1.shape)
                
                ce_loss_seg = ce_loss1(output1, semantic_seg_mask.long())
                dice_loss_seg = dice_loss1(output1, semantic_seg_mask.float(), softmax=True)
                
                ce_loss_nor = ce_loss2(output2, normal_edge_mask.long())
                dice_loss_nor = dice_loss2(output2, normal_edge_mask.float(), softmax=True)
                
                ce_loss_clu = ce_loss3(output3, cluster_edge_mask.long())
                dice_loss_clu = dice_loss3(output3, cluster_edge_mask.float(), softmax=True)
                
                
                loss_seg = 0.4*ce_loss_seg + 0.6*dice_loss_seg
                loss_nor = 0.4*ce_loss_nor + 0.6*dice_loss_nor
                loss_clu = 0.4*ce_loss_clu + 0.6*dice_loss_clu
                

                if phase == 'val':
                    wandb.log({
                        "CE Loss Seg": ce_loss_seg, 
                        "Dice Loss Seg": dice_loss_seg,
                        "CE Loss Edge": ce_loss_nor,
                        "Dice Loss Edge": dice_loss_nor,
                        "CE Loss Cluster": ce_loss_clu,
                        "Dice Loss Cluster": dice_loss_clu
                    })

                    BF1_nor = bf1_nor(output2, normal_edge_mask.float())
                    BDSC_nor = dice_nor(output2, normal_edge_mask.float())
                    BPrec_nor, BRecall_nor = pr_nor(output2, normal_edge_mask.float())
                    HDF_loss_nor = HDLoss_nor(output2, normal_edge_mask)
                    
                    HDF_loss_clu = HDLoss_clu(output3, cluster_edge_mask.float())
                    BF1_clu = bf1_clu(output3, cluster_edge_mask.float())
                    BDSC_clu = dice_clu(output3, cluster_edge_mask.float())
                    BPrec_clu, BRecall_clu = pr_clu(output3, cluster_edge_mask.float())
                    wandb.log({
                        "HDF_loss_nor": HDF_loss_nor.item(),
                        "BF1_nor": BF1_nor.item(),
                        "BDSC_nor": BDSC_nor.item(),
                        "BPrec_nor": BPrec_nor.item(),
                        "BRecall_nor": BRecall_nor.item(),
                        "HDF_loss_clu": HDF_loss_nor.item(),
                        "BF1_clu": BF1_clu.item(),
                        "BDSC_clu": BDSC_clu.item(),
                        "BPrec_clu": BPrec_clu.item(),
                        "BRecall_clu": BRecall_clu.item()
                    })
                    
                    
                    if epoch%30 == 1:
                        log_predictions_to_wandb(img, output1, semantic_seg_mask, output2, normal_edge_mask, output3, cluster_edge_mask, epoch, prefix='comparison')

                ### calculating total loss
                loss = alpha*loss_seg  + beta*loss_nor + gamma*loss_clu 
                wandb.log({"Total Loss":loss})
                #print(f"Total Loss:{loss}")

                running_loss+=loss.item()
                running_loss_seg += loss_seg.item() ## Loss for nuclei segmantation
                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    lr_scheduler.step()
                
            e = time.time()
            batch_loss = running_loss / dataset_sizes[phase]
            batch_loss_seg = running_loss_seg / dataset_sizes[phase]       ## Avg Loss for nuclei segmantation per batch of one epoch
            print(f"{phase} Epoch: {epoch+1}, Batch Seg loss: {batch_loss_seg}, loss: {batch_loss}, lr: {optimizer.param_groups[0]['lr']}, time: {e-s}")
           
           
            if phase == 'train':
                train_loss.append(batch_loss)
                # wandb.log({"Seg Loss/train": batch_loss_seg})
            else:
                test_loss.append(batch_loss_seg)
                wandb.log({"Seg Loss per batch": batch_loss_seg})

            if phase == 'val' and batch_loss_seg < best_loss:
                best_loss = batch_loss_seg
                best_epoch = epoch+1
                best_model_wts = copy.deepcopy(model.state_dict())
                logging.info("Best val seg loss {} save at epoch {}".format(best_loss,epoch+1))
    
    create_dir('./saved')
    
    formatted_time = now.strftime("%m%d_%H%M")  # YYYYMMDD_HHMMSS format
    save_path = f"./saved/model_{dataset}_{model_type}_seed{random_seed}_epoch{best_epoch}_loss_{best_loss:.5f}_{formatted_time}.pt"
    torch.save(best_model_wts, save_path)
    logging.info(f'Model saved at: {save_path}')
    
    ########################## Testing ############################
    #test_main('p2',save_path)
    model.load_state_dict(best_model_wts)
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
            
            #write_images(img, semantic_seg_mask, preds_classes, pred1_classes, pred2_classes, output_dir)
            
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
            pred1_probs = torch.sigmoid(pred1)  # Sigmoid for binary prediction
            pred1_classes = (pred1_probs > 0.5).long()

            HDF_loss_nor = HDLoss_nor(pred1_probs, normal_edge_mask.float())
            BF1_nor = bf1_nor(pred1_classes, normal_edge_mask)
            BDSC_nor = dice_nor(pred1_classes, normal_edge_mask)
            BPrec_nor, BRecall_nor = pr_nor(pred1_classes, normal_edge_mask)
            
            dices_pred1.append(HDF_loss_nor.item())
            bf1s_pred1.append(BF1_nor.item())
            bdscs_pred1.append(BDSC_nor.item())
            bprecs_pred1.append(BPrec_nor.item())
            brecalls_pred1.append(BRecall_nor.item())
            
            # **Thirdary prediction (pred2) evaluation**
            # Metrics for normal edge segmentation (pred2)
            pred2_probs = torch.sigmoid(pred2)  # Sigmoid for binary prediction
            pred2_classes = (pred2_probs > 0.5).long()

            HDF_loss_clu = HDLoss_clu(pred2, clu_edge_mask.float())
            BF1_clu = bf1_clu(pred2_classes, clu_edge_mask)
            BDSC_clu = dice_clu(pred2_classes, clu_edge_mask)
            BPrec_clu, BRecall_clu = pr_clu(pred1_classes, clu_edge_mask)

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
          f"average_sensitivity: {average_sens}\n"
          f"average_specificity: {average_spec}")
    
    parts = save_path.split('/')
    model_part = next((part for part in parts if part.startswith('model')), None) # extract the weight file name instead of whole path
    log_entry = f"{dataset}, 'p2', {random_seed}, {average_dice}, {average_iou}, {average_f1}, {average_accuracy}, {average_prec},{average_sens}, {average_spec}, {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, {model_part}\n"

    # Append the log entry to the file
    with open("./log/results.txt", "a") as file:
        file.write(log_entry)  

if __name__=='__main__':
    main()