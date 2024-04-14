# MoE (phase 1) with 3 experts with Physical Significance using data_type = 0,1,2, phase 1: train 3 experts with Physical Significance without router
import copy
import logging
import math
from os.path import join as pjoin
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
from torch.nn.modules.loss import CrossEntropyLoss

import torch.optim as optim
import torch.utils.data as data
import scipy.io as sio
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchsummary import summary

import matplotlib.pyplot as plt
import random
import time
import sys
from datetime import datetime
import argparse

from miseval import evaluate
from utils import *

from models.transnuseg_MoE_p1 import TransNuSeg as MoE_p1

from sklearn.metrics import f1_score, accuracy_score
from thop import clever_format, profile
import thop
from itertools import chain

import wandb
import warnings

from dataset_lizard import MyDataset

warnings.filterwarnings("ignore", category=UserWarning, message="torch.meshgrid") # Suppress the specific UserWarning related to torch.meshgrid
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

base_data_dir = "/root/autodl-tmp/data"
HISTOLOGY_DATA_PATH = os.path.join(base_data_dir, 'histology') #HISTOLOGY_DATA_PATH  Containing two folders named data and test
RADIOLOGY_DATA_PATH = os.path.join(base_data_dir,'fluorescence') # Containing two folders named data and test
THYROID_DATA_PATH = os.path.join(base_data_dir,'thyroid') # Containing two folders named data and test
LIZARD_DATA_PATH = os.path.join(base_data_dir,'lizard')

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
    # parser.add_argument("--sharing_ratio",required=True,default=0.5, help=" ratio of sharing proportion of decoders")
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
    # sharing_ratio = float(args.sharing_ratio)
    batch_size=int(args.batch_size)
    random_seed = int(args.random_seed)
    num_epoch = int(args.num_epoch)
    base_lr = float(args.lr)
    
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    
    IMG_SIZE = 512
    wandb.init(project = dataset, name = "MoE p1 seed" + str(random_seed) )

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
        print(f"Dataset {dataset}; data_path {data_path} exists")
    else:
        print(f"data_path {data_path} does not exists")


    model = MoE_p1(img_size=IMG_SIZE)

    if args.model_path is not None:
        try:
            model.load_state_dict(torch.load(args.model_path))
        except Exception as err:
            print("{} In Loading previous model weights".format(err))
            
    model.to(device)
#     input_tensor = torch.randn(1, 3, 512, 512).to(device)
#     flops, params = profile(model, inputs=(input_tensor,))

#     print(f"p1 Model FLOPs: {flops / 1e9} G FLOPs")  # Print FLOPs in Giga FLOPs (GFLOPs)
#     print(f"p1 Model Number of Parameters: {params / 1e6} M parameters")  # Print the number of parameters in millions
#     wandb.log({"FLOPs Gs": flops / 1e9 })
#     wandb.log({"Number of Parameters Ms": params / 1e6 })

    now = datetime.datetime.now()
    
    create_dir('./log')
    logging.basicConfig(filename='./log/log_{}_{}_{}.txt'.format(model_type,dataset,str(now)), level=logging.INFO,
                            format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(f"Random seed:{random_seed}, Batch size:{batch_size}, epoch num:{num_epoch}, alpha:{alpha}, beta:{beta}, gamma:{gamma}")
    
    total_data = MyDataset(dir_path=data_path)
    train_set_size = int(len(total_data) * 0.8)
    val_set_size = int(len(total_data) * 0.1)
    test_set_size = len(total_data) - train_set_size - val_set_size
    logging.info(f"train size {train_set_size}, val size {val_set_size}, test size {test_set_size}")
    
    train_set, val_set, test_set = data.random_split(total_data, [train_set_size, val_set_size, test_set_size],generator=torch.Generator().manual_seed(random_seed))

    trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    valloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=True)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=True)
 
    dataloaders = {"train":trainloader,"val":valloader, "test":testloader}
    dataset_sizes = {"train":len(trainloader), "val":len(valloader), "test":len(testloader)}
    logging.info(f"# of batchs: train: {dataset_sizes['train']}, val: {dataset_sizes['val']}, test: {dataset_sizes['test']}")
    
    train_loss = []
    val_loss = []
    
    num_classes = 2
    
    ce_loss1 = CrossEntropyLoss()
    dice_loss1 = DiceLoss(num_classes)
    ce_loss2 = CrossEntropyLoss()
    dice_loss2 = DiceLoss(num_classes)
    ce_loss3 = CrossEntropyLoss()
    dice_loss3 = DiceLoss(num_classes)
    softmax = torch.nn.Softmax(dim=1)

    optimizer = optim.Adam(model.parameters(), lr=base_lr)
    #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.95, last_epoch=-1)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=2000, eta_min=1e-8)

    
    best_loss = 10000
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

                img, instance_seg_mask, semantic_seg_mask,normal_edge_mask,cluster_edge_mask = d
                
                img = img.float().to(device)
                normal_edge_mask = normal_edge_mask.to(device)
                semantic_seg_mask = semantic_seg_mask.to(device).long()
                cluster_edge_mask = cluster_edge_mask.to(device)
                
                # print('semantic_seg_mask shape: ',semantic_seg_mask.shape)
                # print('semantic_seg_mask: ',semantic_seg_mask)
                # print('input img shape: ', img.shape)
                
                output1, output2, output3 = model(img) 
                #output1 = torch.stack([output1, -output1], dim=1)
                
                # print('output1 shape:', output1.shape)
                ce_loss_seg = ce_loss1(output1, semantic_seg_mask.long())
                dice_loss_seg = dice_loss1(output1, semantic_seg_mask.float(), softmax=True)
                
                #ce_loss_nor = ce_loss2(output2, semantic_seg_mask.long())
                ce_loss_nor = ce_loss2(output2, normal_edge_mask.long())
                dice_loss_nor = dice_loss2(output2, normal_edge_mask.float(), softmax=True)
                
                ce_loss_clu = ce_loss3(output3, cluster_edge_mask.long())
                dice_loss_clu = dice_loss3(output3, cluster_edge_mask.float(), softmax=True)
                
                
                loss_seg = 0.4*ce_loss_seg + 0.6*dice_loss_seg
                loss_nor = 0.4*ce_loss_nor + 0.6*dice_loss_nor
                loss_clu = 0.4*ce_loss_clu + 0.6*dice_loss_clu
                
                loss = alpha*loss_seg  + beta*loss_nor + gamma*loss_clu 

                if phase == 'val':
                    wandb.log({"CE Loss Seg":ce_loss_seg, "Dice Loss Seg":dice_loss_seg})
                    wandb.log({"CE Loss Edge":ce_loss_nor, "Dice Loss Edge":dice_loss_nor})
                    wandb.log({"CE Loss Cluster":ce_loss_clu, "Dice Loss Cluster":dice_loss_clu})
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
            print(f"duration: {(e-s)*100/dataset_sizes[phase]}")
            
            batch_loss = running_loss / dataset_sizes[phase]
            batch_loss_seg = running_loss_seg / dataset_sizes[phase]       ## Avg Loss for nuclei segmantation per batch of one epoch
            print(f"{phase} Epoch: {epoch+1}, Batch Seg loss: {batch_loss_seg}, loss: {batch_loss}, lr: {optimizer.param_groups[0]['lr']}, time: {e-s}")
           
            if phase == 'train':
                train_loss.append(batch_loss)
                # wandb.log({"epoch time/train": e-s})
            else:
                val_loss.append(batch_loss)
                wandb.log({"Seg Loss per batch": batch_loss_seg})
                #print(f"Seg Loss per batch in epoch:{batch_loss_seg}")

            if phase == 'val' and batch_loss_seg < best_loss:
                best_loss = batch_loss_seg
                best_epoch = epoch
                best_model_wts = copy.deepcopy(model.state_dict())
                logging.info("Best val seg loss {} save at epoch {}".format(best_loss,epoch+1))

    draw_loss(train_loss,val_loss,str(now))
    
    create_dir('./saved')
    
    formatted_time = now.strftime("%m%d_%H%M")  # YYYYMMDD_HHMM format
    save_path = f"./saved/model_{dataset}_{model_type}_seed{random_seed}_epoch{best_epoch}_loss_{best_loss:.5f}_{formatted_time}.pt"
    torch.save(best_model_wts, save_path)
    logging.info(f'Model saved at: {save_path}')
 
    ########################## Testing ##############################  
    #test_main('p1',save_path)
    model.load_state_dict(best_model_wts)
    model.eval()
    
    dices = []
    f1s= []
    ious = []
    accuracies = []
    precs = []
    senss =[]
    specs = []

    with torch.no_grad():
        for i, d in enumerate(testloader):
            img, _, semantic_seg_mask,_,_ = d

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
            
    average_dice = np.mean(dices)
    average_f1 = np.mean(f1)
    average_iou = np.mean(ious)
    average_accuracy = np.mean(accuracies)
    average_prec = np.mean(precs)
    average_sens = np.mean(sens)
    average_spec = np.mean(spec) 

  
    print(f"average_Dice: {average_dice}\n"
          f"average_F1: {average_f1}\n"
          f"average_IoU_JI: {average_iou}\n"
          f"average_accuracy: {average_accuracy}\n"
          f"average_Precision: {average_prec}\n"
          f"average_sensitivity: {average_sens}\n"
          f"average_specificity: {average_spec}")
    
    parts = save_path.split('/')
    model_part = next((part for part in parts if part.startswith('model')), None) # extract the weight file name instead of whole path
    log_entry = f"{dataset}, {random_seed}, 'p1', {average_dice*100.0}, {average_iou*100.0}, {average_f1*100.0}, {average_accuracy*100.0}, {average_prec*100.0},{average_sens*100.0}, {average_spec*100.0}, {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, {model_part}\n"

    # Append the log entry to the file
    with open("./log/results.txt", "a") as file:
        file.write(log_entry) 
    
    wandb.log({"Dice": average_dice*100.0})
    wandb.log({"F1": average_f1*100.0})
    wandb.log({"IoU_JI": average_iou*100.0})
    wandb.log({"Accuracy": average_accuracy*100.0})
    wandb.log({"Sensitivity": average_sens*100.0})
    wandb.log({"Specificity": average_spec*100.0})  
    wandb.log({"Precision": average_prec*100.0})
    wandb.finish()
            
if __name__=='__main__':
    main()