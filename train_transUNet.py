import torch
import torch.nn as nn
import copy
import logging
import time
import os
import random
import numpy as np
from datetime import datetime
from torch.optim import SGD
from dataset import MyDataset
from models.transUNet import TransUNet
import wandb
from utils import *
import argparse
import sys

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
IMG_SIZE = 512
base_data_dir = "/root/autodl-tmp/data"
HISTOLOGY_DATA_PATH = os.path.join(base_data_dir, 'histology')

def calculate_loss_and_log(ce_loss, dice_loss, output, target, phase):
    ce_l = ce_loss(output, target.float())
    dice_l = dice_loss(output, target.float(), softmax=True)
    total_loss = 0.4 * ce_l + 0.6 * dice_l
    wandb.log({f"CE Loss/{phase}": ce_l, f"Dice Loss/{phase}": dice_l})
    return total_loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', required=True, default="transnuseg", help="model type")
    parser.add_argument("--alpha", required=True, default=0.3, help="weight of nuclei mask loss")
    parser.add_argument("--beta", required=True, default=0.35, help="weight of normal edge loss")
    parser.add_argument("--gamma", required=True, default=0.35, help="weight of cluster edge loss")
    parser.add_argument("--random_seed", required=True, help="random seed")
    parser.add_argument("--batch_size", required=True, help="batch size")
    parser.add_argument("--dataset", required=True, default="Histology", help="dataset")
    parser.add_argument("--num_epoch", required=True, help="number of epochs")
    parser.add_argument("--lr", required=True, help="learning rate")
    parser.add_argument("--model_path", default=None, help="path to pretrained model")
    parser.add_argument("--sharing_ratio",required=True,default=0.5, help=" ratio of sharing proportion of decoders")
    args = parser.parse_args()

    # Seed initialization
    random_seed = int(args.random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    # Dataset and model setup
    dataset = args.dataset
    data_path = HISTOLOGY_DATA_PATH
    model = TransUNet(img_dim=IMG_SIZE, in_channels=3, out_channels=128, class_num=1, head_num=4, mlp_dim=512, block_num=8, patch_dim=16)
    model.to(device)
    
    # Wandb setup
    wandb.init(project=dataset, name="TransUNet" + str(random_seed))
    
    # Logging setup
    now = datetime.datetime.now()
    create_dir('./log')
    logging.basicConfig(filename=f'./log/log_{args.model_type}_{dataset}_{now}.txt', level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(f"Seed: {args.random_seed}, Batch size: {args.batch_size}, Epoch num: {args.num_epoch}")

    # Data loading
    total_data = MyDataset(dir_path=data_path)
    train_size = int(0.8 * len(total_data))
    val_size = len(total_data) - train_size
    train_set, val_set = torch.utils.data.random_split(total_data, [train_size, val_size], generator=torch.Generator().manual_seed(random_seed))
    
    dataloaders = {
        'train': torch.utils.data.DataLoader(train_set, batch_size=int(args.batch_size), shuffle=True, drop_last=True),
        'val': torch.utils.data.DataLoader(val_set, batch_size=int(args.batch_size), shuffle=False, drop_last=True)
    }

    # Loss and optimizer setup
    ce_loss1 = CrossEntropyLoss()
    #ce_loss = nn.BCELoss()
    dice_loss1 = BinaryDiceLoss()
    optimizer = SGD(model.parameters(), lr=float(args.lr), momentum=0.9, weight_decay=1e-4)
    
    # Training loop
    best_loss = float('inf')
    best_epoch = 0
    for epoch in range(int(args.num_epoch)):
        if epoch > best_epoch + 50:
            break

        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running_loss = 0.0
            start_time = time.time()

            for img, _, semantic_seg_mask, _, _ in dataloaders[phase]:
                img = img.float().to(device)
                semantic_seg_mask = semantic_seg_mask.to(device).unsqueeze(1).long()

                with torch.set_grad_enabled(phase == 'train'):
                    output = model(img)
                    # print(f"output={output[0]}")
                    # print(f"semantic_seg_mask={semantic_seg_mask[0]}")
                    loss = calculate_loss_and_log(ce_loss1, dice_loss1, output, semantic_seg_mask, phase)
                    #loss = ce_loss(output, semantic_seg_mask.float())
                    #print(f"loss={loss.item()}")
                    
                    if phase == 'train':
                        
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                
                running_loss += loss.item()

            epoch_loss = running_loss / len(dataloaders[phase])
            logging.info(f"{phase} Epoch: {epoch+1}, loss: {epoch_loss}, lr: {optimizer.param_groups[0]['lr']}, time: {time.time() - start_time}")
            wandb.log({f"{phase} Loss": epoch_loss})

            # Save best model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_epoch = epoch + 1
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, f"./saved/model_{dataset}_{args.model_type}_epoch{best_epoch}_loss_{best_loss:.4f}.pt")
                logging.info(f"Best val loss: {best_loss} saved at epoch {best_epoch}")
                
if __name__ == '__main__':
    main()