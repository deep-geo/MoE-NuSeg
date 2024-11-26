# Standard Libraries
import os
import time
import logging
import sys
from datetime import datetime
from os.path import join as pjoin
import uuid

# Scientific Computing and Image Processing
import numpy as np
from scipy import ndimage
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# PyTorch and Torch Utilities
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from torchvision import transforms
from timm.models.layers import DropPath, trunc_normal_
from einops import rearrange
import torch.utils.checkpoint as checkpoint

# Configuration and Logging
import yaml
import wandb

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        score = score.to(target.device)
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        # print("Inputs shape:", inputs.shape)
        # print("Target shape:", target.shape)
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes
    
class BinaryDiceLoss(nn.Module):  # added 20240117
    def __init__(self):
        super(BinaryDiceLoss, self).__init__()

    def _dice_loss(self, score, target):
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.sigmoid(inputs)
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        dice = self._dice_loss(inputs, target)
        return dice
    


def create_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

        
# class BCE_Loss_logits(nn.Module):
#     def __init__(self):
#         super(BCE_Loss_logits, self).__init__()

#     def forward(self, logits, targets):
#         # Ensure targets are of type float
#         targets = targets.float()
#         y_pred_prob = torch.sigmoid(logits)
        
#         epsilon = 1e-11
        
#         # Positive and negative loss components
#         # pos_loss = targets * F.softplus(-logits)  # Equivalent to -y * log(sigmoid(z))
#         # neg_loss = (1 - targets) * F.softplus(logits)  # Equivalent to -(1 - y) * log(1 - sigmoid(z))
        
#         pos_loss = targets * torch.log(y_pred_prob + epsilon)  # -y * log(sigmoid(logits))
#         neg_loss = (1.0 - targets) * torch.log(1.0 - y_pred_prob + epsilon)  # -(1 - y) * log(1 - sigmoid(logits))

#         # Combine losses
#         loss = -(torch.mean(pos_loss + neg_loss))
#         return loss


class SST_loss(torch.nn.Module):
    def __init__(self, gamma0=0.5, gamma1=0.0):
        super(SST_loss, self).__init__()
        self.gamma0 = gamma0
        self.gamma1 = gamma1
       

    def forward(self, y_pred, y):
        # print(f"y_pred:{y_pred}")
        # print(f"y:{y}")
        
        y = y
        y_pred_prob = torch.sigmoid(y_pred)
        
        epsilon = 1e-11

        # Loss calculations
        y_1_loss = -y * torch.pow(y_pred_prob, -self.gamma0) * torch.log(y_pred_prob + epsilon)
        y_0_loss = -(1.0 - y) * torch.pow(1.0 - y_pred_prob, -self.gamma1) * torch.log(1 - y_pred_prob + epsilon)
      
        return torch.mean(y_0_loss+y_1_loss)
    


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        """
        Focal Loss for binary classification tasks.
        
        Args:
            gamma (float): The focusing parameter that adjusts the loss for well-classified and hard-to-classify examples.
            alpha (float or None): Weighting factor for class imbalance. If set, it balances the importance of positive/negative examples.
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, y_pred, y_true):
        """
        Forward pass of Focal Loss.
        
        Args:
            y_pred (torch.Tensor): Predicted logits from the model (before sigmoid). Shape [batch_size, ...].
            y_true (torch.Tensor): Ground truth labels (0 or 1). Shape [batch_size, ...].
            
        Returns:
            torch.Tensor: The computed focal loss.
        """
        # Apply sigmoid to get probabilities
        y_pred_prob = torch.sigmoid(y_pred)
        
        # Epsilon to avoid log(0)
        epsilon = 1e-11
        y_pred_prob = torch.clamp(y_pred_prob, epsilon, 1.0 - epsilon)
        
        # Compute the cross-entropy loss for both classes
        ce_loss_1 = -y_true * torch.log(y_pred_prob)  # For class 1 (positive class)
        ce_loss_0 = -(1 - y_true) * torch.log(1 - y_pred_prob)  # For class 0 (negative class)

        # Compute modulating factors for focal loss
        focal_weight_1 = (1 - y_pred_prob) ** self.gamma  # For class 1
        focal_weight_0 = y_pred_prob ** self.gamma  # For class 0

        # Apply class weighting if alpha is provided
        if self.alpha is not None:
            focal_weight_1 = focal_weight_1 * self.alpha  # Weight for class 1
            focal_weight_0 = focal_weight_0 * (1 - self.alpha)  # Weight for class 0

        # Combine focal weights with cross-entropy loss
        loss_1 = focal_weight_1 * ce_loss_1  # For positive class (y = 1)
        loss_0 = focal_weight_0 * ce_loss_0  # For negative class (y = 0)

        # Return the mean of the loss
        return torch.mean(loss_1 + loss_0)


class HausdorffLoss(nn.Module):
    def __init__(self):
        super(HausdorffLoss, self).__init__()
    
    def forward(self, y_pred, y):
        """
        y_pred: Predicted segmentation logits (B, C, H, W)
        y: Ground truth segmentation (B, C, H, W)
        """
        # Get predicted probability using sigmoid
        y_pred_prob = torch.sigmoid(y_pred)
        y_pred_prob = y_pred_prob[:, 1:2, :, :]  # Keep only the second channel (foreground)
        
        if y.dim() == 3:
            y = y.unsqueeze(1)  # Add a channel dimension to y


        # Compute the edges for the predicted and ground truth foregrounds
        pred_edge = self.compute_edges(y_pred_prob)
        gt_edge = self.compute_edges(y)

        # Calculate Hausdorff distance
        hd_loss = self.hausdorff_distance(pred_edge, gt_edge)

        return hd_loss
    
    def compute_edges(self, img):
        
        # Convert to float tensor if necessary
        img = img.float() if img.dtype != torch.float32 else img
        
        """Computes the edges using Sobel filters for each channel."""
        sobel_kernel_x = self.sobel_kernel_x().to(img.device)
        sobel_kernel_y = self.sobel_kernel_y().to(img.device)

        grad_x = F.conv2d(img, sobel_kernel_x, padding=1)
        grad_y = F.conv2d(img, sobel_kernel_y, padding=1)
        
        edge = torch.sqrt(grad_x**2 + grad_y**2 + 1e-11)  # Add epsilon to avoid sqrt(0)
        edge = (edge > 0.1).float()
        return edge

    def sobel_kernel_x(self):
        """Create a Sobel kernel for x-gradient."""
        kernel = torch.tensor([[1, 0, -1],
                               [2, 0, -2],
                               [1, 0, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return kernel

    def sobel_kernel_y(self):
        """Create a Sobel kernel for y-gradient."""
        kernel = torch.tensor([[1, 2, 1],
                               [0, 0, 0],
                               [-1, -2, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return kernel

    
    def hausdorff_distance(self, pred_edge, gt_edge):
        """
        Compute the Hausdorff distance between predicted and ground truth edge maps.
        """
        # Flatten the edge maps to compute distances between all points
        pred_edge_flat = pred_edge.view(pred_edge.shape[0], -1).to(pred_edge.device)
        gt_edge_flat = gt_edge.view(gt_edge.shape[0], -1).to(pred_edge.device)

        # For each predicted edge pixel, find the closest ground truth edge pixel
        dist_pred_to_gt = torch.min(torch.cdist(pred_edge_flat, gt_edge_flat, p=2), dim=-1)[0]

        # For each ground truth edge pixel, find the closest predicted edge pixel
        dist_gt_to_pred = torch.min(torch.cdist(gt_edge_flat, pred_edge_flat, p=2), dim=-1)[0]

        # Combine the two distances to approximate Hausdorff distance
        hd_loss = torch.mean(dist_pred_to_gt + dist_gt_to_pred)

        # Normalize by image diagonal length
        diagonal_length = torch.sqrt(torch.tensor(pred_edge.shape[-1]**2 + pred_edge.shape[-2]**2, dtype=torch.float32))
        hd_loss_normalized = hd_loss / diagonal_length

        return hd_loss_normalized
       


class DiceCoefficientBoundary(nn.Module):
    def __init__(self, tolerance=2):
        """
        Initialize the DiceCoefficientBoundary class with a tolerance level.
        """
        super(DiceCoefficientBoundary, self).__init__()
        self.tolerance = tolerance

    def forward(self, pred_edge, gt_edge):
        """
        Calculate the Dice Similarity Coefficient (DSC) for boundary pixels within a specified tolerance.
        """
        device = pred_edge.device
        gt_edge = gt_edge.to(device)
        pred_edge = torch.sigmoid(pred_edge)
        
        # Dilate both edges by a small tolerance to include nearby pixels
        gt_dilated = F.max_pool2d(gt_edge.float(), kernel_size=(2 * self.tolerance + 1), stride=1, padding=self.tolerance)
        pred_dilated = F.max_pool2d(pred_edge.float(), kernel_size=(2 * self.tolerance + 1), stride=1, padding=self.tolerance)

        # Compute intersection and union for Dice calculation
        intersection = (pred_edge * gt_dilated).sum()
        union = pred_edge.sum() + gt_edge.sum()
        dice_score = (2 * intersection) / (union + 1e-8)

        return dice_score


class ContourPrecisionRecall(nn.Module):
    def __init__(self, tolerance=2, threshold=0.5):
        super(ContourPrecisionRecall, self).__init__()
        self.tolerance = tolerance
        self.threshold = threshold

    def forward(self, pred, gt):
        
        device = pred.device
        gt = gt.to(device)
        
        # Apply sigmoid to convert logits to probabilities, then threshold to get binary mask
        pred_prob = torch.sigmoid(pred)  # Apply sigmoid to logits

        # Select the foreground channel (assuming it's the second channel) and squeeze it
        pred_edge = (pred_prob[:, 1, :, :] > self.threshold).float().squeeze(dim=1)

        # Convert ground truth to binary mask assuming it's already single channel
        gt_edge = (gt > self.threshold).float()

        # Dilate ground truth edge by the tolerance to allow nearby matches
        gt_dilated = F.max_pool2d(gt_edge.unsqueeze(1), kernel_size=(2 * self.tolerance + 1), stride=1, padding=self.tolerance).squeeze(dim=1)
        pred_dilated = F.max_pool2d(pred_edge.unsqueeze(1), kernel_size=(2 * self.tolerance + 1), stride=1, padding=self.tolerance).squeeze(dim=1)

        # Precision and recall calculations
        precision = (pred_edge * gt_dilated).sum() / (pred_edge.sum() + 1e-8)
        recall = (gt_edge * pred_dilated).sum() / (gt_edge.sum() + 1e-8)

        return precision, recall
    
class BoundaryF1Score(nn.Module):
    def __init__(self, tolerance=2, threshold=0.5):
        super(BoundaryF1Score, self).__init__()
        # Initialize ContourPrecisionRecall to reuse precision and recall calculations
        self.contour_pr = ContourPrecisionRecall(tolerance=tolerance, threshold=threshold)

    def forward(self, pred, gt):
        # Compute precision and recall using ContourPrecisionRecall
        precision, recall = self.contour_pr(pred, gt)
        
        # Calculate Boundary F1 Score
        bf1_score = 2 * (precision * recall) / (precision + recall + 1e-8)
        return bf1_score


def logits2image(t: torch.Tensor) -> np.ndarray:
    t = t.cpu().detach().numpy()
    binary_mask = np.zeros_like(t, dtype=np.uint8)
    binary_mask[t >= 0] = 255
    return binary_mask

def binary_gt2image(t: torch.Tensor) -> np.ndarray:
    # Convert the tensor to a NumPy array and set all non-zero values to 255
    binary_array = t.cpu().numpy() != 0  # Create a boolean mask
    return (binary_array.astype(np.uint8)) * 255  # Convert to uint8 and scale to 255

def apply_colormap(mask, color='green'):
    """
    Apply colormap to a single-channel mask.
    """
    # Ensure mask is 2D (512, 512)
    # if len(mask.shape) > 2:
    #     mask = mask[0]  # Take the first channel if needed
    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    colored_mask[:, :, 1] = mask  # Apply to green channel
    
    return colored_mask

def log_predictions_to_wandb(raw_input, seg_masks, seg_masks_gt, 
                             normal_edge_masks, normal_edge_masks_gt, 
                             cluster_edge_masks, cluster_edge_masks_gt,
                             step, prefix='comparison'):
    """
    Log raw input, ground truth, and prediction masks and original images to wandb.
    """

    # Prepare the raw input
    color_raw = raw_input.detach().cpu().numpy()[0]*255 
    color_raw = color_raw[:3, :, :]
    color_raw = np.transpose(color_raw, (1, 2, 0))

    # Convert predicted masks to 2D images and ensure they are (512, 512)
    seg_mask_img = logits2image(seg_masks[0])
    normal_edge_mask_img = logits2image(normal_edge_masks[0])
    cluster_edge_mask_img = logits2image(cluster_edge_masks[0])

    # Convert GT masks to 2D and ensure they are (512, 512)
    seg_mask_gt_img = binary_gt2image(seg_masks_gt[0])
    normal_edge_mask_gt = binary_gt2image(normal_edge_masks_gt[0])
    cluster_edge_mask_gt = binary_gt2image(cluster_edge_masks_gt[0])

    # Apply green colormap to GT masks
    seg_mask_gt_img_rgb = apply_colormap(seg_mask_gt_img, color='green')
    normal_edge_mask_gt_rgb = apply_colormap(normal_edge_mask_gt, color='green')
    cluster_edge_mask_gt_rgb = apply_colormap(cluster_edge_mask_gt, color='green')

    # Apply green colormap to prediction masks
    seg_mask_img_rgb = np.zeros((512, 512, 3), dtype=np.uint8)
    normal_edge_mask_img_rgb = np.zeros((512, 512, 3), dtype=np.uint8)
    cluster_edge_mask_img_rgb = np.zeros((512, 512, 3), dtype=np.uint8)

    seg_mask_img_rgb[:, :, 1] = seg_mask_img[1]
    normal_edge_mask_img_rgb[:, :, 1] = normal_edge_mask_img[1]
    cluster_edge_mask_img_rgb[:, :, 1] = cluster_edge_mask_img[1]
    
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print(f"nuclei rgb ratio: {np.sum(seg_mask_img_rgb[:,:,1] == 255)*100.0 / (512 * 512)}")
    print(f"edge rgb ratio: {np.sum(normal_edge_mask_img_rgb[:,:,1] == 255)*100.0/ (512 * 512)}")
    print(f"cluster rgb edge ratio: {np.sum(cluster_edge_mask_img_rgb[:,:,1] == 255)*100.0 / (512 * 512)}")
    print("-------------------------------------------------------------")

    # Prepare separator
    separator = np.ones((512, 10, 3), dtype=np.uint8) * 255  # (512, 10, 3)
    
    # Concatenate images along width (axis=1)
    combined_img = np.concatenate((
        color_raw, separator, 
        seg_mask_gt_img_rgb, separator, 
        seg_mask_img_rgb, separator,
        normal_edge_mask_gt_rgb, separator, 
        normal_edge_mask_img_rgb, separator, 
        cluster_edge_mask_gt_rgb, separator, 
        cluster_edge_mask_img_rgb
    ), axis=1)

    # Ensure the image is in np.uint8 format for OpenCV
    combined_img = combined_img.astype(np.uint8)
    
    # Add text annotations (if needed)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    color = (255, 255, 255)  # White color
    thickness = 1
    line_type = cv2.LINE_AA
    text_y = 30

    cv2.putText(combined_img, "Raw Input", (10, text_y), font, font_scale, color, thickness, line_type)
    cv2.putText(combined_img, "GT", (512 + 20, text_y), font, font_scale, color, thickness, line_type)
    cv2.putText(combined_img, "Pred.", (2 * 512 + 30, text_y), font, font_scale, color, thickness, line_type)
    cv2.putText(combined_img, "GT_Edge", (3 * 512 + 40, text_y), font, font_scale, color, thickness, line_type)
    cv2.putText(combined_img, "Pred._Edge", (4 * 512 + 50, text_y), font, font_scale, color, thickness, line_type)
    cv2.putText(combined_img, "GT_ClusterE", (5 * 512 + 60, text_y), font, font_scale, color, thickness, line_type)
    cv2.putText(combined_img, "Pred._ClusterE", (6 * 512 + 70, text_y), font, font_scale, color, thickness, line_type)

    # Log to WandB
    wandb.log({f"prediction_{step}": wandb.Image(combined_img, caption=f"Step {step}")})


