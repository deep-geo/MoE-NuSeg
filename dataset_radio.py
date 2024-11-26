import os
import numpy as np
import torch
from torch.utils.data import Dataset
import glob
import cv2
from PIL import Image
from torchvision import transforms

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class MyDataset(Dataset):
    '''
    dir_path: path to data, having two folders named data and label respectively
    '''
    def __init__(self,dir_path,transform = None,in_chan = 3,seed=42): 
        self.dir_path = dir_path
        self.transform = transform
        self.data_path = os.path.join(dir_path,"data")
        self.data_lists = sorted(glob.glob(os.path.join(self.data_path,"*.png")))
        self.label_path = os.path.join(dir_path,"label")
        self.label_lists = sorted(glob.glob(os.path.join(self.label_path,"*.png")))
        
        self.in_chan = in_chan
        self.seed = seed
        
    def __getitem__(self,index):
        img_path = self.data_lists[index]
        label_path = self.label_lists[index]
        if self.in_chan == 3:
            img = Image.open(img_path).convert("RGB")
        else:
            img = Image.open(img_path).convert("L")
        label = cv2.imread(label_path)  # instance mask
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        
        label_copy = label.copy()
        
        semantic_mask = label.copy()
        semantic_mask[semantic_mask!=0]=1
        
        instance_mask = label.copy()
        normal_edge_mask = self.extract_instance_boundaries(instance_mask)
        cluster_edge_mask = self.generate_cluster_edge_mask_grey(label_copy)
        
        if self.transform is not None:
            img = self.transform(img)
            label = self.transform(label)
        else:
            T = transforms.Compose([
                transforms.ToTensor()
            ])
            img = T(img)
            img = img
            semantic_mask = torch.tensor(semantic_mask)
        img = img.to(device)
        semantic_mask = semantic_mask.to(device)
        instance_mask = torch.tensor(instance_mask).to(device)
        normal_edge_mask = torch.tensor(normal_edge_mask).to(device)
        cluster_edge_mask = torch.tensor(cluster_edge_mask).to(device)
        return img,instance_mask,semantic_mask, normal_edge_mask,cluster_edge_mask
    
    def __len__(self):
        return len(self.data_lists)

    
    def extract_instance_boundaries(self, instance_mask):

        # Initialize an empty mask for boundaries
        boundaries = np.zeros_like(instance_mask, dtype=np.uint8)
    
        # Find unique instances excluding background
        instances = np.unique(instance_mask)
        if 0 in instances:
            instances = instances[1:]  # Exclude background
    
        for instance_id in instances:
        # Create a binary mask for the current instance
            binary_mask = np.uint8(instance_mask == instance_id)
        
        # Use morphological operations to highlight the boundary
            dilation = cv2.dilate(binary_mask, np.ones((3,3), np.uint8), iterations=1)
            boundary = dilation - binary_mask
        
        # Add the boundary to the overall boundary mask
            boundaries[boundary > 0] = 1  # Highlight boundary in white
    
        return boundaries



    def generate_cluster_edge_mask_grey(self, label):
        """
        Generate cluster edge masks from grayscale instance segmentation masks.
        Identifies edges between touching cells.
    
        Args:
        - label (numpy.ndarray): Grayscale instance segmentation mask.
    
        Returns:
        - numpy.ndarray: Binary cluster edge mask.
        """
    # Dilate the label to make adjacent instances overlap
        kernel = np.ones((3, 3), np.uint8)
        dilated_label = cv2.dilate(label, kernel, iterations=1)
    
    # Identify where the dilated label differs from the original, indicating potential cluster edges
        potential_cluster_edges = dilated_label != label
    
    # Initialize cluster edge mask
        cluster_edge_mask = np.zeros(label.shape, dtype=np.uint8)
    
    # Iterate through potential cluster edges and validate against original label
        for y in range(label.shape[0]):
            for x in range(label.shape[1]):
                if potential_cluster_edges[y, x]:
                # Get the original and dilated values
                    original_value = label[y, x]
                    dilated_value = dilated_label[y, x]
                
                # If the dilated value is different from the original, and neither is background
                    if original_value != dilated_value and original_value != 0 and dilated_value != 0:
                    # Mark as cluster edge
                        cluster_edge_mask[y, x] = 1
        return cluster_edge_mask