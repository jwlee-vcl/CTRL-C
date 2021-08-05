import os
import os.path as osp

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from torchvision.transforms import functional as F

import numpy as np
import numpy.linalg as LA
import cv2
import json
import csv
import matplotlib.pyplot as plt
import scipy.io as sio

import datasets.transforms as T

from pylsd import lsd

import glob

def center_crop(img):
    sz = img.shape[0:2]
    side_length = np.min(sz)
    if sz[0] > sz[1]:
        ul_x = 0  
        ul_y = int(np.floor((sz[0]/2) - (side_length/2)))
        x_inds = [ul_x, sz[1]-1]
        y_inds = [ul_y, ul_y + side_length - 1]
    else:
        ul_x = int(np.floor((sz[1]/2) - (side_length/2)))
        ul_y = 0
        x_inds = [ul_x, ul_x + side_length - 1]
        y_inds = [ul_y, sz[0]-1]

    c_img = img[y_inds[0]:y_inds[1]+1, x_inds[0]:x_inds[1]+1, :]

    return c_img

def create_masks(image):
    masks = torch.zeros((1, height, width), dtype=torch.uint8)
    return masks

def read_line_file(filename, min_line_length=10):
    segs = [] # line segments
    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            segs.append([float(row[0]), float(row[1]), 
                         float(row[2]), float(row[3])])
    segs = np.array(segs, dtype=np.float32)
    lengths = LA.norm(segs[:,2:] - segs[:,:2], axis=1)
    segs = segs[lengths > min_line_length]
    return segs

def filter_length(segs, min_line_length=10):
    lengths = LA.norm(segs[:,2:4] - segs[:,:2], axis=1)
    segs = segs[lengths > min_line_length]
    return segs[:,:4]

def normalize_segs(segs, pp, rho):    
    pp = np.array([pp[0], pp[1], pp[0], pp[1]], dtype=np.float32)    
    return rho*(segs - pp)    

def normalize_safe_np(v, axis=-1, eps=1e-6):
    de = LA.norm(v, axis=axis, keepdims=True)
    de = np.maximum(de, eps)
    return v/de

def segs2lines_np(segs):
    ones = np.ones(len(segs))
    ones = np.expand_dims(ones, axis=-1)
    p1 = np.concatenate([segs[:,:2], ones], axis=-1)
    p2 = np.concatenate([segs[:,2:], ones], axis=-1)
    lines = np.cross(p1, p2)
    return normalize_safe_np(lines)

def sample_segs_np(segs, num_sample, use_prob=True):    
    num_segs = len(segs)
    sampled_segs = np.zeros([num_sample, 4], dtype=np.float32)
    mask = np.zeros([num_sample, 1], dtype=np.float32)
    if num_sample > num_segs:
        sampled_segs[:num_segs] = segs
        mask[:num_segs] = np.ones([num_segs, 1], dtype=np.float32)
    else:    
        lengths = LA.norm(segs[:,2:] - segs[:,:2], axis=-1)
        prob = lengths/np.sum(lengths)        
        idxs = np.random.choice(segs.shape[0], num_sample, replace=True, p=prob)
        sampled_segs = segs[idxs]
        mask = np.ones([num_sample, 1], dtype=np.float32)
    return sampled_segs, mask

def sample_vert_segs_np(segs, thresh_theta=22.5):    
    lines = segs2lines_np(segs)
    (a,b) = lines[:,0],lines[:,1]
    theta = np.arctan2(np.abs(b),np.abs(a))
    thresh_theta = np.radians(thresh_theta)
    return segs[theta < thresh_theta]

def decompose_up_vector(v):
    x,y,z = v
    pitch = np.arcsin(z)
    roll = np.arctan(-x/y)
    return pitch, roll

class KITTIDataset(Dataset):
    def __init__(self, cfg, listpath, basepath, return_masks=False, transform=None):
        self.listpath = listpath
        self.basepath = basepath
        self.input_width = cfg.DATASETS.INPUT_WIDTH
        self.input_height = cfg.DATASETS.INPUT_HEIGHT
        self.min_line_length = cfg.DATASETS.MIN_LINE_LENGTH
        self.num_input_lines = cfg.DATASETS.NUM_INPUT_LINES
        self.num_input_vert_lines = cfg.DATASETS.NUM_INPUT_VERT_LINE
        self.vert_line_angle = cfg.DATASETS.VERT_LINE_ANGLE
        self.return_vert_lines = cfg.DATASETS.RETURN_VERT_LINES
        self.return_masks = return_masks
        self.transform = transform
        
        self.list_filename = []
        self.list_img_filename = []
        self.list_focal = []
        self.list_rot = []
                
        with open(self.listpath, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            for row in csvreader:                
                img_filename = self.basepath + row[0].replace('\\', '/')
                self.list_filename.append(row[0])
                self.list_img_filename.append(img_filename)
                self.list_focal.append(np.float32(row[1]))
                self.list_rot.append(np.float32(row[2:11]))
                
    def __getitem__(self, idx):
        target = {}
        extra = {}
        
        filename = self.list_filename[idx]
        # read image and preprocess
        img_filename = self.list_img_filename[idx]   
                                
        image = cv2.imread(img_filename)
        assert image is not None, print(img_filename)
        image = image[:,:,::-1] # convert to rgb
        
        org_image = image
        org_h, org_w = image.shape[0], image.shape[1]
        org_sz = np.array([org_h, org_w]) 
        crop_sz = np.array([org_h, org_w]) 
                
        image = cv2.resize(org_image, dsize=(self.input_width, self.input_height))
        input_sz = np.array([self.input_height, self.input_width])
        
        # preprocess
        ratio_x = float(self.input_width)/float(org_w)
        ratio_y = float(self.input_height)/float(org_h)

        pp = np.array((org_w/2, org_h/2))
        rho = 2.0/np.minimum(org_w,org_h)
        
        # read line and preprocess      
        gray = cv2.cvtColor(org_image, cv2.COLOR_BGR2GRAY)
        org_segs = lsd(gray, scale=0.5)
        org_segs = filter_length(org_segs, self.min_line_length)
        #assert len(org_segs) > 10, print(line_filename, len(org_segs))
        num_segs = len(org_segs)
        assert num_segs > 10, print(line_filename, num_segs)
       
        segs = normalize_segs(org_segs, pp=pp, rho=rho)
        
        # whole segs
        sampled_segs, line_mask = sample_segs_np(
            segs, self.num_input_lines)
        sampled_lines = segs2lines_np(sampled_segs)
        
        # vertical directional segs
        vert_segs = sample_vert_segs_np(segs, thresh_theta=self.vert_line_angle)
        if len(vert_segs) < 2:
            vert_segs = segs
            
        sampled_vert_segs, vert_line_mask = sample_segs_np(
            vert_segs, self.num_input_vert_lines)
        sampled_vert_lines = segs2lines_np(sampled_vert_segs)
        
        # gt data
        rotm = self.list_rot[idx].reshape(3,3).T
        rotm[1,:] = -rotm[1,:]
        gt_up_vector = rotm[:,1]

        gt_pitch, gt_roll = decompose_up_vector(gt_up_vector)
        gt_focal = rho*self.list_focal[idx]
        
        gt_hl = gt_up_vector.copy()
        gt_hl[2] = gt_focal*gt_hl[2]
        
        gt_zvp = gt_up_vector.copy()
        if gt_zvp[2] < 0:
            gt_zvp = -gt_zvp
        gt_zvp = gt_zvp / np.maximum(gt_zvp[2], 1e-7)
        gt_zvp = gt_focal*gt_zvp
        gt_zvp[2] = 1.0
        
        gt_rp = np.array([gt_roll, gt_pitch]) 
        
        gt_fovy = 2.0 * np.arctan(float(org_h)/(2.0*self.list_focal[idx]))
                        
        if self.return_masks:
            masks = create_masks(image)

        image = np.ascontiguousarray(image)
        target['rp'] = torch.from_numpy(np.ascontiguousarray(gt_rp)).contiguous().float()
        target['fovy'] = torch.from_numpy(np.ascontiguousarray(gt_fovy)).contiguous().float()
        target['up_vector'] = torch.from_numpy(np.ascontiguousarray(gt_up_vector)).contiguous().float()
        target['focal'] = torch.from_numpy(np.ascontiguousarray(gt_focal)).contiguous().float()
        target['zvp'] = torch.from_numpy(np.ascontiguousarray(gt_zvp)).contiguous().float()
        target['hl'] = torch.from_numpy(np.ascontiguousarray(gt_hl)).contiguous().float()
                
        if self.return_vert_lines:
            target['segs'] = torch.from_numpy(np.ascontiguousarray(sampled_vert_segs)).contiguous().float()
            target['lines'] = torch.from_numpy(np.ascontiguousarray(sampled_vert_lines)).contiguous().float()
            target['line_mask'] = torch.from_numpy(np.ascontiguousarray(vert_line_mask)).contiguous().float()
        else:
            target['segs'] = torch.from_numpy(np.ascontiguousarray(sampled_segs)).contiguous().float()
            target['lines'] = torch.from_numpy(np.ascontiguousarray(sampled_lines)).contiguous().float()
            target['line_mask'] = torch.from_numpy(np.ascontiguousarray(line_mask)).contiguous().float()
                    
        if self.return_masks:
            target['masks'] = masks
        target['org_img'] = org_image
        target['org_sz'] = org_sz
        target['crop_sz'] = crop_sz
        target['input_sz'] = input_sz
        target['img_path'] = img_filename
        target['filename'] = filename
        target['num_segs'] = num_segs
        
        extra['lines'] = target['lines'].clone()
        extra['line_mask'] = target['line_mask'].clone()

        return self.transform(image, extra, target)
    
    def __len__(self):
        return len(self.list_img_filename)   

def make_transform():
    return T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]) 

def build_kitti(image_set, cfg):    
    root = '/data/public/rw/kitti/20200714/'
    
    PATHS = {
        "train": 'kitti_train_20200714.csv',
        "val":   'kitti_val_20200714.csv',
        "test":  'kitti_test_20200714.csv',
    }

    img_folder = root
    ann_file = PATHS[image_set]    
    dataset = KITTIDataset(cfg, ann_file, img_folder, 
                           return_masks=cfg.MODELS.MASKS, transform=make_transform())
    return dataset