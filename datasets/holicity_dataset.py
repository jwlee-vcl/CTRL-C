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

import datasets.transforms as T

def eul2rotm_ypr(euler):
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(euler[0]),-np.sin(euler[0])],
                    [0, np.sin(euler[0]), np.cos(euler[0])]], dtype=np.float32)
  
    R_y = np.array([[ np.cos(euler[1]), 0, np.sin(euler[1])],
                    [0, 1, 0 ],
                    [-np.sin(euler[1]), 0, np.cos(euler[1])]], dtype=np.float32)
  
    R_z = np.array([[np.cos(euler[2]),-np.sin(euler[2]), 0],
                    [np.sin(euler[2]), np.cos(euler[2]), 0],
                    [0, 0, 1]], dtype=np.float32)
                   
    return np.dot(R_z, np.dot(R_x, R_y))

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

class HoliCityDataset(Dataset):
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
        self.list_line_filename = []
        self.list_pitch = []
        self.list_roll = []
        self.list_fov = []
        #self.list_hvps = []

        with open(self.listpath) as txtfile:
            target_files = txtfile.read().split('\n')
            target_len = len(target_files[0])

        with open(osp.join(self.basepath, 'split/filelist.txt')) as txtfile:
            for filename in txtfile.read().split('\n'):
                if len(filename) == 0 or filename[:target_len] not in target_files:
                    continue
                camr = np.load(osp.join(self.basepath, 'geo/', filename + '_camr.npz'))
                vpts = np.load(osp.join(self.basepath, 'vpts/', filename + '_vpts.npz'))
                self.list_filename.append(filename)
                self.list_img_filename.append(osp.join(self.basepath, 'image/', filename + '_imag.jpg'))
                self.list_line_filename.append(osp.join(self.basepath, 'line/', filename + '_line.csv'))
                self.list_pitch.append(np.float32(camr['pitch']))
                self.list_roll.append(np.float32(0))
                self.list_fov.append(np.float32(np.radians(camr['fov'])))

                #img_filename  = self.basepath + row[0]
                #line_filename  = self.basepath + row[1]                
                #self.list_filename.append(row[0])
                #self.list_img_filename.append(img_filename)
                #self.list_line_filename.append(line_filename)
                #self.list_pitch.append(np.float32(row[3]))
                #self.list_roll.append(np.float32(row[4]))
                #self.list_focal.append(np.float32(row[5]))
                #self.list_hvps.append([[np.float32(row[6]),np.float32(row[7]), 1.0],
                #                       [np.float32(row[8]),np.float32(row[9]), 1.0]])
        
    def __getitem__(self, idx):
        target = {}
        extra = {}
        
        filename = self.list_filename[idx]
        # read image and preprocess
        img_filename = self.list_img_filename[idx]   
        line_filename = self.list_line_filename[idx]
                
        image = cv2.imread(img_filename)
        assert image is not None, print(img_filename)
        image = image[:,:,::-1] # convert to rgb
        
        org_image = image
        org_h, org_w = image.shape[0], image.shape[1]
        org_sz = np.array([org_h, org_w]) 
                
        image = cv2.resize(image, dsize=(self.input_width, self.input_height))
        input_sz = np.array([self.input_height, self.input_width])
        
        # preprocess
        ratio_x = float(self.input_width)/float(org_w)
        ratio_y = float(self.input_height)/float(org_h)

        pp = (org_w/2, org_h/2)
        rho = 2.0/np.minimum(org_w,org_h)
        
        # read line and preprocess        
        org_segs = read_line_file(line_filename, self.min_line_length)
        num_segs = len(org_segs)
        assert num_segs > 10, print(line_filename, num_segs)
        
#         line_map = np.zeros((self.input_width, self.input_height, 1), dtype=np.float32)
#         for seg in org_segs:            
#             seg = np.int32(np.array(seg) * [ratio_x, ratio_y, ratio_x, ratio_y])
#             line_map = cv2.line(line_map, (seg[0], seg[1]), (seg[2], seg[3]), 1.0, self.line_width)
        
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

        # preprocess GT data
        gt_pitch = np.radians(self.list_pitch[idx])
        gt_roll = np.radians(self.list_roll[idx])
        #gt_focal = rho*self.list_focal[idx]
        gt_focal = 1.0 / np.tan(0.5 * self.list_fov[idx])

        rotm = eul2rotm_ypr([gt_pitch, 0, gt_roll])
        rotm[1,:] = -rotm[1,:]
        gt_up_vector = rotm[:,1]
        
        gt_hl = gt_up_vector.copy()
        gt_hl[2] = gt_focal*gt_hl[2]

        gt_zvp = gt_up_vector.copy()
        if gt_zvp[2] < 0:
            gt_zvp = -gt_zvp
        gt_zvp = gt_zvp / np.maximum(gt_zvp[2], 1e-7)
        gt_zvp = gt_focal*gt_zvp
        gt_zvp[2] = 1.0
        
        gt_rp = np.array([gt_roll, gt_pitch]) 
        
        #gt_fovy = 2.0 * np.arctan(float(org_h)/(2.0*self.list_focal[idx]))
        gt_fovy = self.list_fov[idx]

        # horizon vps
        #gt_hvps = np.array(self.list_hvps[idx], dtype=np.float32)
        #gt_hvps[:,0] = rho*(gt_hvps[:,0] - pp[0])
        #gt_hvps[:,1] = rho*(gt_hvps[:,1] - pp[1])
                
        if self.return_masks:
            masks = create_masks(image)

        image = np.ascontiguousarray(image)
        target['rp'] = torch.from_numpy(np.ascontiguousarray(gt_rp)).contiguous().float()
        target['fovy'] = torch.from_numpy(np.ascontiguousarray(gt_fovy)).contiguous().float()
        target['up_vector'] = torch.from_numpy(np.ascontiguousarray(gt_up_vector)).contiguous().float()
        target['focal'] = torch.from_numpy(np.ascontiguousarray(gt_focal)).contiguous().float()
        target['zvp'] = torch.from_numpy(np.ascontiguousarray(gt_zvp)).contiguous().float()
        target['hl'] = torch.from_numpy(np.ascontiguousarray(gt_hl)).contiguous().float()
        #target['hvps'] = torch.from_numpy(np.ascontiguousarray(gt_hvps)).contiguous().float()
        
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

def build_holicity(image_set, cfg):
    root = '/workspace/holicity/'

    if image_set == 'test':
        ann_file = osp.join(root, 'split/test-randomsplit.txt')
    else:
        raise Exception('unsupported image set: {}'.format(image_set))

    dataset = HoliCityDataset(cfg, ann_file, root, 
                         return_masks=cfg.MODELS.MASKS, transform=make_transform())
    return dataset

