import os
import os.path as osp
import argparse
from datetime import date
import json
import random
import time
from pathlib import Path
import numpy as np
import numpy.linalg as LA
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
import cv2
import csv

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import datasets
import util.misc as utils
from datasets import build_image_dataset
from models import build_model
from config import cfg

def get_args_parser():
    parser = argparse.ArgumentParser('Set CTRL-C', add_help=False)
    parser.add_argument('--config-file', 
                        metavar="FILE",
                        help="path to config file",
                        type=str,
                        default='config-files/ctrl-c.yaml')
    parser.add_argument('--sample', default='sample.jpg')
    parser.add_argument("--opts",
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    return parser

def extract_hl(left, right, width):
    hl_homo = np.cross(np.append(left, 1), np.append(right, 1))
    hl_left_homo = np.cross(hl_homo, [-1, 0, -width/2]);
    hl_left = hl_left_homo[0:2]/hl_left_homo[-1];
    hl_right_homo = np.cross(hl_homo, [-1, 0, width/2]);
    hl_right = hl_right_homo[0:2]/hl_right_homo[-1];
    return hl_left, hl_right


def compute_horizon(hl, crop_sz, org_sz, eps=1e-6):
    a,b,c = hl
    if b < 0:
        a, b, c = -hl
    b = np.maximum(b, eps)    
    left = (a - c)/b
    right = (-a - c)/b

    c_left = left*(crop_sz[0]/2)
    c_right = right*(crop_sz[0]/2)

    left_tmp = np.asarray([-crop_sz[1]/2, c_left])
    right_tmp = np.asarray([crop_sz[1]/2, c_right])
    left, right = extract_hl(left_tmp, right_tmp, org_sz[1])

    return [np.squeeze(left), np.squeeze(right)]

def compute_up_vector(zvp, fovy, eps=1e-7):
    # image size 2 (-1~1)
    focal = 1.0/np.tan(fovy/2.0)
    
    if zvp[2] < 0:
        zvp = -zvp
    zvp = zvp / np.maximum(zvp[2], eps)
    zvp[2] = focal
    return normalize_safe_np(zvp)

def decompose_up_vector(v):
    pitch = np.arcsin(v[2])
    roll = np.arctan(-v[0]/v[1])
    return pitch, roll

def normalize_safe_np(v, eps=1e-7):
    return v/np.maximum(LA.norm(v), eps)

def to_device(data, device):
    if type(data) == dict:
        return {k: v.to(device) for k, v in data.items()}
    return [{k: v.to(device) if isinstance(v, torch.Tensor) else v
             for k, v in t.items()} for t in data]

def main(cfg, sample_path):
    device = torch.device(cfg.DEVICE)
    
    model, _ = build_model(cfg)
    model.to(device)
    
    checkpoint = torch.load('logs/checkpoint.pth', map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model = model.eval()
        
    dataset_test = build_image_dataset(image_set=sample_path, cfg=cfg)
    sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    data_loader_test = DataLoader(dataset_test, 1, sampler=sampler_test,
                                 drop_last=False, 
                                 collate_fn=utils.collate_fn, 
                                 num_workers=1)
    
    for i, (samples, extra_samples, targets) in enumerate(tqdm(data_loader_test)):
        with torch.no_grad():
            samples = samples.to(device)
            extra_samples = to_device(extra_samples, device)
            outputs, extra_info = model(samples, extra_samples)

            zvp = outputs['pred_zvp'].to('cpu')[0].numpy()
            fovy = outputs['pred_fovy'].to('cpu')[0].numpy()
            hl = outputs['pred_hl'].to('cpu')[0].numpy()

            img_sz = targets[0]['org_sz']
            crop_sz = targets[0]['crop_sz']
            filename = targets[0]['filename']
            filename = osp.splitext(filename)[0]

            segs = targets[0]['segs'].cpu().numpy()
            line_mask = targets[0]['line_mask'].cpu().numpy()
            line_mask = np.squeeze(line_mask, axis=1)

            hl_pts = compute_horizon(hl, crop_sz, img_sz)

            img = targets[0]['org_img']
            extent=[-img_sz[1]/2, img_sz[1]/2, img_sz[0]/2, -img_sz[0]/2]
            
            plt.figure(figsize=(5,5))                
            plt.imshow(img, extent=extent)
            plt.plot([hl_pts[0][0], hl_pts[1][0]], 
                     [hl_pts[0][1], hl_pts[1][1]], 'r-', alpha=1.0)
            plt.xlim(-img_sz[1]/2, img_sz[1]/2)
            plt.ylim( img_sz[0]/2,-img_sz[0]/2)
            plt.axis('off')
            plt.savefig(filename+'_hl.jpg',pad_inches=0, bbox_inches='tight')
            plt.close('all')
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser('CTRL-C test script', parents=[get_args_parser()])
    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    
    if cfg.OUTPUT_DIR:
        Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    main(cfg, args.sample)