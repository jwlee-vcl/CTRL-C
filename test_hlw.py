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
from datasets import build_hlw_dataset
from models import build_model
from config import cfg

cmap = plt.get_cmap("jet")
norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

def c(x):
    return sm.to_rgba(x)

def get_args_parser():
    parser = argparse.ArgumentParser('Set gptran', add_help=False)
    parser.add_argument('--config-file', 
                        metavar="FILE",
                        help="path to config file",
                        type=str,
                        default='config-files/gptran.yaml')
    parser.add_argument("--opts",
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER
                        )
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

def compute_horizon_hlw(hl, sz, eps=1e-6):
    (a,b,c) = hl
    if b < 0:
        a, b, c = -a, -b, -c
    b = np.maximum(b, eps)

    scale = sz[1]/2
    left = np.array([-scale, (scale*a - c)/b])        
    right = np.array([scale, (-scale*a - c)/b])

    return [np.squeeze(left), np.squeeze(right)]

def normalize_safe_np(v, eps=1e-7):
    return v/np.maximum(LA.norm(v), eps)

def compute_horizon_error(pred_hl, target_hl, img_sz, crop_sz):
    target_hl_pts = compute_hl_np(target_hl, img_sz)
    pred_hl_pts = compute_hl_np(pred_hl, img_sz)
    err_hl = np.maximum(np.abs(target_hl_pts[0][1] - pred_hl_pts[0][1]),
                        np.abs(target_hl_pts[1][1] - pred_hl_pts[1][1]))
    err_hl /= img_sz[0] # height
    return err_hl

def to_device(data, device):
    if type(data) == dict:
        return {k: v.to(device) for k, v in data.items()}
    return [{k: v.to(device) if isinstance(v, torch.Tensor) else v
             for k, v in t.items()} for t in data]

def main(cfg):
    device = torch.device(cfg.DEVICE)
    
    model, _ = build_model(cfg)
    model.to(device)
    
    dataset_test = build_hlw_dataset(image_set='test', cfg=cfg)
    sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    data_loader_test = DataLoader(dataset_test, 1, sampler=sampler_test,
                                 drop_last=False, 
                                 collate_fn=utils.collate_fn, 
                                 num_workers=2)
    
    output_dir = Path(cfg.OUTPUT_DIR)
    
    checkpoint = torch.load('logs/checkpoint.pth', map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model = model.eval()
    
    # initlaize for visualization
    name = f'hlw_test_{date.today()}'
    if cfg.TEST.DISPLAY:
        fig_output_dir = osp.join(output_dir,'{}'.format(name))
        os.makedirs(fig_output_dir, exist_ok=True)
    
    csvpath = osp.join(output_dir,'{}.csv'.format(name))
    print('Writing the evaluation results of the {} dataset into {}'.format(name, csvpath))
    with open(csvpath, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['filename', 'hl-err', 'num_segs'])
        
    csvpath_hls = osp.join(output_dir,'{}_hls.csv'.format(name))
    print('Writing the results of the {} dataset into {}'.format(name, csvpath_hls))
    with open(csvpath_hls, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)    
    
    for i, (samples, extra_samples, targets) in enumerate(tqdm(data_loader_test)):
        with torch.no_grad():
            samples = samples.to(device)
            extra_samples = to_device(extra_samples, device)
            outputs, extra_info = model(samples, extra_samples)
        
            pred_zvp = outputs['pred_zvp'].to('cpu')[0].numpy()
            pred_fovy = outputs['pred_fovy'].to('cpu')[0].numpy()
            pred_hl = outputs['pred_hl'].to('cpu')[0].numpy()
                
            img_sz = targets[0]['org_sz']
            crop_sz = targets[0]['crop_sz']
            num_segs = targets[0]['num_segs']
            filename = targets[0]['filename']
            filename = osp.splitext(filename)[0]
                
            target_hl = targets[0]['hl'].numpy()
            
            # horizon line error
            target_hl_pts = compute_horizon_hlw(target_hl, img_sz)
            pred_hl_pts = compute_horizon(pred_hl, crop_sz, img_sz)
            
            err_hl = np.maximum(np.abs(target_hl_pts[0][1] - pred_hl_pts[0][1]),
                        np.abs(target_hl_pts[1][1] - pred_hl_pts[1][1]))
            err_hl /= img_sz[0] # height
                
            with open(csvpath, 'a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow([filename, err_hl, num_segs])
            
            with open(csvpath_hls, 'a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow([pred_hl_pts[0][0], pred_hl_pts[0][1],pred_hl_pts[1][0], pred_hl_pts[1][1]])
        
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser('GPANet training and evaluation script', 
                                     parents=[get_args_parser()])
    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    
    if cfg.OUTPUT_DIR:
        Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    main(cfg)