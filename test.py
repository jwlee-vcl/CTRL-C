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
from datasets import build_dataset
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

def compute_vp_err(vp1, vp2, dim=-1):
    cos_sim = F.cosine_similarity(vp1, vp2, dim=dim).abs()
    cos_sim = np.clip(cos_sim.item(), 0.0, 1.0)    
    return np.degrees(np.arccos(cos_sim))

def compute_hl_np(hl, sz, eps=1e-6):
    (a,b,c) = hl
    if b < 0:
        a, b, c = -a, -b, -c
    b = np.maximum(b, eps)
    
    left = np.array([-1.0, (a - c)/b])        
    right = np.array([1.0, (-a - c)/b])

    # scale back to original image    
    scale = sz[1]/2
    left = scale*left
    right = scale*right
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

def cosine_similarity(v1, v2, eps=1e-7):
    v1 = v1 / np.maximum(LA.norm(v1), eps)
    v2 = v2 / np.maximum(LA.norm(v2), eps)
    return np.sum(v1*v2)

def normalize_safe_np(v, eps=1e-7):
    return v/np.maximum(LA.norm(v), eps)

def compute_up_vector_error(pred_zvp, pred_fovy, target_up_vector):
    pred_up_vector = compute_up_vector(pred_zvp, pred_fovy)
    cos_sim = cosine_similarity(target_up_vector, pred_up_vector)

    target_pitch, target_roll = decompose_up_vector(target_up_vector)

    if cos_sim < 0:
        pred_pitch, pred_roll = decompose_up_vector(-pred_up_vector)
    else:
        pred_pitch, pred_roll = decompose_up_vector(pred_up_vector)

    err_angle = np.degrees(np.arccos(np.clip(np.abs(cos_sim),0.0, 1.0)))
    err_pitch = np.degrees(np.abs(pred_pitch - target_pitch))
    err_roll = np.degrees(np.abs(pred_roll - target_roll))
    return err_angle, err_pitch, err_roll

def compute_fovy_error(pred_fovy, target_fovy):
    pred_fovy = np.degrees(pred_fovy)
    target_fovy = np.degrees(target_fovy)
    err_fovy = np.abs(pred_fovy - target_fovy)
    return err_fovy.item()

def compute_horizon_error(pred_hl, target_hl, img_sz):
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
    
    dataset_test = build_dataset(image_set='test', cfg=cfg)
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
    name = f'gsv_test_{date.today()}'
    if cfg.TEST.DISPLAY:
        fig_output_dir = osp.join(output_dir,'{}'.format(name))
        os.makedirs(fig_output_dir, exist_ok=True)
    
    csvpath = osp.join(output_dir,'{}.csv'.format(name))
    print('Writing the evaluation results of the {} dataset into {}'.format(name, csvpath))
    with open(csvpath, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['filename',                            
                            'angle-err', 'pitch-err', 'roll-err', 'fovy-err', 'hl-err',
                            'angle-err0', 'pitch-err0', 'roll-err0', 'fovy-err0', 'hl-err0',
                            'angle-err1', 'pitch-err1', 'roll-err1', 'fovy-err1', 'hl-err1',
                            'angle-err2', 'pitch-err2', 'roll-err2', 'fovy-err2', 'hl-err2',
                            'angle-err3', 'pitch-err3', 'roll-err3', 'fovy-err3', 'hl-err3',
                            'angle-err4', 'pitch-err4', 'roll-err4', 'fovy-err4', 'hl-err4'])
        
    hlspath = osp.join(output_dir,'{}_hls_GoogleStreetView.csv'.format(name))
    print('Writing the evaluation results of the {} dataset into {}'.format(name, hlspath))
    with open(hlspath, 'w', newline='') as csvfile:
        pass # initialize file
    
    for i, (samples, extra_samples, targets) in enumerate(tqdm(data_loader_test)):
        with torch.no_grad():
            samples = samples.to(device)
            extra_samples = to_device(extra_samples, device)
            outputs, extra_info = model(samples, extra_samples)
        
            pred_zvp = outputs['pred_zvp'].to('cpu')[0].numpy()
            pred_fovy = outputs['pred_fovy'].to('cpu')[0].numpy()
            pred_hl = outputs['pred_hl'].to('cpu')[0].numpy()
            
#             pred_vweight = outputs['pred_vline_logits'].sigmoid()
#             pred_vweight = pred_vweight.to('cpu')[0].numpy()
#             pred_vweight = np.squeeze(pred_vweight, axis=1)
            
#             pred_hweight = outputs['pred_hline_logits'].sigmoid()
#             pred_hweight = pred_hweight.to('cpu')[0].numpy()
#             pred_hweight = np.squeeze(pred_hweight, axis=1)
            
            aux_outputs = outputs['aux_outputs']
    
            img_sz = targets[0]['org_sz']
            filename = targets[0]['filename']
            filename = osp.splitext(filename)[0]
                
            target_up_vector = targets[0]['up_vector'].numpy()
            target_zvp = targets[0]['zvp'].numpy()
            target_fovy = targets[0]['fovy'].numpy()
            target_hl = targets[0]['hl'].numpy()
                
            segs = targets[0]['segs'].cpu().numpy()
            line_mask = targets[0]['line_mask'].cpu().numpy()
            line_mask = np.squeeze(line_mask, axis=1)
            
            target_zvp = normalize_safe_np(target_zvp)                
            
            pred_fovy = np.squeeze(pred_fovy)
            target_fovy = np.squeeze(target_fovy)
            
            # up vector error
            err_angle, err_pitch, err_roll = (
                compute_up_vector_error(pred_zvp, pred_fovy, target_up_vector))
            
            # fovy error
            err_fovy = compute_fovy_error(pred_fovy, target_fovy)
            
            # horizon line error
            err_hl = compute_horizon_error(pred_hl, target_hl, img_sz)
            
#             pred_hl_pts = compute_hl_np(pred_hl, img_sz)
            
            list_err_angle = []
            list_err_pitch = []
            list_err_roll = []
            list_err_fovy = []
            list_err_hl = []
            
            aux_pred_zvps = []
            aux_pred_fovys = []
            aux_pred_hls = []
            
            for aux_output in aux_outputs:
                aux_pred_zvp = aux_output['pred_zvp'].to('cpu')[0].numpy()
                aux_pred_fovy = aux_output['pred_fovy'].to('cpu')[0].numpy()
                aux_pred_hl = aux_output['pred_hl'].to('cpu')[0].numpy()
                
                if aux_pred_zvp[2] < 0:
                    aux_pred_zvp = -aux_pred_zvp
                
                err_angle, err_pitch, err_roll = (
                    compute_up_vector_error(aux_pred_zvp, aux_pred_fovy, target_up_vector))            
                
                err_fovy = compute_fovy_error(aux_pred_fovy, target_fovy)

                err_hl = compute_horizon_error(aux_pred_hl, target_hl, img_sz)
                
                aux_pred_zvps.append(aux_pred_zvp)
                aux_pred_fovys.append(aux_pred_fovys)
                aux_pred_hls.append(aux_pred_hl)
                
                list_err_angle.append(err_angle)
                list_err_pitch.append(err_pitch)
                list_err_roll.append(err_roll)
                list_err_fovy.append(err_fovy)
                list_err_hl.append(err_hl)
            
            with open(csvpath, 'a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow([filename,                                    
                                    err_angle, err_pitch, err_roll, err_fovy, err_hl,
                                    list_err_angle[0], list_err_pitch[0], list_err_roll[0], list_err_fovy[0], list_err_hl[0],
                                    list_err_angle[1], list_err_pitch[1], list_err_roll[1], list_err_fovy[1], list_err_hl[1],
                                    list_err_angle[2], list_err_pitch[2], list_err_roll[2], list_err_fovy[2], list_err_hl[2],
                                    list_err_angle[3], list_err_pitch[3], list_err_roll[3], list_err_fovy[3], list_err_hl[3],
                                    list_err_angle[4], list_err_pitch[4], list_err_roll[4], list_err_fovy[4], list_err_hl[4]])
            
            
#             with open(hlspath, 'a', newline='') as csvfile:
#                 csvwriter = csv.writer(csvfile)
#                 csvwriter.writerow([pred_hl_pts[0][0], pred_hl_pts[0][1],
#                                     pred_hl_pts[1][0], pred_hl_pts[1][1]])

            
if __name__ == '__main__':
    parser = argparse.ArgumentParser('GPANet training and evaluation script', 
                                     parents=[get_args_parser()])
    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    
    if cfg.OUTPUT_DIR:
        Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    main(cfg)