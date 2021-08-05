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

def draw_attention(img, weights, extent, cmap, savepath):
    num_layer = len(weights)
    plt.figure(figsize=(num_layer*3,3))
    for idx_l in range(num_layer):                    
        plt.subplot(1, num_layer, idx_l + 1)
        plt.imshow(img, extent=extent)
        plt.imshow(weights[idx_l], cmap=cmap, alpha=0.3, extent=[-1, 1, 1, -1])
        plt.xlim(extent[0], extent[1])
        plt.ylim(extent[2], extent[3])
        plt.axis('off')
    plt.savefig(savepath, pad_inches=0, bbox_inches='tight')
    plt.close('all')

def draw_attention_segs(img, weights, segs, extent, cmap, savepath):
    num_layer = len(weights)
    num_segs = len(segs)
    plt.figure(figsize=(num_layer*3,3))
    for idx_l in range(num_layer):                    
        plt.subplot(1, num_layer, idx_l + 1)
        plt.imshow(img, extent=extent)                 
        ws = weights[idx_l]
        ws = (ws - ws.min())/(ws.max() - ws.min())
        for idx_s in range(num_segs):
            plt.plot((segs[idx_s,0], segs[idx_s,2]), 
                     (segs[idx_s,1], segs[idx_s,3]), c=cmap(ws[idx_s]))
        plt.xlim(extent[0], extent[1])
        plt.ylim(extent[2], extent[3])
        plt.axis('off')
    plt.savefig(savepath, pad_inches=0, bbox_inches='tight')
    plt.close('all') 

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
    
    for i, (samples, extra_samples, targets) in enumerate(tqdm(data_loader_test)):
        with torch.no_grad():
            samples = samples.to(device)
            extra_samples = to_device(extra_samples, device)
            outputs, extra_info = model(samples, extra_samples)
        
            pred_zvp = outputs['pred_zvp'].to('cpu')[0].numpy()
            pred_fovy = outputs['pred_fovy'].to('cpu')[0].numpy()
            pred_hl = outputs['pred_hl'].to('cpu')[0].numpy()
            
            pred_vweight = outputs['pred_vline_logits'].sigmoid()
            pred_vweight = pred_vweight.to('cpu')[0].numpy()
            pred_vweight = np.squeeze(pred_vweight, axis=1)
            
            pred_hweight = outputs['pred_hline_logits'].sigmoid()
            pred_hweight = pred_hweight.to('cpu')[0].numpy()
            pred_hweight = np.squeeze(pred_hweight, axis=1)
                
            img_sz = targets[0]['org_sz']
            crop_sz = targets[0]['crop_sz']
            filename = targets[0]['filename']
            filename = osp.splitext(filename)[0]
                
            target_hl = targets[0]['hl'].numpy()
                
            target_segs = targets[0]['segs'].cpu().numpy()
            target_mask = targets[0]['line_mask'].cpu().numpy()
            target_mask = np.squeeze(target_mask, axis=1)
            
            # horizon line error
            target_hl_pts = compute_horizon_hlw(target_hl, img_sz)
            pred_hl_pts = compute_horizon(pred_hl, crop_sz, img_sz)
            
            err_hl = np.maximum(np.abs(target_hl_pts[0][1] - pred_hl_pts[0][1]),
                        np.abs(target_hl_pts[1][1] - pred_hl_pts[1][1]))
            err_hl /= img_sz[0] # height
                
            if cfg.TEST.DISPLAY:
                os.makedirs(osp.dirname(osp.join(fig_output_dir, filename)), 
                                exist_ok=True)
                img = targets[0]['org_img']
                lowtone_img = np.array(img)//4 + 190
                h,w = lowtone_img.shape[:2]

                min_sz = np.min(img_sz)
                extent = [-img_sz[1]/min_sz,  img_sz[1]/min_sz, 
                           img_sz[0]/min_sz, -img_sz[0]/min_sz] # l, r, b, t
                                
                plt.figure(figsize=(5,5))
                
                plt.imshow(lowtone_img, 
                        extent=[-img_sz[1]/2, img_sz[1]/2, img_sz[0]/2, -img_sz[0]/2])
                plt.plot([target_hl_pts[0][0], target_hl_pts[1][0]], 
                        [target_hl_pts[0][1], target_hl_pts[1][1]], 'r-', alpha=1.0)
                plt.plot([pred_hl_pts[0][0], pred_hl_pts[1][0]], 
                        [pred_hl_pts[0][1], pred_hl_pts[1][1]], 'g-', alpha=1.0)
                plt.xlim(-img_sz[1]/2, img_sz[1]/2)
                plt.ylim( img_sz[0]/2, -img_sz[0]/2)
                plt.axis('off')
                plt.savefig(osp.join(fig_output_dir, filename+'_hl.jpg'),  
                            pad_inches=0, bbox_inches='tight')
                plt.close('all')
                
                # visualize attention map
                # encoder attentions
                enc_attns = extra_info['enc_attns'][0].to('cpu').numpy()
              
                points = [[4,4],[4,12],[8,8],[12,4],[12,12]]
                #import pdb; pdb.set_trace()
                for pt in points:                    
                    draw_attention(lowtone_img,
                               enc_attns[:,pt[0],pt[1]], extent=extent, 
                               cmap='jet', 
                         savepath=osp.join(fig_output_dir, f'{filename}_attn_enc_{pt[0]}_{pt[1]}.jpg'))
                
                # decoder attentions
                num_segs = int(target_mask.sum())
                segs = target_segs[:num_segs]
                
                dec_attns = extra_info['dec_self_attns'][0].to('cpu').numpy()                
                dec_attns_zvp = dec_attns[:,0,3:] 
                dec_attns_fovy = dec_attns[:,1,3:] 
                dec_attns_hl = dec_attns[:,2,3:]
                
                # zvp attention
                weights = dec_attns_zvp[:num_segs]                                
                draw_attention_segs(lowtone_img, 
                                    weights=weights, segs=segs, extent=extent, 
                                    cmap=c, 
                savepath=osp.join(fig_output_dir, filename+'_zvp_self_attn.jpg'))
                                
                # fovy attention
                weights = dec_attns_fovy[:num_segs]                
                draw_attention_segs(lowtone_img, 
                                    weights=weights, segs=segs, extent=extent,
                                    cmap=c, 
                savepath=osp.join(fig_output_dir, filename+'_fovy_self_attn.jpg'))
                
                # hl attention
                weights = dec_attns_hl[:num_segs]
                draw_attention_segs(lowtone_img, 
                                    weights=weights, segs=segs, extent=extent,
                                    cmap=c, 
                savepath=osp.join(fig_output_dir, filename+'_hl_self_attn.jpg'))
                
                
                dec_attns = extra_info['dec_cross_attns'][0].to('cpu').numpy()
                dec_attns_zvp = dec_attns[:,0] 
                dec_attns_fovy = dec_attns[:,1] 
                dec_attns_hl = dec_attns[:,2] 
                
                # zvp attention
                draw_attention(lowtone_img, dec_attns_zvp, extent=extent, cmap='jet',
                         savepath=osp.join(fig_output_dir, filename+'_zvp_cross.jpg'))
                                
                # fovy attention
                draw_attention(lowtone_img, dec_attns_fovy, extent=extent, cmap='jet',
                         savepath=osp.join(fig_output_dir, filename+'_fovy_cross.jpg'))
                
                # hl attention
                draw_attention(lowtone_img, dec_attns_hl, extent=extent, cmap='jet',
                         savepath=osp.join(fig_output_dir, filename+'_hl_cross.jpg'))

                # visualize line weights
                num_segs = int(target_mask.sum())
                segs = target_segs
                                
                vw = pred_vweight
                hw = pred_hweight
                
                plt.figure(figsize=(10,5))
                plt.subplot(1,2,1)
                plt.title('zenith vp lines')
                plt.imshow(lowtone_img, extent=extent)
                for i in range(num_segs):
                    plt.plot((segs[i,0], segs[i,2]), (segs[i,1], segs[i,3]), 
                             c=c(vw[i]), alpha=1.0)
                plt.xlim(extent[0], extent[1])
                plt.ylim(extent[2], extent[3])
                plt.axis('off')
                
                plt.subplot(1,2,2)
                plt.title('horizon vps lines')
                plt.imshow(lowtone_img, extent=extent)
                for i in range(num_segs):
                    plt.plot((segs[i,0], segs[i,2]), (segs[i,1], segs[i,3]), 
                             c=c(hw[i]), alpha=1.0)
                plt.xlim(extent[0], extent[1])
                plt.ylim(extent[2], extent[3])
                plt.axis('off')
                plt.savefig(osp.join(fig_output_dir, filename+'_lines.jpg'),  
                            pad_inches=0, bbox_inches='tight')
                plt.close('all')
        
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser('GPANet training and evaluation script', 
                                     parents=[get_args_parser()])
    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    
    if cfg.OUTPUT_DIR:
        Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    main(cfg)