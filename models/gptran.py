# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from .backbone import build_backbone
from .transformer import build_transformer


class GPTran(nn.Module):    
    def __init__(self, backbone, transformer, num_queries, 
                 aux_loss=False, use_structure_tensor=True):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        
        self.use_structure_tensor = use_structure_tensor

        hidden_dim = transformer.d_model
        self.zvp_embed = nn.Linear(hidden_dim, 3)
        self.fovy_embed = nn.Linear(hidden_dim, 1)
        self.hl_embed = nn.Linear(hidden_dim, 3)
        self.vline_class_embed = nn.Linear(hidden_dim, 1)
        self.hline_class_embed = nn.Linear(hidden_dim, 1)
        
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)        
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        line_dim = 3
        if self.use_structure_tensor:
            line_dim = 6        
        self.input_line_proj = nn.Linear(line_dim, hidden_dim)        
        self.backbone = backbone
        self.aux_loss = aux_loss   

    def forward(self, samples: NestedTensor, extra_samples):
#     def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
           - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
           - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels            
        """
        extra_info = {}
        #import pdb; pdb.set_trace()
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None
        lines = extra_samples['lines']
        lmask = ~extra_samples['line_mask'].squeeze(2).bool()
        
        # vlines [bs, n, 3]
        if self.use_structure_tensor:
            lines = self._to_structure_tensor(lines)
                
        hs, memory, enc_attn, dec_self_attn, dec_cross_attn = (
            self.transformer(src=self.input_proj(src), mask=mask,
                             query_embed=self.query_embed.weight,
                             tgt=self.input_line_proj(lines), 
                             tgt_key_padding_mask=lmask,
                             pos_embed=pos[-1]))
        # ha [n_dec_layer, bs, num_query, ch]

        extra_info['enc_attns'] = enc_attn
        extra_info['dec_self_attns'] = dec_self_attn
        extra_info['dec_cross_attns'] = dec_cross_attn

        outputs_zvp = self.zvp_embed(hs[:,:,0,:]) # [n_dec_layer, bs, 3]
        outputs_zvp = F.normalize(outputs_zvp, p=2, dim=-1)  

        outputs_fovy = self.fovy_embed(hs[:,:,1,:]) # [n_dec_layer, bs, 1]
        outputs_fovy = outputs_fovy.sigmoid()*np.pi # 0 ~ 180

        outputs_hl = self.hl_embed(hs[:,:,2,:]) # [n_dec_layer, bs, 3]
        outputs_hl = F.normalize(outputs_hl, p=2, dim=-1)

        outputs_vline_class = self.vline_class_embed(hs[:,:,3:,:])
        outputs_hline_class = self.hline_class_embed(hs[:,:,3:,:])

        # import pdb; pdb.set_trace()

        out = {
            'pred_zvp': outputs_zvp[-1], 
            'pred_fovy': outputs_fovy[-1],
            'pred_hl': outputs_hl[-1],
            'pred_vline_logits': outputs_vline_class[-1], # [bs, n, 1]
            'pred_hline_logits': outputs_hline_class[-1], # [bs, n, 1]
        }
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_zvp, 
                                                    outputs_fovy, 
                                                    outputs_hl, 
                                                    outputs_vline_class,
                                                    outputs_hline_class)
        return out, extra_info

    @torch.jit.unused
    def _set_aux_loss(self, outputs_zvp, outputs_fovy, outputs_hl, 
                      outputs_vline_class, outputs_hline_class):
        return [{'pred_zvp': a, 'pred_fovy': b, 'pred_hl': c, 
                 'pred_vline_logits': d, 'pred_hline_logits': e}
                for a, b, c, d, e in zip(outputs_zvp[:-1], 
                                      outputs_fovy[:-1], 
                                      outputs_hl[:-1], 
                                      outputs_vline_class[:-1],
                                      outputs_hline_class[:-1])]

    def _to_structure_tensor(self, params):    
        (a,b,c) = torch.unbind(params, dim=-1)
        return torch.stack([a*a, a*b,
                            b*b, b*c,
                            c*c, c*a], dim=-1)
    
    def _evaluate_whls_zvp(self, weights, vlines):
        vlines = F.normalize(vlines, p=2, dim=-1)
        u, s, v = torch.svd(weights * vlines)
        return v[:, :, :, -1]
    
class SetCriterion(nn.Module):
    def __init__(self, weight_dict, losses, 
                       line_pos_angle, line_neg_angle):
        super().__init__()
        self.weight_dict = weight_dict        
        self.losses = losses
        self.thresh_line_pos = np.cos(np.radians(line_pos_angle), dtype=np.float32) # near 0.0
        self.thresh_line_neg = np.cos(np.radians(line_neg_angle), dtype=np.float32) # near 0.0
        # 
        
    def loss_zvp(self, outputs, targets, **kwargs):
        assert 'pred_zvp' in outputs
        src_zvp = outputs['pred_zvp']                                    
        target_zvp = torch.stack([t['zvp'] for t in targets], dim=0)

        cos_sim = F.cosine_similarity(src_zvp, target_zvp, dim=-1).abs()      
        loss_zvp_cos = (1.0 - cos_sim).mean()
                
        losses = {'loss_zvp': loss_zvp_cos}
        return losses

    def loss_fovy(self, outputs, targets, **kwargs):
        assert 'pred_fovy' in outputs
        src_fovy = outputs['pred_fovy']                                    
        target_fovy = torch.stack([t['fovy'] for t in targets], dim=0)
        
        loss_fovy_mae = F.l1_loss(src_fovy, target_fovy)
                        
        losses = {'loss_fovy': loss_fovy_mae}
        return losses

    def compute_hl(self, hl, dim=-1, eps=1e-6):        
        (a,b,c) = torch.split(hl, 1, dim=dim) # [b,3]
        hl = torch.where(b < 0.0, hl.neg(), hl)
        (a,b,c) = torch.split(hl, 1, dim=dim) # [b,3]
        b = torch.max(b, torch.tensor(eps, device=b.device))
        # compute horizon line
        left  = (a - c)/b  # [-1.0, ( a - c)/b]
        right = (-a - c)/b # [ 1.0, (-a - c)/b]
        return left, right

    def loss_hl(self, outputs, targets, **kwargs):
        assert 'pred_hl' in outputs
        src_hl = outputs['pred_hl']                                    
        target_hl = torch.stack([t['hl'] for t in targets], dim=0)
        target_hl = F.normalize(target_hl, p=2, dim=-1)

        src_l, src_r = self.compute_hl(src_hl)
        target_l, target_r = self.compute_hl(target_hl)

        error_l = (src_l - target_l).abs()
        error_r = (src_r - target_r).abs()
        loss_hl_max = torch.max(error_l, error_r).mean()

        losses = {'loss_hl': loss_hl_max,}
        return losses

    def project_points(self, pts, f=1.0, eps=1e-7):
        # project point on z=1 plane
        device = pts.device
        (x,y,z) = torch.split(pts, 1, dim=-1)
        de = torch.max(torch.abs(z), torch.tensor(eps, device=device))
        de = torch.where(z < 0.0, de.neg(), de)
        return f*(pts/de)

    def loss_consistency(self, outputs, targets, **kwargs):
        assert 'pred_zvp' in outputs
        assert 'pred_hl' in outputs
        src_zvp = outputs['pred_zvp'] # [bs, 3]
        src_hl = outputs['pred_hl']   # [bs, 3]
        target_focal = torch.stack([t['focal'] for t in targets], dim=0) # [bs, 1] 

        up_zvp = src_zvp.clone()
        up_zvp = self.project_points(up_zvp) # zvp on image (z=1) space
        up_zvp[:,2:3] = target_focal # up_vector
        
        up_hl = src_hl.clone()
        up_hl[:,0:2] *= target_focal # up_vector
        
        cos_sim = F.cosine_similarity(up_zvp, up_hl, dim=-1).abs() 
        loss_zvp_hl = (1.0 - cos_sim).mean()

        losses = {'loss_consistency': loss_zvp_hl,}
        return losses
    
    def loss_vline_labels(self, outputs, targets, **kwargs):
        # positive < thresh_pos < no label < thresh_neg < negative
        assert 'pred_vline_logits' in outputs
        src_logits = outputs['pred_vline_logits'] # [bs, n, 1]        
        target_lines = torch.stack([t['lines'] for t in targets], dim=0) # [bs, n, 3]
        target_mask = torch.stack([t['line_mask'] for t in targets], dim=0) # [bs, n, 1]
        target_zvp = torch.stack([t['zvp'] for t in targets], dim=0) # [bs, 3]
        target_zvp = target_zvp.unsqueeze(1) # [bs, 1, 3]
        
        with torch.no_grad():
            cos_sim = F.cosine_similarity(target_lines, target_zvp, dim=-1).abs()
            # [bs, n]
            cos_sim = cos_sim.unsqueeze(-1) # [bs, n, 1]

            ones = torch.ones_like(src_logits)
            zeros = torch.zeros_like(src_logits)
            target_classes = torch.where(cos_sim < self.thresh_line_pos, ones, zeros)
            mask = torch.where(torch.gt(cos_sim, self.thresh_line_pos) &
                               torch.lt(cos_sim, self.thresh_line_neg), 
                               zeros, ones) 
            #import pdb; pdb.set_trace()
            
            # [bs, n, 1]            
            mask = target_mask*mask
                
        loss_ce = F.binary_cross_entropy_with_logits(
            src_logits, target_classes, reduction='none')
        loss_ce = mask*loss_ce
        loss_ce = loss_ce.sum(dim=1)/mask.sum(dim=1)
        
        losses = {'loss_vline_ce': loss_ce.mean(),}
        return losses
    
    def loss_hline_labels(self, outputs, targets, **kwargs):
        assert 'pred_hline_logits' in outputs
        src_logits = outputs['pred_hline_logits'] # [bs, n, 1]
        target_lines = torch.stack([t['lines'] for t in targets], dim=0) # [bs, n, 3]
        target_mask = torch.stack([t['line_mask'] for t in targets], dim=0) # [bs, n, 1]
        
        target_hvps = torch.stack([t['hvps'] for t in targets], dim=0) # [bs, 2, 3]
        
        with torch.no_grad():
            target_lines = target_lines.unsqueeze(dim=2) # [bs, n, 1, 3]
            target_hvps = target_hvps.unsqueeze(dim=1) # [bs, 1, 2, 3]
            
            cos_sim = F.cosine_similarity(target_lines, target_hvps, dim=-1).abs() # [bs, n, 2]
            cos_sim = cos_sim.min(dim=-1, keepdim=True)[0] # [bs, n, 1]
            
#             import pdb; pdb.set_trace()
            
            ones = torch.ones_like(src_logits)
            zeros = torch.zeros_like(src_logits)
            target_classes = torch.where(cos_sim < self.thresh_line_pos, ones, zeros)
            mask = torch.where(torch.gt(cos_sim, self.thresh_line_pos) &
                               torch.lt(cos_sim, self.thresh_line_neg), 
                               zeros, ones)
            # [bs, n, 1]            
            mask = target_mask*mask
    
        loss_ce = F.binary_cross_entropy_with_logits(
            src_logits, target_classes, reduction='none')
        loss_ce = mask*loss_ce
        loss_ce = loss_ce.sum(dim=1)/mask.sum(dim=1)
        
        losses = {'loss_hline_ce': loss_ce.mean(),}
        return losses
    
    
    def get_loss(self, loss, outputs, targets, **kwargs):
        loss_map = {            
            'zvp': self.loss_zvp,
            'fovy': self.loss_fovy,
            'hl': self.loss_hl,
            'consistency': self.loss_consistency,
            'vline_labels': self.loss_vline_labels,
            'hline_labels': self.loss_hline_labels,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                for loss in self.losses:
                    kwargs = {}
                    l_dict = self.get_loss(loss, aux_outputs, targets, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(cfg, train=True):
    device = torch.device(cfg.DEVICE)

    backbone = build_backbone(cfg)
    transformer = build_transformer(cfg)

    model = GPTran(
        backbone,
        transformer,        
        num_queries=cfg.MODELS.TRANSFORMER.NUM_QUERIES,
        aux_loss=cfg.LOSS.AUX_LOSS,
        use_structure_tensor=cfg.MODELS.USE_STRUCTURE_TENSOR,
    )
    weight_dict = dict(cfg.LOSS.WEIGHTS)
    
    # TODO this is a hack
    if cfg.LOSS.AUX_LOSS:
        aux_weight_dict = {}
        for i in range(cfg.MODELS.TRANSFORMER.DEC_LAYERS - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = cfg.LOSS.LOSSES    
    criterion = SetCriterion(weight_dict=weight_dict,
                             losses=losses,
                             line_pos_angle=cfg.LOSS.LINE_POS_ANGLE,
                             line_neg_angle=cfg.LOSS.LINE_NEG_ANGLE)
    criterion.to(device)    

    return model, criterion
