import copy
import hashlib
import json
from pathlib import Path
import yaml
from runtime import ROOT, inside

CONFIGS = dict(m0='m0_swin_unetr',m1='m1_deeplab_2d',m2='m2_deeplab_2p5d',m3='m3_prior_residual',
               m5_br='m5_selective',m5_gate='m5_selective',unbounded='ablation_unbounded',diffunet='ablation_diffunet')
FIXED = {'m5_br','m5_gate','unbounded'}

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()

def load_config(model, path=None):
    path=inside(path or ROOT/'configs'/f'{CONFIGS[model]}.yaml')
    c=yaml.safe_load(path.read_text())
    if model in ('m5_br','m5_gate'):
        stage=c['stage1_bounded_residual' if model=='m5_br' else 'stage2_selective_gating']
        c={'model':{k:stage[k] for k in ('in_channels','width','delta_max','initial_acceptance') if k in stage},
           'training':{k:v for k,v in stage.items() if k not in ('in_channels','width','delta_max','initial_acceptance','parameters')}}
    c=copy.deepcopy(c); c['model_id']=model
    t=c['training']; t.setdefault('seed',20260918); t.setdefault('precision','float32')
    t.setdefault('min_lr',1e-6); t.setdefault('min_delta',0.001)
    if t['batch_size']<1 or t['max_epochs']<1 or t['lr']<=0: raise ValueError('Invalid training configuration')
    if t.get('lr_schedule') not in ('constant','cosine'): raise ValueError('Unsupported LR schedule')
    if model=='m5_gate' and (t['loss']['ce_weight'],t['loss']['dice_weight']) != (.5,.5): raise ValueError('Gate segmentation loss requires 0.5 CE + 0.5 Dice')
    if t['optimizer']!='AdamW': raise ValueError('Only AdamW is implemented')
    if t['precision'] not in ('float32','bfloat16'): raise ValueError('Unsupported precision')
    if model in FIXED:
        expected=(2,16,15360) if model=='m5_gate' else (4,64,30720)
        if (t['batch_size'],t['samples_per_case'],t['max_epochs']) != (expected[0],expected[1],30):
            raise ValueError('Fixed residual/gate protocol requires original batch/samples/30 epochs')
        if t.get('lr_schedule')!='constant': raise ValueError('Fixed stages require constant LR')
    return c

def build_model(c, initialize=True):
    from models import M0SwinUNETR,M1DeepLab2D,M2DeepLab2p5D,M3PriorResidual,BoundedResidualNet,SelectiveGating3DNet,UnboundedResidualNet,DiffUNetSegmentation
    m=c['model']; name=c['model_id']
    if name=='m0': return M0SwinUNETR(img_size=tuple(m['img_size']),feature_size=m['feature_size'])
    if name in ('m1','m2','m3'):
        cls={'m1':M1DeepLab2D,'m2':M2DeepLab2p5D,'m3':M3PriorResidual}[name]
        return cls(num_classes=8,pretrained=bool(m.get('pretrained',True) and initialize))
    if name=='m5_br': return BoundedResidualNet(in_channels=m['in_channels'],width=m['width'],delta_max=m['delta_max'])
    if name=='m5_gate': return SelectiveGating3DNet(in_channels=m['in_channels'],width=m['width'],initial_acceptance=m['initial_acceptance'])
    if name=='unbounded': return UnboundedResidualNet(in_channels=m['in_channels'],width=m['width'],scale=m['scale'])
    if name=='diffunet': return DiffUNetSegmentation(features=tuple(m['features']),timesteps=m['timesteps'],ddim_steps=m['ddim_steps'])
    raise ValueError(name)

def load_checkpoint(path,device='cpu'):
    import torch
    path=inside(path)
    ck=torch.load(path,map_location='cpu',weights_only=False)
    if not isinstance(ck,dict) or ck.get('format')!='feta-clean-v2':
        raise ValueError('Expected a new feta-clean-v2 checkpoint; historical checkpoints need an explicit converter')
    model=build_model(ck['config'],initialize=False)
    model.load_state_dict(ck['model_state_dict'],strict=True)
    model.to(device).eval()
    return model,ck
