"""Full-volume inference shared by validation and standalone prediction."""
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


def context(x,z):
    return np.stack([x[:,:,min(x.shape[2]-1,max(0,z+i))] for i in (-1,0,1)])

@torch.inference_mode()
def predict_volume(model,name,x,device,prior=None,deep=None,swin=None,lr=None,progress=False,roi=(96,96,96)):
    if name=='m4':
        if deep is None or swin is None:raise ValueError('M4 requires both experts')
        return (np.asarray(deep)+np.asarray(swin))*np.float32(.5)
    model.eval()
    if name in ('m0','diffunet'):
        from monai.inferers import sliding_window_inference
        tensor=torch.from_numpy(np.ascontiguousarray(x))[None,None]
        def predictor(patch):
            if name=='diffunet':return model.sample(patch).float().sigmoid()
            return model(patch).float()
        y=sliding_window_inference(tensor,roi_size=tuple(roi),sw_batch_size=1,predictor=predictor,
                                   overlap=.5,mode='gaussian',sw_device=device,device='cpu',progress=progress)
        if name=='diffunet':
            # Aggregate continuous seven-channel states, then mutually exclusive decode.
            p=y[0].numpy()
            return np.concatenate([np.full((1,*x.shape),.5,dtype=np.float32),p],axis=0).astype(np.float32)
        return y[0].numpy().astype(np.float32)
    if name=='m5_gate':
        if deep is None or swin is None or lr is None:raise ValueError('Gate requires expert and BR logits; no M4 fallback')
        l4=(np.asarray(deep)+np.asarray(swin))*np.float32(.5)
        out=np.empty_like(l4)
        starts=[range(0,n,64) for n in x.shape]
        from itertools import product
        for start in tqdm(list(product(*starts)),desc='Gate tiles',disable=not progress):
            stop=[min(a+64,n) for a,n in zip(start,x.shape)]
            lo=[max(0,a-2) for a in start];hi=[min(n,b+2) for n,b in zip(x.shape,stop)]
            box=tuple(slice(a,b) for a,b in zip(lo,hi));core=tuple(slice(a-l,b-l) for a,b,l in zip(start,stop,lo))
            im=torch.from_numpy(np.ascontiguousarray(x[box]))[None,None].to(device)
            a=torch.from_numpy(np.ascontiguousarray(l4[(slice(None),*box)]))[None].to(device)
            b=torch.from_numpy(np.ascontiguousarray(lr[(slice(None),*box)]))[None].to(device)
            fused,_=model(im,a,b)
            dest=tuple(slice(a,b) for a,b in zip(start,stop))
            out[(slice(None),*dest)]=fused[0][(slice(None),*core)].float().cpu().numpy()
        return out
    out=np.empty((8,*x.shape),dtype=np.float32)
    for z in tqdm(range(x.shape[2]),desc='Slices',disable=not progress):
        a=context(x,z)
        if name=='m1':a=np.repeat(x[:,:,z][None],3,axis=0)
        t=torch.from_numpy(np.ascontiguousarray(a))[None].to(device);h,w=t.shape[-2:]
        if name in ('m5_br','unbounded'):
            if deep is None or swin is None:raise ValueError('Residual inference requires both experts')
            dl=torch.from_numpy(np.array(deep[:,:,:,z],copy=True))[None].to(device)
            sw=torch.from_numpy(np.array(swin[:,:,:,z],copy=True))[None].to(device)
            result=model(t,dl,sw)
        else:
            pad=(0,(-w)%32,0,(-h)%32)
            if name=='m3':
                if prior is None:raise ValueError('M3 requires frozen M0 prior')
                pr=torch.from_numpy(np.array(prior[:,:,:,z],copy=True))[None].to(device)
                result=model(F.pad(torch.cat([t,pr.softmax(1)],1),pad),F.pad(pr,pad))
            else: result=model(F.pad(t,pad))
        out[:,:,:,z]=result[0,:,:h,:w].float().cpu().numpy()
    if not np.isfinite(out).all():raise ValueError('Nonfinite predicted logits')
    return out
