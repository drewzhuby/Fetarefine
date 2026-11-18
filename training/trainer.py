"""New-run trainer: explicit model inputs and full-case Dice selection."""
import json,math,random,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from training.configuration import FIXED
from models.m5_selective import SelectiveRetentionLoss
from training.losses import DiffUNetLoss
from inference import predict_volume
from data.volumes import image_volume,supervised_volume,load_logits,raw_file
from data.preprocessing import inverse_map_to_native
import nibabel as nib


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False

def segmentation_loss(z,y,m):
    den=m.sum().clamp_min(1);ce=(F.cross_entropy(z.float(),y,reduction='none')*m).sum()/den
    p=z.float().softmax(1);t=F.one_hot(y,8).movedim(-1,1).float();mask=m[:,None]
    axes=(0,*range(2,z.ndim));ns=(t*mask).sum(axes);ps=(p*mask).sum(axes);tp=(p*t*mask).sum(axes)
    values=(1-(2*tp[1:]+1e-5)/(ns[1:]+ps[1:]+1e-5))[ns[1:]>0]
    return .5*ce+.5*(values.mean() if values.numel() else z.sum()*0)

def training_loss(model,name,batch,config):
    x=batch['image'];y=batch['label'];m=batch['valid']
    if not m.any():raise ValueError('All-invalid batch')
    if name=='m3':
        if 'prior' not in batch:raise ValueError('Missing M0 prior')
        pr=batch['prior'];z=model(torch.cat([x,pr.softmax(1)],1),pr)
    elif name in ('m5_br','unbounded'):
        z,delta=model(x,batch['deep'],batch['swin'],return_residual=True)
        return segmentation_loss(z,y,m)+config['training']['lambda_residual']*(delta.abs()*m[:,None]).sum()/(m.sum()*8)
    elif name=='m5_gate':
        if 'lr' not in batch:raise ValueError('Missing BR logits; refusing M4 fallback')
        l4=(batch['deep']+batch['swin'])*.5;lr=batch['lr'].detach()
        z,_=model(x,l4,lr)
        weight=config['training']['loss']['retention_keep_weight']
        loss, _ = SelectiveRetentionLoss(keep_weight=weight)(z,lr,y,m)
        return loss
    elif name=='diffunet':
        target=F.one_hot(y,8).movedim(-1,1)[:,1:].float();z=model(x,target)
        return DiffUNetLoss()(z,target,m)
    else:z=model(x)
    return segmentation_loss(z,y,m)

@torch.inference_mode()
def validate_cases(model,config,root,cases,device,directories,raw_root):
    scores=[]
    for case in tqdm(cases,desc='Full-case validation Dice',unit='case'):
        x,y,m=supervised_volume(root,case)
        kwargs={k:load_logits(v,case,root,verify=False) for k,v in directories.items()}
        z=predict_volume(model,config['model_id'],x,device,**kwargs,roi=config['training'].get('patch_size',[96]*3))
        pred=inverse_map_to_native(z.argmax(0),Path(root)/case/'anat/properties.json')
        gt=nib.load(str(raw_file(raw_root,case,'dseg')));y=np.asarray(gt.dataobj);m=np.ones(y.shape,dtype=np.uint8)
        props=json.loads((Path(root)/case/'anat/properties.json').read_text())
        if pred.shape!=y.shape or not np.allclose(gt.affine,props['original_affine']):raise ValueError('Native validation grid mismatch')
        ds=[]
        for c in range(1,8):
            a=(pred==c)&(m>0);b=(y==c)&(m>0)
            if b.any():ds.append(float(2*(a&b).sum()/(a.sum()+b.sum())))
        if not ds:raise ValueError(f'No foreground in validation case {case}')
        scores.append(float(np.mean(ds)))
    return float(np.mean(scores))

class Trainer:
    def __init__(self,model,loader,config,out,device,phase,root,val_cases,directories,parents,identity,epochs,raw_root):
        self.model=model.to(device);self.loader=loader;self.c=config;self.out=Path(out);self.device=device
        self.phase=phase;self.root=root;self.val_cases=val_cases;self.directories=directories
        self.parents=parents;self.identity=identity;self.epochs=epochs;self.raw_root=raw_root
    def fit(self):
        t=self.c['training'];name=self.c['model_id'];model=self.model
        opt=torch.optim.AdamW(model.parameters(),lr=t['lr'],weight_decay=t['weight_decay'])
        history=[];best=-float('inf');significant=-float('inf');stale=0;best_epoch=None;updates=0
        # LR schedule keeps the configured development horizon in final64 too.
        for epoch in tqdm(range(1,self.epochs+1),desc=f'{name} epochs',unit='epoch'):
            seed_all(t['seed']+epoch);self.loader.dataset.epoch=epoch
            horizon=t['max_epochs'];warm=t.get('warmup_epochs',0)
            if t['lr_schedule']=='constant':lr=t['lr']
            elif warm and epoch<=warm:lr=t['lr']*epoch/warm
            else:lr=t.get('min_lr',1e-6)+(t['lr']-t.get('min_lr',1e-6))*.5*(1+math.cos(math.pi*(epoch-1-warm)/max(1,horizon-warm)))
            for group in opt.param_groups:group['lr']=lr
            model.train();total=0.;batches=0;wait=0.;compute=0.;tick=time.perf_counter()
            for batch in tqdm(self.loader,desc=f'epoch {epoch}/{self.epochs} batches',unit='batch',leave=False):
                wait+=time.perf_counter()-tick;start=time.perf_counter()
                batch={k:v.to(self.device) for k,v in batch.items()}
                opt.zero_grad(set_to_none=True)
                amp=t['precision']=='bfloat16'
                with torch.autocast(device_type=torch.device(self.device).type,dtype=torch.bfloat16,enabled=amp):
                    loss=training_loss(model,name,batch,self.c)
                if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
                total+=float(loss.detach());batches+=1;updates+=1;compute+=time.perf_counter()-start;tick=time.perf_counter()
            if name in FIXED and batches!=len(self.loader.dataset)//t['batch_size']:raise ValueError('Fixed-stage update count mismatch')
            row=dict(epoch=epoch,loss=total/batches,updates=updates,lr=lr,data_wait_seconds=wait,compute_seconds=compute)
            checked=self.phase=='phase1' and (epoch%t['val_every']==0 or epoch==self.epochs)
            if checked:
                score=validate_cases(model,self.c,self.root,self.val_cases,self.device,self.directories,self.raw_root);row['validation_dice']=score
                if score>best:
                    best=score;best_epoch=epoch
                    self.save('best.pt',opt,epoch,updates,best)
                if score>significant+t['min_delta']:significant=score;stale=0
                else:stale+=1
            history.append(row);tqdm.write(json.dumps(row))
            self.save('last.pt',opt,epoch,updates,best if checked else None)
            (self.out/'history.json').write_text(json.dumps(history,indent=2))
            if checked and epoch>=t['min_epochs'] and stale>=t['patience_checks']:break
        if self.phase=='phase1':
            (self.out/'SELECTION.json').write_text(json.dumps(dict(format='feta-selection-v2',model_id=name,best_epoch=best_epoch,
                validation_dice=best,config=self.c,criterion='case_equal_reference_present_foreground_dice_native',
                train_cases=self.loader.dataset.cases,val_cases=self.val_cases,identity=self.identity),indent=2))
        else:self.save('final.pt',opt,epoch,updates)
        (self.out/'COMPLETE.json').write_text(json.dumps(dict(model=name,phase=self.phase,epochs=epoch,updates=updates),indent=2))
    def save(self,name,opt,epoch,updates,score=None):
        tmp=self.out/(name+'.pending')
        torch.save(dict(format='feta-clean-v2',config=self.c,phase=self.phase,epoch=epoch,updates=updates,
            validation_dice=score,model_state_dict=self.model.state_dict(),optimizer_state_dict=opt.state_dict(),
            parents=self.parents,identity=self.identity),tmp)
        tmp.replace(self.out/name)
