"""Generate new PP1 logits and native predictions, without reading GT."""
from runtime import setup_environment,inside,output_path
setup_environment()
import argparse,json,shutil
from pathlib import Path
import nibabel as nib
import numpy as np
import torch
from tqdm.auto import tqdm
from data.splits import TRAIN_64_CASES,VAL_8_CASES,TEST_16_CASES
from data.volumes import image_volume,image_path,load_logits,cache_receipt
from data.preprocessing import inverse_map_to_native
from training.configuration import load_checkpoint,sha
from inference import predict_volume


def cache_inputs(args,case,root,name,verify=True):
    values={};parents={}
    for key,attr,expected in [('prior','priors_dir','m0'),('deep','expert_2p5d_dir','m2'),('swin','expert_3d_dir','m0'),('lr','lr_dir','m5_br')]:
        needed=(name=='m3' and key=='prior') or (name in ('m4','m5_br','unbounded','m5_gate') and key in ('deep','swin')) or (name=='m5_gate' and key=='lr')
        if not needed:continue
        directory=getattr(args,attr,None)
        if not directory:raise ValueError(f'{name} requires --{attr}')
        directory=inside(directory)
        values[key]=load_logits(directory,case,root,expected,verify=verify)
        receipt=cache_receipt(directory,case);parents[key]=receipt['checkpoint_sha256']
        if key in ('deep','swin') and receipt.get('phase')!='phase2':raise ValueError('Fusion requires final64 expert caches')
        if key=='prior' and getattr(args,'phase',None) and receipt.get('phase')!=args.phase:
            raise ValueError('M3 development/final prior phase mismatch')
        if key=='lr' and receipt.get('phase')!='phase2':raise ValueError('Gate requires a completed train64 BR cache')
        if key=='lr' and receipt['parents']!={'deep':parents['deep'],'swin':parents['swin']}:
            raise ValueError('BR cache was generated from different expert checkpoints')
    return values,parents


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True,choices=['m0','m1','m2','m3','m4','m5_br','m5_gate','m5','unbounded','diffunet'])
    p.add_argument('--checkpoint');p.add_argument('--data_root',default='data/preprocessed')
    p.add_argument('--output_dir',required=True);p.add_argument('--case');p.add_argument('--group',choices=['train64','val8','test16','all80'],default='test16')
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    for arg in ['priors_dir','expert_3d_dir','expert_2p5d_dir','lr_dir']:p.add_argument('--'+arg)
    args=p.parse_args();name='m5_gate' if args.model=='m5' else args.model
    root=inside(args.data_root);out=output_path(args.output_dir)
    if out.exists():raise FileExistsError(f'Fresh output directory required: {out}')
    cases=[args.case] if args.case else {'train64':TRAIN_64_CASES,'val8':VAL_8_CASES,'test16':TEST_16_CASES,'all80':sorted(TRAIN_64_CASES+TEST_16_CASES)}[args.group]
    if any(c not in TRAIN_64_CASES+TEST_16_CASES for c in cases):raise ValueError('Unknown case')
    if name=='m4':model=None;ck={'config':{'model_id':'m4'},'phase':'phase2','parents':{}};checkpoint_sha='no_trainable_parameters'
    else:
        if not args.checkpoint:raise ValueError('Checkpoint required')
        model,ck=load_checkpoint(args.checkpoint,args.device);checkpoint_sha=sha(inside(args.checkpoint))
        if ck['config']['model_id']!=name:raise ValueError('Checkpoint model mismatch')
    # Reserve disk for all FP32 logits plus native predictions before starting.
    needed=1024**3
    for case in cases:
        props=json.loads((root/case/'anat/properties.json').read_text())
        needed+=int(np.prod(props['derived_shape']))*32+int(np.prod(props['original_shape']))*2
    parent=out.parent
    while not parent.exists():parent=parent.parent
    if shutil.disk_usage(parent).free<needed:raise ValueError(f'Insufficient output disk: need about {needed} bytes')
    out.mkdir(parents=True)
    for case in tqdm(cases,desc='Inference',unit='case'):
        x,aff=image_volume(root,case);inputs,parents=cache_inputs(args,case,root,name)
        if ck.get('parents',{})!=parents and name!='m4':raise ValueError('Inference parents differ from training parents')
        logits=predict_volume(model,name,x,args.device,**inputs,progress=True,roi=ck['config'].get('training',{}).get('patch_size',[96]*3))
        if not np.isfinite(logits).all():raise ValueError(f'Nonfinite prediction: {case}')
        file=out/f'{case}_logits.npy';np.save(file,logits,allow_pickle=False)
        receipt=dict(case=case,model_id=name,phase=ck['phase'],shape=list(logits.shape),affine=aff.tolist(),
                     checkpoint_sha256=checkpoint_sha,parents=parents,array_sha256=sha(file),image_sha256=sha(image_path(root,case)))
        (out/f'{case}_logits.json').write_text(json.dumps(receipt,indent=2))
        props_path=root/case/'anat/properties.json';props=json.loads(props_path.read_text())
        native=inverse_map_to_native(logits.argmax(0),props_path)
        nib.save(nib.Nifti1Image(native,np.array(props['original_affine'])),out/f'{case}_pred.nii.gz')
    (out/'COMPLETE.json').write_text(json.dumps(dict(cases=cases,model=name,checkpoint_sha256=checkpoint_sha),indent=2))

if __name__=='__main__':main()
