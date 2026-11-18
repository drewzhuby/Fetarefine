"""Fresh, source-bound training; phase1 Dice selection then fresh phase2."""
from runtime import setup_environment,inside,output_path,ROOT
setup_environment()
import argparse,json,sys,platform,shutil
from pathlib import Path
import nibabel as nib
import numpy as np
import torch,yaml
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from data.splits import TRAIN_56_CASES,VAL_8_CASES,TRAIN_64_CASES,TEST_16_CASES
from training.configuration import CONFIGS,FIXED,load_config,build_model,sha
from training.resources import configure,init_worker
from training.trainer import Trainer,seed_all
from data.training_dataset import TrainingDataset,CaseOrderSampler
from data.volumes import image_path,raw_file
from infer import cache_inputs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True,choices=list(CONFIGS));p.add_argument('--phase',choices=['phase1','phase2'],required=True)
    p.add_argument('--config');p.add_argument('--data_root');p.add_argument('--raw_root')
    p.add_argument('--output_dir',required=True);p.add_argument('--selection',help='New phase1 SELECTION.json; required for baseline phase2')
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu');p.add_argument('--workers',type=int)
    p.add_argument('--preflight-only',action='store_true')
    for arg in ['priors_dir','expert_3d_dir','expert_2p5d_dir','lr_dir']:p.add_argument('--'+arg)
    a=p.parse_args();c=load_config(a.model,a.config)
    dataset=yaml.safe_load((ROOT/'configs/dataset.yaml').read_text())
    root=inside(a.data_root or dataset['data']['preprocessed_root']);raw=inside(a.raw_root or dataset['data']['raw_root'])
    if a.model in FIXED and a.phase!='phase2':raise ValueError('Residual and gate stages are fixed train64 runs')
    if set(TRAIN_56_CASES)&set(VAL_8_CASES) or set(TRAIN_64_CASES)&set(TEST_16_CASES) or set(TRAIN_56_CASES+VAL_8_CASES)!=set(TRAIN_64_CASES):raise ValueError('Invalid split')
    cases=TRAIN_56_CASES if a.phase=='phase1' else TRAIN_64_CASES
    val=VAL_8_CASES if a.phase=='phase1' else []
    epochs=c['training']['max_epochs'];selection_sha=None
    if a.phase=='phase2' and a.model not in FIXED:
        if not a.selection:raise ValueError('Fresh final64 requires --selection from a completed phase1 run')
        sp=inside(a.selection);s=json.loads(sp.read_text())
        if s.get('format')!='feta-selection-v2' or s['config']!=c:raise ValueError('Selection/config mismatch')
        if s['train_cases']!=TRAIN_56_CASES or s['val_cases']!=VAL_8_CASES:raise ValueError('Selection split mismatch')
        if not (sp.parent/'COMPLETE.json').exists():raise ValueError('Phase1 must be completed')
        epochs=s['best_epoch'];selection_sha=sha(sp)
        if not isinstance(epochs,int) or not 1<=epochs<=c['training']['max_epochs']:raise ValueError('Invalid selected epoch')
    elif a.selection:raise ValueError('--selection only applies to baseline phase2')
    out=output_path(a.output_dir)
    if out.exists():raise FileExistsError(f'Fresh run output required: {out}')
    directories={};parents=None;input_hashes={};max_voxels=0
    for case in tqdm(cases+val,desc='Preflight inputs and cache hashes',unit='case'):
        im=nib.load(str(image_path(root,case)));max_voxels=max(max_voxels,int(np.prod(im.shape)))
        folder=root/case/'anat'
        for n in ['image.nii.gz','label.nii.gz','valid_mask.nii.gz','properties.json']:
            f=folder/n;input_hashes[str(f.relative_to(ROOT))]=sha(f)
        props=json.loads((folder/'properties.json').read_text())
        if tuple(props['derived_shape'])!=im.shape or not np.allclose(props['derived_affine'],im.affine):raise ValueError('Properties/grid mismatch')
        values,pa=cache_inputs(a,case,root,a.model)
        if parents is None:parents=pa
        if pa!=parents:raise ValueError('Mixed checkpoint sources across cases')
        for key,attr in [('prior','priors_dir'),('deep','expert_2p5d_dir'),('swin','expert_3d_dir'),('lr','lr_dir')]:
            if key in values:
                directories[key]=inside(getattr(a,attr))
                for ext in ['npy','json']:
                    f=directories[key]/f'{case}_logits.{ext}';input_hashes[str(f.relative_to(ROOT))]=sha(f)
        if case in val:
            f=raw_file(raw,case,'dseg');input_hashes[str(f.relative_to(ROOT))]=sha(f)
        del values
    resources=configure(max_voxels,1+8*len(directories),a.workers)
    if c['training']['precision']=='bfloat16' and str(a.device).startswith('cuda') and not torch.cuda.is_bf16_supported():raise ValueError('BF16 unsupported; do not silently change precision')
    source_files=[*ROOT.glob('*.py'),*ROOT.glob('models/**/*.py'),*ROOT.glob('data/*.py'),*ROOT.glob('training/*.py'),*ROOT.glob('evaluation/*.py'),*ROOT.glob('configs/*.yaml')]
    identity=dict(format='feta-clean-run-v2',model=a.model,phase=a.phase,config=c,epochs=epochs,
        train_cases=cases,val_cases=val,selection_sha256=selection_sha,parents=parents,
        input_sha256=input_hashes,source_sha256={str(f.relative_to(ROOT)):sha(f) for f in source_files},
        resources=resources,python=sys.version,torch=torch.__version__,platform=platform.platform(),
        note='New implementation; no claim of bitwise historical reproduction or exact resume')
    if selection_sha is not None:
        for key,value in s['identity']['source_sha256'].items():
            if identity['source_sha256'].get(key)!=value:raise ValueError('Source changed since phase1 selection; use a new development run')
        for key,value in s['identity']['input_sha256'].items():
            if key in input_hashes and input_hashes[key]!=value:raise ValueError('Training data changed since phase1')
    parent=out.parent
    while not parent.exists():parent=parent.parent
    if shutil.disk_usage(parent).free < 4*1024**3:raise ValueError('Need at least 4 GiB free for new checkpoints and logs')
    if a.preflight_only:
        print(json.dumps(dict(status='PREFLIGHT_PASS',epochs=epochs,resources=resources,parents=parents),indent=2));return
    seed_all(c['training']['seed'])
    ds=TrainingDataset(root,cases,c,directories)
    workers=resources['workers'];gen=torch.Generator().manual_seed(c['training']['seed'])
    loader=DataLoader(ds,batch_size=c['training']['batch_size'],sampler=CaseOrderSampler(ds),drop_last=a.model in ('m1','m2','m3'),
        num_workers=workers,worker_init_fn=init_worker,generator=gen,persistent_workers=False,
        **({'prefetch_factor':1} if workers else {}))
    if not len(loader):raise ValueError('No complete training batch')
    model=build_model(c,initialize=True)
    out.mkdir(parents=True);(out/'IDENTITY.json').write_text(json.dumps(identity,indent=2));(out/'CONFIG.json').write_text(json.dumps(c,indent=2))
    Trainer(model,loader,c,out,a.device,a.phase,root,val,directories,parents,identity,epochs,raw_root=raw).fit()

if __name__=='__main__':main()
