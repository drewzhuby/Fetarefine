"""Validated PP1 volumes and source-bound logit caches."""
import json
from pathlib import Path
import nibabel as nib
import numpy as np
from training.configuration import sha


def raw_file(root,case,kind):
    folder=Path(root)/case/'anat'
    matches=sorted(p for p in folder.glob(f'{case}*_{kind}.nii.gz') if p.is_file())
    if len(matches)!=1: raise ValueError(f'Expected one {kind} for {case}: {matches}')
    return matches[0]

def image_path(root,case): return Path(root)/case/'anat/image.nii.gz'

def image_volume(root,case):
    p=image_path(root,case); im=nib.load(str(p)); x=im.get_fdata(dtype=np.float32)
    if x.ndim!=3 or not np.isfinite(x).all(): raise ValueError(f'Invalid image: {p}')
    return x,im.affine

def supervised_volume(root,case):
    x,aff=image_volume(root,case); folder=Path(root)/case/'anat'
    volumes=[nib.load(str(folder/n)) for n in ('label.nii.gz','valid_mask.nii.gz')]
    if any(v.shape!=x.shape or not np.allclose(v.affine,aff,atol=1e-5) for v in volumes):raise ValueError(f'PP1 geometry mismatch: {case}')
    y=np.asarray(volumes[0].dataobj);mask=np.asarray(volumes[1].dataobj)
    if not np.isin(y,np.arange(8)).all() or not np.isin(mask,[0,1]).all() or not mask.any():raise ValueError(f'Invalid labels/mask: {case}')
    return x,y.astype(np.int64),mask.astype(np.float32)

def cache_receipt(directory,case):
    p=Path(directory)/f'{case}_logits.json'
    if not p.exists(): raise ValueError(f'Missing cache provenance: {p}; generate using infer.py')
    return json.loads(p.read_text())

def load_logits(directory,case,root,expected=None,verify=True):
    receipt=cache_receipt(directory,case);p=Path(directory)/f'{case}_logits.npy';im=nib.load(str(image_path(root,case)))
    if expected and receipt['model_id']!=expected: raise ValueError(f'Expected {expected} cache: {p}')
    if receipt['case']!=case or receipt['shape']!=[8,*im.shape] or not np.allclose(receipt['affine'],im.affine):raise ValueError(f'Cache grid mismatch: {p}')
    if verify and (sha(p)!=receipt['array_sha256'] or sha(image_path(root,case))!=receipt['image_sha256']):raise ValueError(f'Cache content mismatch: {p}')
    a=np.load(p,mmap_mode='r',allow_pickle=False)
    if a.dtype!=np.float32 or a.shape!=(8,*im.shape):raise ValueError(f'Invalid logits: {p}')
    return a
