"""Invert existing PP1 predictions using its recorded physical affine.

This module does not regenerate or overwrite preprocessing assets.
"""
import json
from pathlib import Path
import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np

def inverse_map_to_native(pred_pp1,properties_path):
    if pred_pp1.ndim != 3 or not np.isin(pred_pp1, np.arange(8)).all():
        raise ValueError('Expected discrete labels 0..7 before inverse mapping')
    p=json.loads(Path(properties_path).read_text())
    if 'derived_affine' not in p or 'derived_shape' not in p:
        raise ValueError('Expected PP1 properties with derived_shape/derived_affine')
    if tuple(pred_pp1.shape)!=tuple(p['derived_shape']):raise ValueError('Prediction does not match PP1 derived grid')
    im=nib.Nifti1Image(pred_pp1.astype(np.uint8),np.array(p['derived_affine']))
    native=resample_from_to(im,(tuple(p['original_shape']),np.array(p['original_affine'])),order=0,mode='constant',cval=0)
    return np.asarray(native.dataobj,dtype=np.uint8)
