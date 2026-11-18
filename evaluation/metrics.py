"""26-neighbor world-affine boundary points; pooled bidirectional distances."""
import numpy as np
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree

METRICS=('dice','hd95_mm','assd_mm','nsd','sensitivity','precision')

def surface_distances(pred,gt,affine):
    if not pred.any() or not gt.any():return None
    struct=np.ones((3,3,3),dtype=bool)
    a=np.argwhere(pred & ~binary_erosion(pred,structure=struct,border_value=0))
    b=np.argwhere(gt & ~binary_erosion(gt,structure=struct,border_value=0))
    a=a@affine[:3,:3].T+affine[:3,3];b=b@affine[:3,:3].T+affine[:3,3]
    return np.concatenate([cKDTree(b).query(a,workers=1)[0],cKDTree(a).query(b,workers=1)[0]])

def evaluate_case(pred,gt,affine,tau_mm=1.):
    if pred.shape!=gt.shape or pred.ndim!=3:raise ValueError('Prediction/GT shape mismatch')
    if not np.isin(pred,np.arange(8)).all() or not np.isin(gt,np.arange(8)).all():raise ValueError('Invalid labels')
    rows=[]
    for c in range(1,8):
        a=pred==c;b=gt==c;tp=int((a&b).sum());na=int(a.sum());nb=int(b.sum());present=nb>0
        row=dict(class_id=c,reference_present=present,dice=2*tp/(na+nb) if present else None,
            sensitivity=tp/nb if present else None,precision=tp/na if na else (0. if present else None))
        ds=surface_distances(a,b,affine) if present else None
        row.update(hd95_mm=float(np.percentile(ds,95)) if ds is not None else None,
                   assd_mm=float(ds.mean()) if ds is not None else None,nsd=float((ds<=tau_mm).mean()) if ds is not None else None)
        rows.append(row)
    return rows

def case_means(rows):
    # A failed distance is propagated rather than silently excluding difficult classes.
    present=[r for r in rows if r['reference_present']]
    if not present:raise ValueError('No reference foreground')
    return {k:float(np.mean([r[k] for r in present])) if all(r[k] is not None for r in present) else None for k in METRICS}
