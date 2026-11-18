"""Deterministic joint image/logit sampling for fresh experiments."""
from pathlib import Path
from collections import OrderedDict
import numpy as np
import torch
from torch.utils.data import Dataset,Sampler
from tqdm.auto import tqdm
from data.volumes import supervised_volume,load_logits


def crop_pad(a,starts,sizes):
    d=len(sizes);prefix=a.ndim-d
    sl=(slice(None),)*prefix+tuple(slice(s,s+n) for s,n in zip(starts,sizes))
    a=np.array(a[sl],copy=True)
    return np.pad(a,[(0,0)]*prefix+[(0,n-a.shape[prefix+i]) for i,n in enumerate(sizes)])

class TrainingDataset(Dataset):
    def __init__(self,root,cases,config,directories):
        self.root=Path(root);self.cases=list(cases);self.config=config;self.directories=directories
        self.name=config['model_id'];self.t=config['training'];self.epoch=0;self.cache=OrderedDict();self.index=[]
        if self.name in ('m1','m2','m3'):
            for case in tqdm(self.cases,desc='Index training slices',unit='case'):
                x,y,m=supervised_volume(root,case)
                for z in range(y.shape[2]):
                    if np.count_nonzero((y[:,:,z]>0)&(m[:,:,z]>0))>=self.t.get('minimum_slice_foreground',10):self.index.append((case,z))
        else:self.index=[(c,None) for c in self.cases for _ in range(self.t['samples_per_case'])]
        if not self.index:raise ValueError('Empty training dataset')
    def __len__(self):return len(self.index)
    def arrays(self,case):
        if case not in self.cache:
            self.cache.clear()
            x,y,m=supervised_volume(self.root,case);v={'image':x,'label':y,'valid':m}
            for key,directory in self.directories.items():
                v[key]=load_logits(directory,case,self.root,verify=False)
            self.cache[case]=v
        return self.cache[case]
    def __getitem__(self,index):
        case,z=self.index[index];v=self.arrays(case);x=v['image'];m=v['valid'];y=v['label']
        rng=np.random.default_rng(np.random.SeedSequence([self.t['seed'],self.epoch,index]))
        two=self.name in ('m1','m2','m3','m5_br','unbounded')
        if z is None:
            candidates=np.argwhere((y>0)&(m>0)) if self.name in ('m0','diffunet') and rng.random()<2/3 else np.argwhere(m>0)
            if len(candidates)==0:candidates=np.argwhere(m>0)
            center=candidates[rng.integers(len(candidates))];z=int(center[2])
        else:
            pts=np.argwhere(m[:,:,z]>0);center=(*pts[rng.integers(len(pts))],z)
        if two:
            size=self.t.get('tile_size',[256,256]);size=[size,size] if isinstance(size,int) else size
            starts=[max(0,min(int(c)-n//2,s-n)) for c,n,s in zip(center,size,x.shape)]
            image=np.stack([x[:,:,min(x.shape[2]-1,max(0,z+i))] for i in (-1,0,1)])
            if self.name=='m1':image=np.repeat(x[:,:,z][None],3,axis=0)
            arrays={'image':image,'label':y[:,:,z],'valid':m[:,:,z]}
            arrays.update({k:a[:,:,:,z] for k,a in v.items() if k not in ('image','label','valid')})
        else:
            size=self.t['patch_size'];starts=[max(0,min(int(c)-n//2,s-n)) for c,n,s in zip(center,size,x.shape)]
            arrays={'image':x[None],'label':y,'valid':m}
            arrays.update({k:a for k,a in v.items() if k not in ('image','label','valid')})
        arrays={k:crop_pad(a,starts,size) for k,a in arrays.items()}
        for axis in (0,1):
            if rng.random()<.5:
                arrays={k:np.flip(a,axis=a.ndim-len(size)+axis) for k,a in arrays.items()}
        result={k:torch.from_numpy(np.ascontiguousarray(a)).float() for k,a in arrays.items()}
        result['label']=result['label'].long();return result

class CaseOrderSampler(Sampler):
    """Keep residual/gate case batches adjacent to bound repeated volume loads."""
    def __init__(self,dataset):self.dataset=dataset
    def __len__(self):return len(self.dataset)
    def __iter__(self):
        ds=self.dataset;rng=np.random.default_rng(ds.t['seed']+ds.epoch)
        if ds.name in ('m5_br','m5_gate','unbounded'):
            n=ds.t['samples_per_case']
            return iter([int(c*n+j) for c in rng.permutation(len(ds.cases)) for j in range(n)])
        return iter(rng.permutation(len(ds)).tolist())
