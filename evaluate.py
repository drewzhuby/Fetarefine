"""Evaluate a complete prediction cohort in native physical space."""
from runtime import setup_environment,inside,output_path
setup_environment()
import argparse,csv,json
from pathlib import Path
import nibabel as nib
import numpy as np
from tqdm.auto import tqdm
from data.splits import TEST_16_CASES
from data.volumes import raw_file
from data.preprocessing import inverse_map_to_native
from evaluation.metrics import evaluate_case,case_means,METRICS
from training.configuration import sha


def find_native_label(raw_root,case_id):return raw_file(raw_root,case_id,'dseg')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pred_dir',required=True);p.add_argument('--raw_root',default='data/raw');p.add_argument('--data_root',default='data/preprocessed')
    p.add_argument('--output_dir',required=True);p.add_argument('--prediction_grid',choices=['native','pp1'],default='native')
    a=p.parse_args();pred_dir=inside(a.pred_dir);raw=inside(a.raw_root);prep=inside(a.data_root);out=output_path(a.output_dir)
    if out.exists():raise FileExistsError('Fresh evaluation directory required')
    files={case:pred_dir/f'{case}_pred.nii.gz' for case in TEST_16_CASES}
    missing=[str(p) for p in files.values() if not p.exists()]
    if missing:raise FileNotFoundError(f'Incomplete 16-case cohort: {missing}')
    rows=[];per_case=[];hashes={}
    for case in tqdm(TEST_16_CASES,desc='Native evaluation',unit='case'):
        gi=nib.load(str(find_native_label(raw,case)));pi=nib.load(str(files[case]));pred=np.asarray(pi.dataobj)
        if a.prediction_grid=='pp1':
            props=prep/case/'anat/properties.json';metadata=json.loads(props.read_text())
            if not np.allclose(pi.affine,metadata['derived_affine']):raise ValueError('PP1 prediction affine mismatch')
            pred=inverse_map_to_native(pred,props)
            if not np.allclose(gi.affine,metadata['original_affine']):raise ValueError('Native GT affine mismatch')
        elif pi.shape!=gi.shape or not np.allclose(pi.affine,gi.affine,atol=1e-5):raise ValueError(f'Native geometry mismatch: {case}')
        values=evaluate_case(pred,np.asarray(gi.dataobj),gi.affine)
        rows.extend([dict(case=case,**r) for r in values]);per_case.append(dict(case=case,**case_means(values)))
        hashes[case]=dict(prediction=sha(files[case]),label=sha(find_native_label(raw,case)))
    summary={k:float(np.mean([r[k] for r in per_case])) if all(r[k] is not None for r in per_case) else None for k in METRICS}
    summary.update(cases=16,evaluation_grid='native',surface_definition='26-neighbor world-affine; pooled bidirectional distances',
                   distance_failure_count=sum(r['reference_present'] and r['hd95_mm'] is None for r in rows))
    out.mkdir(parents=True)
    for name,values in [('per_case_class',rows),('per_case',per_case)]:
        (out/f'{name}.json').write_text(json.dumps(values,indent=2,allow_nan=False))
        with (out/f'{name}.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False));(out/'INPUT_SHA256.json').write_text(json.dumps(hashes,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
