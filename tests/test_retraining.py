import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from runtime import setup_environment,ROOT,output_path
setup_environment()
import unittest,tempfile,json,subprocess,copy
from unittest.mock import patch
import numpy as np
import torch
import nibabel as nib
from torch.utils.data import DataLoader
from training.configuration import load_config,build_model,sha,load_checkpoint
from training.trainer import training_loss,Trainer,validate_cases
from data.training_dataset import TrainingDataset,CaseOrderSampler
from data.preprocessing import inverse_map_to_native
from data.volumes import load_logits,raw_file
from evaluation.metrics import evaluate_case,case_means
from inference import predict_volume
from models.m5_selective import BoundedResidualNet,SelectiveGating3DNet
from models.ablation_unbounded import UnboundedResidualNet

torch.set_num_threads(2)

class Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=ROOT/'.runtime/tmp');self.r=Path(self.tmp.name);self.prep=self.r/'prep';self.raw=self.r/'raw'
        self.case='sub-001';self.x=np.random.default_rng(1).normal(size=(9,11,3)).astype('float32');self.y=np.ones(self.x.shape,dtype='uint8')*2
        f=self.prep/self.case/'anat';f.mkdir(parents=True);self.aff=np.diag([.5,.5,.5,1.])
        for name,a in [('image',self.x),('label',self.y),('valid_mask',np.ones_like(self.y))]:nib.save(nib.Nifti1Image(a,self.aff),f/f'{name}.nii.gz')
        self.props=dict(derived_shape=list(self.x.shape),derived_affine=self.aff.tolist(),original_shape=list(self.x.shape),original_affine=self.aff.tolist())
        (f/'properties.json').write_text(json.dumps(self.props))
        rf=self.raw/self.case/'anat';rf.mkdir(parents=True);nib.save(nib.Nifti1Image(self.y,self.aff),rf/f'{self.case}_rec-irtk_dseg.nii.gz')
        self.deep=self.r/'deep';self.swin=self.r/'swin';self.lr=self.r/'br';self.prior=self.r/'prior'
        for directory,name,identity,parents in [(self.deep,'m2','deep',{}),(self.swin,'m0','swin',{}),(self.lr,'m5_br','br',{'deep':'deep','swin':'swin'}),(self.prior,'m0','prior',{})]:
            directory.mkdir();a=np.zeros((8,*self.x.shape),dtype='float32');a[2]=2
            if name=='m5_br':a[2]=3
            p=directory/f'{self.case}_logits.npy';np.save(p,a)
            (directory/f'{self.case}_logits.json').write_text(json.dumps(dict(case=self.case,model_id=name,phase='phase2',shape=list(a.shape),affine=self.aff.tolist(),checkpoint_sha256=identity,parents=parents,array_sha256=sha(p),image_sha256=sha(f/'image.nii.gz'))))
    def tearDown(self):self.tmp.cleanup()
    def ds(self,name):
        c=load_config(name);t=c['training'];t.update(tile_size=[16,16],patch_size=[8,8,8],samples_per_case=4,minimum_slice_foreground=1,batch_size=2)
        d={'prior':self.prior} if name=='m3' else {'deep':self.deep,'swin':self.swin} if name in ('m5_br','unbounded','m5_gate') else {}
        if name=='m5_gate':d['lr']=self.lr
        return c,TrainingDataset(self.prep,[self.case],c,d),d
    def test_configs_and_budgets(self):
        for name,budget in [('m5_br',30720),('unbounded',30720),('m5_gate',15360)]:
            c=load_config(name);t=c['training'];self.assertEqual(64*t['samples_per_case']//t['batch_size']*t['max_epochs'],budget)
        self.assertEqual(load_config('m0')['training']['max_epochs'],300)
    def test_cache_integrity_and_geometry(self):
        load_logits(self.deep,self.case,self.prep,'m2')
        with self.assertRaises(ValueError):load_logits(self.deep,self.case,self.prep,'m0')
        p=self.deep/f'{self.case}_logits.npy';a=np.load(p);a[0]=10;np.save(p,a)
        with self.assertRaises(ValueError):load_logits(self.deep,self.case,self.prep,'m2')
    def test_raw_discovery_ambiguity(self):
        self.assertIn('irtk',raw_file(self.raw,self.case,'dseg').name)
        f=self.raw/self.case/'anat'/f'{self.case}_rec-mial_dseg.nii.gz';f.touch()
        with self.assertRaises(ValueError):raw_file(self.raw,self.case,'dseg')
    def test_geometry_identity_and_translation(self):
        p=self.prep/self.case/'anat/properties.json';self.assertTrue(np.array_equal(inverse_map_to_native(self.y,p),self.y))
        props=copy.deepcopy(self.props);props['original_affine'][0][3]=.5;p.write_text(json.dumps(props));actual=inverse_map_to_native(self.y,p)
        self.assertTrue(np.all(actual[:-1]==2));self.assertTrue(np.all(actual[-1]==0))
    def test_native_surface_affine(self):
        p=np.zeros((5,5,5),dtype='uint8');g=p.copy();p[1,2,2]=1;g[2,2,2]=1
        aff=np.eye(4);aff[0,0]=2;v=evaluate_case(p,g,aff)[0];self.assertEqual(v['hd95_mm'],2);self.assertEqual(v['nsd'],0)
        v=evaluate_case(g,g,aff)[0];self.assertEqual(v['dice'],1);self.assertEqual(v['hd95_mm'],0)
        self.assertIsNone(case_means(evaluate_case(np.zeros_like(g),g,aff))['hd95_mm'])
    def test_residual_training_shapes_loss_gradient(self):
        for name,cls in [('m5_br',BoundedResidualNet),('unbounded',UnboundedResidualNet)]:
            c,ds,_=self.ds(name);batch=next(iter(DataLoader(ds,batch_size=2)));self.assertEqual(batch['image'].shape,(2,3,16,16))
            model=cls();self.assertEqual(sum(p.numel() for p in model.parameters()),7512)
            loss=training_loss(model,name,batch,c);loss.backward();self.assertTrue(model.head.weight.grad.abs().sum()>0)
    def test_gate_missing_refuses_and_real_gradient(self):
        c,ds,_=self.ds('m5_gate');b=next(iter(DataLoader(ds,batch_size=2)));model=SelectiveGating3DNet();self.assertEqual(sum(p.numel() for p in model.parameters()),14424)
        loss=training_loss(model,'m5_gate',b,c);loss.backward();self.assertTrue(model.head.bias.grad.abs().sum()>0)
        b.pop('lr')
        with self.assertRaises(ValueError):training_loss(model,'m5_gate',b,c)
    def test_m3_passes_prior(self):
        c,ds,_=self.ds('m3');b=next(iter(DataLoader(ds,batch_size=2)))
        class Prior(torch.nn.Module):
            def __init__(self):super().__init__();self.conv=torch.nn.Conv2d(11,8,1)
            def forward(self,x,prior):return self.conv(x)+prior
        loss=training_loss(Prior(),'m3',b,c);self.assertTrue(torch.isfinite(loss))
    def test_diffunet_training_dispatch(self):
        c,ds,_=self.ds('diffunet');b=next(iter(DataLoader(ds,batch_size=2)))
        class Diff(torch.nn.Module):
            def __init__(self):super().__init__();self.conv=torch.nn.Conv3d(1,7,1)
            def forward(self,x,target):
                assert target.shape[1]==7
                return self.conv(x)
        loss=training_loss(Diff(),'diffunet',b,c);loss.backward()
    def test_sampler_reproducibility(self):
        _,ds,_=self.ds('m5_br');ds.epoch=2;a=ds[0];b=ds[0]
        self.assertTrue(all(torch.equal(a[k],b[k]) for k in a));self.assertEqual(list(CaseOrderSampler(ds)),[0,1,2,3])
    def test_gate_tiling_matches_full(self):
        rng=np.random.default_rng(4);x=rng.normal(size=(67,5,4)).astype('float32');a=rng.normal(size=(8,*x.shape)).astype('float32');b=a+.4
        model=SelectiveGating3DNet();torch.nn.init.normal_(model.head.weight,std=.1)
        actual=predict_volume(model,'m5_gate',x,'cpu',deep=a,swin=a,lr=b)
        with torch.no_grad():expected=model(torch.tensor(x)[None,None],torch.tensor(a)[None],torch.tensor(b)[None])[0][0].numpy()
        np.testing.assert_allclose(actual,expected,atol=1e-6,rtol=1e-6)
    def test_full_native_validation(self):
        class Constant(torch.nn.Module):
            def forward(self,x):
                out=torch.zeros((x.shape[0],8,*x.shape[2:]));out[:,2]=2;return out
        self.assertEqual(validate_cases(Constant(),load_config('m1'),self.prep,[self.case],'cpu',{},self.raw),1.)
    def test_dice_selection_and_checkpoint(self):
        c,ds,d=self.ds('m5_br');c['model_id']='m2';c['training'].update(max_epochs=2,val_every=1,min_epochs=1,patience_checks=10,lr_schedule='constant')
        # Exercise trainer with a small differentiable network and full-volume validator mock.
        ds.name='m5_br';ds.directories={};ds.cache.clear()
        model=torch.nn.Conv2d(3,8,1);out=self.r/'training';out.mkdir()
        loader=DataLoader(ds,batch_size=2)
        with patch('training.trainer.validate_cases',side_effect=[.9,.8]):
            Trainer(model,loader,c,out,'cpu','phase1',self.prep,[self.case],{}, {},{},2,self.raw).fit()
        sel=json.loads((out/'SELECTION.json').read_text());self.assertEqual(sel['best_epoch'],1);self.assertEqual(sel['validation_dice'],.9)
    def test_cli_actual_residual_prediction(self):
        c=load_config('m5_br');model=BoundedResidualNet();p=self.r/'br.pt'
        torch.save(dict(format='feta-clean-v2',config=c,phase='phase2',model_state_dict=model.state_dict(),parents={'deep':'deep','swin':'swin'}),p)
        out=self.r/'prediction'
        cmd=[sys.executable,'-B',str(ROOT/'infer.py'),'--model','m5_br','--checkpoint',str(p),'--data_root',str(self.prep),'--output_dir',str(out),'--case',self.case,'--expert_3d_dir',str(self.swin),'--expert_2p5d_dir',str(self.deep),'--device','cpu']
        result=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True);self.assertEqual(result.returncode,0,result.stderr)
        pred=np.asarray(nib.load(out/f'{self.case}_pred.nii.gz').dataobj);self.assertTrue(np.all(pred==2));self.assertTrue((out/'COMPLETE.json').exists())
    def test_fresh_phase1_phase2_entrypoint(self):
        import yaml,shutil
        import train
        from data.splits import TRAIN_64_CASES
        # Synthetic 64-case cohort; T16 is never required for training or selection.
        for case in TRAIN_64_CASES:
            if case==self.case:continue
            f=self.prep/case/'anat';f.mkdir(parents=True)
            for src in (self.prep/self.case/'anat').iterdir():shutil.copyfile(src,f/src.name)
            rf=self.raw/case/'anat';rf.mkdir(parents=True)
            nib.save(nib.Nifti1Image(self.y,self.aff),rf/f'{case}_rec-mial_dseg.nii.gz')
        cfg=yaml.safe_load((ROOT/'configs/m1_deeplab_2d.yaml').read_text())
        cfg['model']['pretrained']=False
        cfg['training'].update(max_epochs=2,min_epochs=1,val_every=1,patience_checks=10,tile_size=[16,16],minimum_slice_foreground=1)
        cp=self.r/'synthetic.yaml';cp.write_text(yaml.safe_dump(cfg))
        initial=[]
        def factory(c,initialize=True):
            m=torch.nn.Conv2d(3,8,1);initial.append(m.weight.detach().clone());return m
        common=['train.py','--model','m1','--config',str(cp),'--data_root',str(self.prep),'--raw_root',str(self.raw),'--device','cpu','--workers','0']
        with patch('train.build_model',side_effect=factory):
            with patch.object(sys,'argv',common+['--phase','phase1','--output_dir',str(self.r/'dev')]):train.main()
            with patch.object(sys,'argv',common+['--phase','phase2','--selection',str(self.r/'dev/SELECTION.json'),'--output_dir',str(self.r/'final')]):train.main()
        self.assertTrue(torch.equal(initial[0],initial[1]))
        result=torch.load(self.r/'final/final.pt',weights_only=False)
        selection=json.loads((self.r/'dev/SELECTION.json').read_text())
        self.assertEqual(result['epoch'],selection['best_epoch']);self.assertEqual(len(result['identity']['train_cases']),64)
    def test_complete_evaluation_cli(self):
        from data.splits import TEST_16_CASES
        pred=self.r/'pred';pred.mkdir()
        for case in TEST_16_CASES:
            rf=self.raw/case/'anat';rf.mkdir(parents=True)
            nib.save(nib.Nifti1Image(self.y,self.aff),rf/f'{case}_rec-irtk_dseg.nii.gz')
            nib.save(nib.Nifti1Image(self.y,self.aff),pred/f'{case}_pred.nii.gz')
        cmd=[sys.executable,'-B',str(ROOT/'evaluate.py'),'--pred_dir',str(pred),'--raw_root',str(self.raw),'--output_dir',str(self.r/'eval')]
        p=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True);self.assertEqual(p.returncode,0,p.stderr)
        summary=json.loads((self.r/'eval/summary.json').read_text());self.assertEqual(summary['dice'],1);self.assertEqual(summary['cases'],16)
        (pred/f'{TEST_16_CASES[0]}_pred.nii.gz').unlink()
        cmd[-1]=str(self.r/'eval_missing');p=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True)
        self.assertNotEqual(p.returncode,0);self.assertIn('Incomplete 16-case cohort',p.stderr)
    def test_output_containment(self):
        for p in ['/tmp/new_run',ROOT/'data/test',ROOT/'result/new']:
            with self.assertRaises(ValueError):output_path(p)

if __name__=='__main__':unittest.main(verbosity=2)
