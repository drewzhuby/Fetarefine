"""Synthetic CPU smoke check using the real refinement source, independent of MONAI."""
import importlib.util
import json
from pathlib import Path
import torch

if __name__ == '__main__':
    torch.manual_seed(7)
    torch.set_num_threads(2)
    source = Path(__file__).parent/'models/m5_selective.py'
    spec = importlib.util.spec_from_file_location('refinement', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    residual_model = module.BoundedResidualNet()
    gate_model = module.SelectiveGating3DNet()
    a, b = torch.randn(1, 8, 12, 12), torch.randn(1, 8, 12, 12)
    base = (a+b)/2
    refined, residual = residual_model(torch.randn(1,3,12,12), a, b, return_residual=True)
    final, gate = gate_model(torch.randn(1,1,1,12,12), base.unsqueeze(2), refined.unsqueeze(2))
    final.square().mean().backward()
    assert residual.abs().max() <= 2
    assert gate.min() >= 0 and gate.max() <= 1
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in gate_model.parameters())
    print(json.dumps({'synthetic':True,'logits_shape':list(final.shape),
                      'residual_max':float(residual.detach().abs().max()),
                      'gate_mean':float(gate.detach().mean()),'backward':'passed'}))
