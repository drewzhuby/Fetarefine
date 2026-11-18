"""Keep generated caches and outputs inside this standalone project."""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parent

def setup_environment():
    for key, rel in {'TMPDIR':'.runtime/tmp','TMP':'.runtime/tmp','TEMP':'.runtime/tmp',
                     'XDG_CACHE_HOME':'.runtime/cache','MPLCONFIGDIR':'.runtime/matplotlib',
                     'TORCH_HOME':'.runtime/torch','PYTHONPYCACHEPREFIX':'.runtime/pycache'}.items():
        path = ROOT / rel
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    import tempfile
    tempfile.tempdir = str(ROOT / '.runtime/tmp')

def inside(path):
    p = Path(path).expanduser()
    p = (ROOT / p).resolve() if not p.is_absolute() else p.resolve()
    if not p.is_relative_to(ROOT):
        raise ValueError(f'Path must stay inside {ROOT}: {p}')
    return p

def output_path(path):
    p = inside(path)
    if p == ROOT or any(p == ROOT/n or p.is_relative_to(ROOT/n) for n in ('data','result','models','configs','training','evaluation')):
        raise ValueError(f'Output must not overwrite source or data assets: {p}')
    return p
