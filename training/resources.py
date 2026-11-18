import json,os,shutil
from pathlib import Path
import torch

def configure(max_voxels,channels,requested_workers=None):
    cpus=len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else (os.cpu_count() or 1)
    p=Path('/sys/fs/cgroup/cpu.max')
    if p.exists():
        q,period=p.read_text().split()
        if q!='max':cpus=min(cpus,max(1,int(q)//int(period)))
    p=Path('/sys/fs/cgroup/cpu/cpu.cfs_quota_us')
    if p.exists() and int(p.read_text())>0:cpus=min(cpus,max(1,int(p.read_text())//int(p.with_name('cpu.cfs_period_us').read_text())))
    try:
        import psutil
        memory=psutil.virtual_memory().available
    except ImportError:
        memory=os.sysconf('SC_PAGE_SIZE')*os.sysconf('SC_PHYS_PAGES')//2
    for lim,use in [('/sys/fs/cgroup/memory.max','/sys/fs/cgroup/memory.current'),('/sys/fs/cgroup/memory/memory.limit_in_bytes','/sys/fs/cgroup/memory/memory.usage_in_bytes')]:
        if Path(lim).exists() and Path(lim).read_text().strip()!='max':
            memory=min(memory,max(0,int(Path(lim).read_text())-int(Path(use).read_text())))
    main_cost=max_voxels*(channels*4+96)+2*1024**3
    if memory < main_cost:raise ValueError(f'Insufficient RAM: estimated main requirement {main_cost}, available {memory}')
    worker_cost=max_voxels*(channels*4+64)+512*1024**2
    safe=max(0,min(2,cpus-2,int((memory-main_cost)//max(1,worker_cost))))
    shm=shutil.disk_usage('/dev/shm').free if Path('/dev/shm').exists() else None
    if shm is not None and shm<512*1024**2:safe=0
    workers=safe if requested_workers is None else requested_workers
    if workers<0 or workers>safe:raise ValueError(f'Requested workers {workers} exceeds conservative capacity {safe}')
    threads=max(1,min(4,cpus-workers-1));torch.set_num_threads(threads)
    info=dict(cpu_budget=cpus,available_memory_bytes=memory,shared_memory_bytes=shm,workers=workers,
              main_threads=threads,worker_threads=1,prefetch_factor=1,estimated_worker_bytes=worker_cost,
              reason='bounded case caches; reserve memory and CPU; log preparation/compute time for throughput review')
    print(json.dumps(info));return info

def init_worker(worker_id):
    import random,numpy as np
    torch.set_num_threads(1);seed=torch.initial_seed()%2**32;random.seed(seed);np.random.seed(seed)
