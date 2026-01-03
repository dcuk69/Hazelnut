import importlib
packages = ['torch','torchvision','matplotlib','numpy','PIL']
results = {}
for p in packages:
    try:
        m = importlib.import_module(p)
        v = getattr(m, '__version__', getattr(m, 'VERSION', 'unknown'))
        results[p] = ('ok', v)
    except Exception as e:
        results[p] = ('error', str(e))
import torch
results['cuda_available'] = torch.cuda.is_available()
for k,v in results.items():
    print(f"{k}: {v}")
