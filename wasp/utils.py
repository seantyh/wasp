
try:
    get_ipython()
    from tqdm import tqdm_notebook as tqdm    
except NameError:
    from tqdm import tqdm
from pathlib import Path

def install_cache_dir(name):
    cache_dir = Path(__file__).parent / f"../data/{name}"
    cache_dir.mkdir(parents=True, exist_ok=True)

def get_cache_path(name, path):
    cache_dir = Path(__file__).parent / f"../data/{name}"
    return cache_dir.joinpath(path).resolve()

def get_resource_path(name, path):
    cache_dir = Path(__file__).parent / f"../resources/{name}"
    return cache_dir.joinpath(path).resolve()

def get_exp_dir(cache_dir, exp_name):
    cache_dir = get_cache_path(cache_dir, "")
    exp_dir = cache_dir / exp_name
    return exp_dir.resolve()