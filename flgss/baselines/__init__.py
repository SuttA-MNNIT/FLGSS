from .fedavg import run_fedavg
from .fedprox import run_fedprox
from .scaffold import run_scaffold
from .moon import run_moon
from .fedsam import run_fedsam
from .fedkd import run_fedkd
from .fedclustering import run_fedclustering
from .pfedkd import run_pfedkd

def get_baseline_runner(algo_name: str):
    name = algo_name.lower()
    if name == "fedavg":
        return run_fedavg
    elif name == "fedprox":
        return run_fedprox
    elif name == "scaffold":
        return run_scaffold
    elif name == "moon":
        return run_moon
    elif name == "fedsam":
        return run_fedsam
    elif name == "fedkd":
        return run_fedkd
    elif name == "fedclustering":
        return run_fedclustering
    elif name == "pfedkd":
        return run_pfedkd
    else:
        raise ValueError(f"Unknown baseline '{algo_name}'")

__all__ = [
    "run_fedavg",
    "run_fedprox",
    "run_scaffold",
    "run_moon",
    "run_fedsam",
    "run_fedkd",
    "run_fedclustering",
    "run_pfedkd",
    "get_baseline_runner"
]
