from .partition import partition_data_non_iid, IndexedDataset
from .cifar10 import get_cifar10_data
from .uci_har import get_uci_har_data
from .intel_lab import get_intel_lab_data
from .nbaiot import get_nbaiot_data

def get_dataset(args):
    """
    Factory method to load any of the 4 datasets supported in the benchmark.
    """
    name = args.dataset.lower()
    if name == "cifar10":
        return get_cifar10_data(
            data_path=args.data_path,
            batch_size=args.batch_size,
            test_batch_size=args.test_batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory
        )
    elif name == "uci_har":
        return get_uci_har_data(
            data_path=args.data_path,
            batch_size=args.batch_size,
            test_batch_size=args.test_batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory
        )
    elif name == "intel_lab":
        return get_intel_lab_data(
            data_path=args.data_path,
            batch_size=args.batch_size,
            test_batch_size=args.test_batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory
        )
    elif name == "nbaiot":
        return get_nbaiot_data(
            data_path=args.data_path,
            batch_size=args.batch_size,
            test_batch_size=args.test_batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory
        )
    else:
        raise ValueError(f"Unknown dataset '{name}'. Expected one of: ['cifar10', 'uci_har', 'intel_lab', 'nbaiot']")

__all__ = [
    "partition_data_non_iid",
    "IndexedDataset",
    "get_cifar10_data",
    "get_uci_har_data",
    "get_intel_lab_data",
    "get_nbaiot_data",
    "get_dataset"
]
