from torchvision.datasets import CIFAR100

from coarse_cifar100 import CoarseCIFAR100

root = "~/pytorch/"

def test_num_samples():
    original_dataset = dataset = CIFAR100(root, download=True)
    dataset = CoarseCIFAR100(root, download=True)

    assert len(original_dataset) == len(dataset)
    assert len(dataset.class_to_idx) == len(dataset.classes) == 20
    assert all([ 0 <= label <= 19 for label in dataset.targets])
