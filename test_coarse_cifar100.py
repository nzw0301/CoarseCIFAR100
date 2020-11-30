import numpy as np
from torchvision.datasets import CIFAR100

from coarse_cifar100 import CoarseCIFAR100, TransformCIFAR100TOCoarseCIFAR100

root = "~/pytorch/"

def test_num_samples():
    original_dataset = dataset = CIFAR100(root, download=True)
    dataset = CoarseCIFAR100(root, download=True)

    assert len(original_dataset) == len(dataset)
    assert len(dataset.class_to_idx) == len(dataset.classes) == 20
    assert all([ 0 <= label <= 19 for label in dataset.targets])


def test_transform_without_subsampling():
    train_dataset = CIFAR100(root, download=True)
    val_dataset = CIFAR100(root, download=True, train=False)

    fine2coarse = TransformCIFAR100TOCoarseCIFAR100(num_fine_classes_per_coarse_class=5)
    train_dataset = fine2coarse.fit_transform(train_dataset)

    assert len(set(train_dataset.targets)) == 20
    assert len(train_dataset) == 50_000
    assert len(train_dataset.classes) == 20

    val_dataset = fine2coarse.transform(val_dataset)
    assert len(set(val_dataset.targets)) == 20
    assert len(val_dataset) == 10_000
    assert len(val_dataset.classes) == 20

def test_transfrom_with_subsampling():

    for i in range(1, 6):
        train_dataset = CIFAR100(root, download=True)
        val_dataset = CIFAR100(root, download=True, train=False)
        fine2coarse = TransformCIFAR100TOCoarseCIFAR100(num_fine_classes_per_coarse_class=i, rnd=np.random.RandomState(7))
        train_dataset = fine2coarse.fit_transform(train_dataset)

        assert len(set(train_dataset.targets)) == 20
        assert len(train_dataset) == 20 * i * 500
        assert len(train_dataset.classes) == 20

        val_dataset = fine2coarse.transform(val_dataset)
        assert len(set(val_dataset.targets)) == 20
        assert len(val_dataset) == 20 * i * 100
        assert len(val_dataset.classes) == 20
