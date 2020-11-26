# CoarseCIFAR100


[CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html) dataset has 20 coarse-classes (or superclasses). Each coarse-class contains 5 fine-grained classes (or 5 original classes). The details are found in the CIFAR100 page.

This PyTorch's dataset class aims to access the dataset like usual `CIFAR100`.

---

## Usage

```python
from coarse_cifar100 import CoarseCIFAR100

training_dataset = CoarseCIFAR100(root="~/pytorch_dataset/", download=True)
validation_dataset = CoarseCIFAR100(root="~/pytorch_dataset/", download=True, train=False)

print(training_._dataset.class_to_idx)

{
    'aquatic mammals': 0,
    'fish': 1,
    'flowers': 2,
    'food containers': 3,
    'fruit and vegetables': 4,
    'household electrical devices': 5,
    'household furniture': 6,
    'insects': 7,
    'large carnivores': 8,
    'large man-made outdoor things': 9,
    'large natural outdoor scenes': 10,
    'large omnivores and herbivores': 11,
    'medium-sized mammals': 12,
    'non-insect invertebrates': 13,
    'people': 14,
    'reptiles': 15,
    'small mammals': 16,
    'trees': 17,
    'vehicles 1': 18,
    'vehicles 2': 19
}

```
