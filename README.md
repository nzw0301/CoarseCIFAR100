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

The following transfrom class provides the same feature above:

```
import numpy as np

from coarse_cifar100 import TransformCIFAR100TOCoarseCIFAR100


training_dataset = CIFAR100(root="~/pytorch_dataset/", download=True)
validation_dataset = CIFAR100(root="~/pytorch_dataset/", download=True, train=False)

num_fine_classes_per_coarse_class = 5

coarse2fine = TransformCIFAR100TOCoarseCIFAR100(num_fine_classes_per_coarse_class)

training_dataset = coarse2fine.fit_transform(training_dataset)
validation_dataset = coarse2fine.fit_transform(validation_dataset)
```

This transform class might be useful when you want to reduce the number of CIFAR100's classes per coarse class by changing `num_fine_classes_per_coarse_class`.
