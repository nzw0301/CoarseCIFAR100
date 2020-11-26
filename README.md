# CoarseCIFAR100


[CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html) dataset has 20 coarse-class (or superclass). Each superclass contains 5 fine-grained classes (or 5 original classes). The details are found in the CIFAR100 page.

This PyTorch's dataset class aims to access the dataset like usual `CIFAR100`.

---

## Usage

```python
from coarse_cifar100 import CoarseCIFAR100

training_dataset = CoarseCIFAR100(root="~/pytorch_dataset/", download=True)
validation_dataset = CoarseCIFAR100(root="~/pytorch_dataset/", download=True, train=False)
```
