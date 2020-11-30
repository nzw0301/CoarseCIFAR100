import numpy as np
from torchvision.datasets import CIFAR100


class CoarseCIFAR100(CIFAR100):
    COARSE_CLASS_TO_FINE_CLASSES = {
        "aquatic mammals": ["beaver", "dolphin", "otter", "seal", "whale"],
        "fish": ["aquarium_fish", "flatfish", "ray", "shark", "trout"],
        "flowers": ["orchid", "poppy", "rose", "sunflower", "tulip"],
        "food containers": ["bottle", "bowl", "can", "cup", "plate"],
        "fruit and vegetables": ["apple", "mushroom", "orange", "pear", "sweet_pepper"],
        "household electrical devices": ["clock", "keyboard", "lamp", "telephone", "television"],
        "household furniture": ["bed", "chair", "couch", "table", "wardrobe"],
        "insects": ["bee", "beetle", "butterfly", "caterpillar", "cockroach"],
        "large carnivores": ["bear", "leopard", "lion", "tiger", "wolf"],
        "large man-made outdoor things": ["bridge", "castle", "house", "road", "skyscraper"],
        "large natural outdoor scenes": ["cloud", "forest", "mountain", "plain", "sea"],
        "large omnivores and herbivores": ["camel", "cattle", "chimpanzee", "elephant", "kangaroo"],
        "medium-sized mammals": ["fox", "porcupine", "possum", "raccoon", "skunk"],
        "non-insect invertebrates": ["crab", "lobster", "snail", "spider", "worm"],
        "people": ["baby", "boy", "girl", "man", "woman"],
        "reptiles": ["crocodile", "dinosaur", "lizard", "snake", "turtle"],
        "small mammals": ["hamster", "mouse", "rabbit", "shrew", "squirrel"],
        "trees": ["maple_tree", "oak_tree", "palm_tree", "pine_tree", "willow_tree"],
        "vehicles 1": ["bicycle", "bus", "motorcycle", "pickup_truck", "train"],
        "vehicles 2": ["lawn_mower", "rocket", "streetcar", "tank", "tractor"]
    }

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):

        super(CoarseCIFAR100, self).__init__(root, train=train, transform=transform, target_transform=target_transform,
                                             download=download)

        self.coarse_class_to_coarse_idx = {}  # str2int: coarse class name to class id in range(20)
        self.fine_idx_to_coarse_idx = {}  # int2int: the original class name to coarse class id
        for coarse_class_idx, coarse_class in enumerate(sorted(self.COARSE_CLASS_TO_FINE_CLASSES.keys())):
            self.coarse_class_to_coarse_idx[coarse_class] = coarse_class_idx

            for fine_class in self.COARSE_CLASS_TO_FINE_CLASSES[coarse_class]:
                self.fine_idx_to_coarse_idx[self.class_to_idx[fine_class]] = coarse_class_idx

        self.classes = list(self.coarse_class_to_coarse_idx.keys())
        self.class_to_idx = self.coarse_class_to_coarse_idx.copy()

        # replace fine-class index with coarse-class index
        self.targets = [self.fine_idx_to_coarse_idx[fine_idx] for fine_idx in self.targets]


class TransformCIFAR100TOCoarseCIFAR100(object):
    COARSE_CLASS_TO_FINE_CLASSES = {
        "aquatic mammals": ["beaver", "dolphin", "otter", "seal", "whale"],
        "fish": ["aquarium_fish", "flatfish", "ray", "shark", "trout"],
        "flowers": ["orchid", "poppy", "rose", "sunflower", "tulip"],
        "food containers": ["bottle", "bowl", "can", "cup", "plate"],
        "fruit and vegetables": ["apple", "mushroom", "orange", "pear", "sweet_pepper"],
        "household electrical devices": ["clock", "keyboard", "lamp", "telephone", "television"],
        "household furniture": ["bed", "chair", "couch", "table", "wardrobe"],
        "insects": ["bee", "beetle", "butterfly", "caterpillar", "cockroach"],
        "large carnivores": ["bear", "leopard", "lion", "tiger", "wolf"],
        "large man-made outdoor things": ["bridge", "castle", "house", "road", "skyscraper"],
        "large natural outdoor scenes": ["cloud", "forest", "mountain", "plain", "sea"],
        "large omnivores and herbivores": ["camel", "cattle", "chimpanzee", "elephant", "kangaroo"],
        "medium-sized mammals": ["fox", "porcupine", "possum", "raccoon", "skunk"],
        "non-insect invertebrates": ["crab", "lobster", "snail", "spider", "worm"],
        "people": ["baby", "boy", "girl", "man", "woman"],
        "reptiles": ["crocodile", "dinosaur", "lizard", "snake", "turtle"],
        "small mammals": ["hamster", "mouse", "rabbit", "shrew", "squirrel"],
        "trees": ["maple_tree", "oak_tree", "palm_tree", "pine_tree", "willow_tree"],
        "vehicles 1": ["bicycle", "bus", "motorcycle", "pickup_truck", "train"],
        "vehicles 2": ["lawn_mower", "rocket", "streetcar", "tank", "tractor"]
    }

    def __init__(self, rnd: np.random.RandomState = None, num_fine_classes_per_coarse_class: int = 5):
        """

        :param rnd: np.random.RandomState instance for reproducibility
        :param num_fine_classes_per_coarse_class: the numper of fine classes per corarse class.
        """

        assert 1 <= num_fine_classes_per_coarse_class <= 5
        self.rnd = rnd
        self.num_fine_classes_per_coarse_class = num_fine_classes_per_coarse_class

        self.used_classes = self.COARSE_CLASS_TO_FINE_CLASSES.copy()
        self.is_fitted = False

        self.coarse_class_to_coarse_idx = {}  # str2int: coarse class name to class id in range(20)
        self.fine_idx_to_coarse_idx = {}  # int2int: the original class id to coarse class id
        self.idx_to_class = {}

    def fit(self, dataset: CIFAR100) -> None:

        assert not self.is_fitted

        # sub-sampling coarse_classes
        if self.num_fine_classes_per_coarse_class != 5:
            assert isinstance(self.rnd, np.random.RandomState)
            sampled_used_classes = {}
            for fine_class in sorted(self.used_classes.keys()):
                coarse_classes = self.used_classes[fine_class]
                self.rnd.shuffle(coarse_classes)
                sampled_used_classes[fine_class] = coarse_classes[:self.num_fine_classes_per_coarse_class]
            self.used_classes = sampled_used_classes

        for coarse_class_idx, coarse_class in enumerate(sorted(self.used_classes.keys())):
            self.coarse_class_to_coarse_idx[coarse_class] = coarse_class_idx

            for fine_class in self.used_classes[coarse_class]:
                self.fine_idx_to_coarse_idx[dataset.class_to_idx[fine_class]] = coarse_class_idx

        self.idx_to_class = {i: _class for _class, i in dataset.class_to_idx.items()}
        self.is_fitted = True

    def fit_transform(self, dataset: CIFAR100) -> CIFAR100:
        self.fit(dataset)
        return self.transform(dataset)

    def transform(self, dataset) -> CIFAR100:

        assert self.is_fitted

        # replace fine-class index with coarse-class index
        if self.num_fine_classes_per_coarse_class < 5:
            used_fine_classes = []
            for fine_classes in self.used_classes.values():
                for fine_class in fine_classes:
                    used_fine_classes.append(fine_class)

            used_fine_classes = set(used_fine_classes)

            targets = []
            data = []
            for x, fine_idx in zip(dataset.data, dataset.targets):
                if self.idx_to_class[fine_idx] in used_fine_classes:
                    targets.append(self.fine_idx_to_coarse_idx[fine_idx])
                    data.append(x)
            dataset.data = data
            dataset.targets = targets
        else:
            dataset.targets = [self.fine_idx_to_coarse_idx[fine_idx] for fine_idx in dataset.targets]

        dataset.classes = list(self.coarse_class_to_coarse_idx.keys())
        dataset.class_to_idx = self.coarse_class_to_coarse_idx.copy()

        return dataset
