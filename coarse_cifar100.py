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
        self.fine_idx_to_coarse_idx = {}  # str2int: the original class name to coarse class id
        for coarse_class_idx, coarse_class in enumerate(sorted(self.COARSE_CLASS_TO_FINE_CLASSES.keys())):
            self.coarse_class_to_coarse_idx[coarse_class] = coarse_class_idx

            for fine_class in self.COARSE_CLASS_TO_FINE_CLASSES[coarse_class]:
                self.fine_idx_to_coarse_idx[self.class_to_idx[fine_class]] = coarse_class_idx

        self.classes = list(self.coarse_class_to_coarse_idx.keys())
        self.class_to_idx = self.coarse_class_to_coarse_idx.copy()

        # replace fine-class index with coarse-class index
        self.targets = [self.fine_idx_to_coarse_idx[fine_idx] for fine_idx in self.targets]
