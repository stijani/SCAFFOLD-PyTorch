# super_class: [sub_classes]
CIFAR100_RELABEL_DICT = {
    0: ["beaver", "dolphin", "otter", "seal", "whale"],
    1: ["aquarium_fish", "flatfish", "ray", "shark", "trout"],
    2: ["orchid", "poppy", "rose", "sunflower", "tulip"],
    3: ["bottle", "bowl", "can", "cup", "plate"],
    4: ["apple", "mushroom", "orange", "pear", "sweet_pepper"],
    5: ["clock", "keyboard", "lamp", "telephone", "television",],
    6: ["bed", "chair", "couch", "table", "wardrobe"],
    7: ["bee", "beetle", "butterfly", "caterpillar", "cockroach"],
    8: ["bear", "leopard", "lion", "tiger", "wolf"],
    9: ["cloud", "forest", "mountain", "plain", "sea"],
    10: ["bridge", "castle", "house", "road", "skyscraper",],
    11: ["camel", "cattle", "chimpanzee", "elephant", "kangaroo",],
    12: ["fox", "porcupine", "possum", "raccoon", "skunk"],
    13: ["crab", "lobster", "snail", "spider", "worm"],
    14: ["baby", "boy", "girl", "man", "woman"],
    15: ["crocodile", "dinosaur", "lizard", "snake", "turtle"],
    16: ["hamster", "mouse", "rabbit", "shrew", "squirrel"],
    17: ["maple_tree", "oak_tree", "palm_tree", "pine_tree", "willow_tree"],
    18: ["bicycle", "bus", "motorcycle", "pickup_truck", "train"],
    19: ["lawn_mower", "rocket", "streetcar", "tank", "tractor"],
}


MEAN = {
    "mnist": 0.1307,
    "cifar10": (0.4914, 0.4822, 0.4465),
    "cifar100": (0.5071, 0.4865, 0.4409),
    "femnist": 0,  # dummy code, it's useless
    "synthetic": 0,  # dummy code, it's useless
    "emnist": 0.1736,
    "fmnist": 0.2860,
}

STD = {
    "mnist": 0.3015,
    "cifar10": (0.2023, 0.1994, 0.2010),
    "cifar100": (0.2009, 0.1984, 0.2023),
    "femnist": 1.0,  # dummy code, it's useless
    "synthetic": 1.0,  # dummy code, it's useless
    "emnist": 0.3248,
    "fmnist": 0.3205,
}
