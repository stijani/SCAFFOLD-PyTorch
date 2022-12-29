from torchvision import transforms


AUGMENTATIONS = {
    "cifar10": transforms.Compose([
        transforms.RandomCrop(32,padding=4), # Data augmentation
        transforms.RandomHorizontalFlip(),  # Data augmentation
        transforms.ToTensor(),
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]), 
    "mnist": transforms.Compose([])
}