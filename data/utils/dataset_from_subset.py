import torch
from torch.utils.data import Dataset, TensorDataset, random_split
from torchvision import transforms

class DatasetFromSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        self.indices = subset.indices
        self.dataset = subset.dataset

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


class DatasetFromSubset_(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        self.indices = subset.indices
        self.dataset = subset.dataset

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


class DatasetFromSubset(Dataset):
    def __init__(self, subset, transform=None, use_aug=True):
        # features, labels = dataset 
        # self.use_aug = use_aug
        # assert features.shape[0] == labels.shape[0]
        # self.features = features
        # self.labels = labels
        # self.transforms = transform

        self.subset = subset
        self.transform = transform
        self.use_aug = use_aug
        self.indices = subset.indices
        self.dataset = subset.dataset

    def __getitem__(self, index):
        img, label = self.subset[index]
         # convert the image to channel first if needed
        if img.shape[0] > 3:
            img = np.transpose(img, (2, 0, 1))
        # transform the image or just normalize
        if self.use_aug and self.transforms:
            # first convert image from numpy array to a pil image
            img = self.numpy_2_pil_image(img) 
            img = self.transforms(img)
        else:
            # no transform, just normalize
            img = torch.Tensor((img / 255.))
            #label = torch.Tensor(label).long()
        return img, label

    def __len__(self):
        return len(self.subset)

    def numpy_2_pil_image(self, numpy_img):
        # convert to channel first if array is channel last
        if numpy_img.shape[-1] != 3:
            # pil only deals with channel last format 
            numpy_img = np.transpose(numpy_img, (1, 2, 0))
        img_pil = Image.fromarray(numpy_img.astype(np.uint8))
        return img_pil