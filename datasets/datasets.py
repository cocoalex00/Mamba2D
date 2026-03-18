import torch
from PIL import Image
from .imagenet.classes import IMAGENET2012_CLASSES

class ImageNet1k(torch.utils.data.Dataset):
    def __init__(self, root: str, split: str = 'train', transform=None):
        """
        Args:
            root (string): Directory with all the images.
            split (int): which split of the dataset to load.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.data_dir = root
        assert split in ['train', 'val'], 'Split must be either "train" or "val"'
        self.transform = transform
        self.split = split
        self.classes = list(IMAGENET2012_CLASSES)

        # Ground truth file
        if split in ['val']:
            self.gt_file = "datasets/imagenet/ImageNet_val_label.txt"
        else:
            self.gt_file = "datasets/imagenet/ImageNet_train_labels.txt"
        
        # # Open the file with GT labels
        with open(self.gt_file, 'r') as file:
            # Read lines into a list
            gt = file.readlines()

        # Remove newline characters
        self.gt = [line.strip() for line in gt]


    def __len__(self):
        return len(self.gt)

    def __getitem__(self, idx):

        sample = self.gt[idx]
        if self.split == 'train': # Train and test splits have different GT files (ugly)
            folder = sample.split('_')[0]
            path = '/'.join([self.data_dir,folder, sample])
            target = self.classes.index(folder)
            
        elif self.split == 'val':
            file = sample.split(' ')[0]
            
            path = '/'.join([self.data_dir, file])

            label_name = sample.split(' ')[1]
            target = self.classes.index(label_name)

        sample = Image.open(path).convert("RGB") 

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target
