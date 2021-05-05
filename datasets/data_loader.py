import os
from PIL import Image
import numpy as np
import torch
from collections import OrderedDict
from torch.utils.data import DataLoader, Sampler, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder


class WSDataset(Dataset):
    def __init__(self, ws_dataset_path, chars_include=None, img_size=96, return_path=False, transform=None):
        super().__init__()
        if not chars_include:
            chars_include = os.listdir(ws_dataset_path)
        if not transform:
            transform = transforms.Compose([
                # transforms.ToTensor(),
                # transforms.Resize((img_size, img_size)),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
            )
        self.images_path, self.targets, self.class_to_idx = [], [], OrderedDict()
        self.classes = []
        self.img_size, self.return_path, self.transform = img_size, return_path, transform
        for char in chars_include:
            self.class_to_idx[char] = len(self.class_to_idx)
            self.classes.append(char)
            char_path = os.path.join(ws_dataset_path, char)
            image_files = [os.path.join(char_path, file) for file in os.listdir(char_path)]
            self.images_path.extend(image_files)
            self.targets.extend([self.class_to_idx.get(char) for _ in image_files])

    def __getitem__(self, index):
        # read in image
        image = np.asarray(Image.open(self.images_path[index]).convert("RGB")) / 255
        # swap color axis because numpy image is (H, W, C), but torch image is (C, H, W)
        image = image.transpose((2, 0, 1))
        # convert numpy array to torch tensor
        image = self.transform(torch.tensor(image, dtype=torch.float))
        target = torch.tensor(self.targets[index], dtype=torch.long)
        if self.return_path:
            return image, target, self.images_path[index]
        return image, target

    def __len__(self):
        return len(self.images_path)


class AncientDataLoader(DataLoader):
    def __init__(self, root_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=0, transform=None):
        if not transform:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        self.dataset = ImageFolder(root_dir, transform)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class CharSampler(Sampler):

    def __init__(self, sampler, batch_size: int, drop_last: bool) -> None:
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        for char_batch in self.sampler:
            batch = []
            for idx in char_batch:
                batch.append(idx)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
            if len(batch) > 0 and not self.drop_last:
                yield batch

    def __len__(self):
        return len(self.sampler)


class WSDataLoader(DataLoader):
    def __init__(self, root_dir, batch_size, num_workers=0, transform=None, chars_include=None, img_size=96,
                 return_path=False):
        if not transform:
            transform = transforms.Compose([
                # transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        self.dataset = WSDataset(root_dir, chars_include, img_size, return_path, transform)
        self.init_kwargs = {
            'dataset': self.dataset,
            "batch_size": batch_size,
            'num_workers': num_workers
        }
        super().__init__(**self.init_kwargs)


# data_loader = WSDataLoader("../datasets/ancient_5_ori/jia", 32, chars_include=['ㄗ', '一', '三', '上', '丁'])
# batches = [b for b in data_loader]

