import numpy as np
import os
import torch
import pandas as pd
from PIL import Image
from torchvision.transforms import transforms

from datasets.dataset_manager import download_dataset
from helper import ModelConfiguration
from torch.utils.data import DataLoader, Dataset


def load_image(image_path, transform):
    # read in image
    image = np.asarray(Image.open(image_path).convert("RGB")) / 255
    # swap color axis because numpy image is (H, W, C), but torch image is (C, H, W)
    image = image.transpose((2, 0, 1))
    # convert numpy array to torch tensor
    image = transform(torch.tensor(image, dtype=torch.float))
    return image


def load_paths(path_dir):
    for root, dirs, files in os.walk(path_dir):
        if not dirs:
            target_paths = [f"{root}/{file}" for file in files]
            return target_paths


class WSDataset(Dataset):
    def __init__(self, ws_dir, chars_included, transform):
        self.transform = transform
        self.ws_df = pd.DataFrame(columns=["ws_paths", "label"])
        self.classes = {i: char for i, char in enumerate(chars_included)}
        for char in chars_included:
            char_df = {c: [] for c in self.ws_df.columns}
            # load target paths
            ws_paths = load_paths(f"{ws_dir}/{char}")
            char_df["label"] = [char for _ in range(len(ws_paths))]
            char_df["ws_paths"] = ws_paths
            # add to dataframe object
            self.ws_df = self.ws_df.append(pd.DataFrame(char_df), ignore_index=True)

    def __getitem__(self, item):
        # read in target image
        ws_image = load_image(self.ws_df.loc[item, "ws_paths"], self.transform)
        # read in target image
        label = self.ws_df.loc[item, "label"]
        return ws_image, label

    def __len__(self):
        return len(self.ws_df["label"])


class AWSDataset(Dataset):
    def __init__(self, target_dir, source_dir, chars_included, transform):
        self.transform = transform
        self.aws_df = pd.DataFrame(columns=["target_paths", "source_paths", "label"])
        self.classes = {i: char for i, char in enumerate(chars_included)}
        for char in chars_included:
            char_df = {c: [] for c in self.aws_df.columns}
            # load target paths
            char_df["target_paths"] = load_paths(f"{target_dir}/{char}")
            char_df["label"] = [char for _ in range(len(char_df["target_paths"]))]
            # load source paths
            char_df["source_paths"] = load_paths(f"{source_dir}/{char}")
            assert len(char_df["source_paths"]) == len(char_df["target_paths"]), \
                "The length of source images should equal to target images."
            # add to dataframe object
            self.aws_df = self.aws_df.append(pd.DataFrame(char_df), ignore_index=True)

    def __getitem__(self, item):
        # read in target image
        target_image = load_image(self.aws_df.loc[item, "target_paths"], self.transform)
        # read in target image
        source_image = load_image(self.aws_df.loc[item, "source_paths"], self.transform)
        label = torch.tensor(self.aws_df.loc[item, "label"], dtype=torch.long)
        return target_image, source_image, label

    def __len__(self):
        return len(self.aws_df["label"])


class AncientDataset:

    def __init__(self, train_chars=None, val_chars=None, conf=None, transform=None, root_dir="./", batch_size=64):
        conf = ModelConfiguration(model_params={"input_size": 96, "in_channels": 3}) if conf is None else conf
        self.conf = conf
        # define original and expansion path
        self.ancient_train_dir = os.path.join(root_dir, "ancient_3_exp")
        self.ancient_val_dir = os.path.join(root_dir, "ancient_5_ori")

        if not os.path.exists(self.ancient_train_dir) or not os.path.exists(self.ancient_val_dir):
            download_dataset(root_dir)

        self.target_name, self.source_name = conf.paired_chars[0], conf.paired_chars[1]
        # training root directory
        self.target_train_dir = os.path.join(self.ancient_train_dir, self.target_name)
        self.source_train_dir = os.path.join(self.ancient_train_dir, self.source_name)

        # validation root directory
        self.source_val_dir = os.path.join(self.ancient_val_dir, self.source_name)
        self.target_val_dir = os.path.join(self.ancient_val_dir, self.target_name)
        # define two types of character list
        self.shared_chars = set(os.listdir(self.source_val_dir)) & set(os.listdir(self.target_val_dir))
        self.source_full_chars = os.listdir(self.source_val_dir)
        self.transform = self.load_transform() if not transform else transform
        # define adjacent training dataset and loader
        if train_chars:
            self.train_dataset = AWSDataset(self.target_train_dir, self.source_train_dir, train_chars, self.transform)
            self.aws_train_loader = DataLoader(self.train_dataset, batch_size=batch_size)
        # define validation target dataset and loader
        if val_chars:
            self.target_dataset = WSDataset(self.target_val_dir, val_chars, self.transform)
            self.target_loader = DataLoader(self.target_dataset, batch_size=batch_size)
        # define validation shared, full source dataset and loader.
        self.source_shared_dataset = WSDataset(self.source_val_dir, self.shared_chars, self.transform)
        self.source_shared_loader = DataLoader(self.source_shared_dataset, batch_size=batch_size)
        self.source_full_dataset = WSDataset(self.source_val_dir, self.source_full_chars, self.transform)
        self.source_full_loader = DataLoader(self.source_full_dataset, batch_size=batch_size)

    def load_transform(self):
        input_size, in_channels = self.conf.model_params["input_size"], self.conf.model_params["in_channels"]
        if in_channels == 1:
            transform = transforms.Compose([transforms.Grayscale(), transforms.Resize([input_size, input_size])])
        else:
            transform = transforms.Compose([transforms.Resize([input_size, input_size]),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        return transform
