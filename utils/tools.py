import json
import os
import shutil
from collections import OrderedDict
from datetime import datetime
from itertools import repeat
from pathlib import Path

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import yaml
from torchvision import transforms

from datasets.data_loader import WSDataLoader

seed = 42
torch.manual_seed(seed)
cudnn.benchmark = False
cudnn.deterministic = True


def load_config(filename):
    with open(filename, "r") as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    return config


def make_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)


def print_log(info_str, other_info='', file=None):
    current_time = datetime.now().strftime('%Y-%m-%c %H:%M:%S')
    if file:
        print("%s [INFO]: %s %s" % (current_time, info_str, other_info), file=file)
        file.flush()
    else:
        print("%s [INFO]: %s %s" % (current_time, info_str, other_info))


def copy_files(source_paths, des_paths, is_debug=False):
    """
    copy files from source to destination
    """
    for source_path, des_path in zip(source_paths, des_paths):
        if not os.path.exists(os.path.dirname(des_path)):
            os.makedirs(os.path.dirname(des_path))
        shutil.copyfile(source_path, des_path)
        if is_debug:
            print_log("Copy file from %s to %s" % (source_path, des_path))


def get_device(i=0):
    """
    setup GPU device if available, move models into configured device
    """
    if torch.cuda.is_available():
        return torch.device("cuda:%d" % i)
    else:
        return torch.device("cpu")


def get_model_class(model_type="AE", **model_params):
    model_object = __import__("models")
    model_class = getattr(model_object, model_type)
    return model_class(**model_params)


def get_default_transform(input_channels=3, img_size=96):
    if input_channels == 1:
        transform = transforms.Compose([transforms.Grayscale(), transforms.Resize([img_size, img_size]),
                                        transforms.ToTensor()])
    else:
        transform = transforms.Compose([transforms.Resize([img_size, img_size]), transforms.ToTensor()])
    return transform


def get_model_by_state(state_dic_path, model_class, device=get_device()):
    if state_dic_path is not None and os.path.exists(state_dic_path):
        model_class.load_state_dict(torch.load(state_dic_path, map_location=device))
    return model_class


def get_model_opt(state_dic_path, model_class, learning_rate=1e-3, device=get_device()):
    model = get_model_by_state(state_dic_path, model_class, device=device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    return model, optimizer


def get_dataloader(path, chars_include, transform):
    """
    Return dataloader images data in the path
    Args:
        path: path to dataset directory
        chars_include: source name
        transform: the transform functions applied on images

    Returns: dataloader of target datasets

    """
    return WSDataLoader(path, chars_include=chars_include, batch_size=64, transform=transform, return_path=True)


def ensure_dir(d, exist_ok=False):
    if os.path.exists(d) and not exist_ok:
        shutil.rmtree(d)
    os.makedirs(d, exist_ok=exist_ok)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader


def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)
