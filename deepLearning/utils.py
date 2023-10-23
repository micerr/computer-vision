import os
import random
import torch
import numpy as np
import cv2
from typing import Optional
from IPython.display import Image, display
import matplotlib.pyplot as plt
import pandas as pd
import torchvision

# TP8:

def img_path_to_np_flt(fpath: str):
    """returns a numpy float32 array from RGB image path (8-16 bits per component)
    shape: c, h, w
    """
    if not os.path.isfile(fpath):
        raise ValueError(f"File not found {fpath}")
    try:
        rgb_img = cv2.cvtColor(
            cv2.imread(fpath, flags=cv2.IMREAD_COLOR + cv2.IMREAD_ANYDEPTH),
            cv2.COLOR_BGR2RGB,
        ).transpose(2, 0, 1)
    except cv2.error as e:
        print(f"img_path_to_np_flp: error {e} with {fpath}")
        breakpoint()
    if rgb_img.dtype == np.ubyte:
        return rgb_img.astype(np.single) / 255
    elif rgb_img.dtype == np.ushort:
        return rgb_img.astype(np.single) / 65535
    else:
        raise TypeError(
            "img_path_to_np_flt: Error: fpath={fpath} has unknown format ({rgb_img.dtype})"
        )


def img_fpath_to_pt_tensor(fpath: str, batch: bool = True):
    """Open an image file path and convert it to PyTorch tensor."""
    tensor = torch.tensor(img_path_to_np_flt(fpath))
    if batch:
        tensor = tensor.unsqueeze(0)
    return tensor
    
def display_pt_img(tensor_img, zoom: bool = False):
    """Display a tensor image."""
    zoom = True  # forcing zoom=True to work with google colab
    if tensor_img.dim() == 4:
        disp_img = tensor_img.squeeze(0) # squeze remove all the dimentions with size equal to 1
    else:
        disp_img = tensor_img
    disp_img = disp_img.permute(1,2,0) # change the order of the dimensions. Here we put the 3 color dimensions at the end
    if zoom:
        fig = plt.figure()
        plt.imshow(disp_img)
        display(fig)
        plt.close()
    else:
        disp_img = cv2.cvtColor(disp_img.numpy()*255, cv2.COLOR_RGB2BGR)
        disp_img = cv2.imencode('.jpg', disp_img)[1]
        display(Image(data=disp_img))

def get_random_testimg_fpath(category: str = 'misc'):
    """Return the path to a random image in test_images/<category>."""
    testimg_dpath = os.path.join('test_images', category)
    assert os.path.isdir(testimg_dpath), f'Directory does not exist: {testimg_dpath}'
    return os.path.join(testimg_dpath, random.choice(os.listdir(testimg_dpath)))

# Part II of TP9:

def crop_to_multiple(tensor, multiple: int = 64):
    return tensor[
        ...,
        : tensor.size(-2) - tensor.size(-2) % multiple,
        : tensor.size(-1) - tensor.size(-1) % multiple,
    ]

def img_path_to_np_flt(fpath: str):
    """returns a numpy float32 array from RGB image path (8-16 bits per component)
    shape: c, h, w
    """
    if not os.path.isfile(fpath):
        raise ValueError(f"File not found {fpath}")
    try:
        rgb_img = cv2.cvtColor(
            cv2.imread(fpath, flags=cv2.IMREAD_COLOR + cv2.IMREAD_ANYDEPTH),
            cv2.COLOR_BGR2RGB,
        ).transpose(2, 0, 1)
    except cv2.error as e:
        print(f"img_path_to_np_flp: error {e} with {fpath}")
        breakpoint()
    if rgb_img.dtype == np.ubyte:
        return rgb_img.astype(np.single) / 255
    elif rgb_img.dtype == np.ushort:
        return rgb_img.astype(np.single) / 65535
    else:
        raise TypeError(
            "img_path_to_np_flt: Error: fpath={fpath} has unknown format ({rgb_img.dtype})"
        )


def img_fpath_to_pt_tensor(fpath: str, batch: bool = True, crop_to_multiple_of: Optional[int]=None):
    """Open an image file path and convert it to PyTorch tensor."""
    tensor = torch.tensor(img_path_to_np_flt(fpath))
    if crop_to_multiple_of:
        tensor = crop_to_multiple(tensor, crop_to_multiple_of)
    if batch:
        tensor = tensor.unsqueeze(0)
    return tensor
    
def display_pt_img(tensor_img, zoom: bool = False):
    """Display a tensor image."""
    zoom = True  # forcing zoom=True to work with google colab
    if tensor_img.dim() == 4:
        disp_img = tensor_img.squeeze(0)
    else:
        disp_img = tensor_img
    disp_img = disp_img.permute(1,2,0)
    if zoom:
        fig = plt.figure()
        plt.imshow(disp_img)
        display(fig)
        plt.close()
    else:
        disp_img = cv2.cvtColor(disp_img.numpy()*255, cv2.COLOR_RGB2BGR)
        disp_img = cv2.imencode('.jpg', disp_img)[1]
        display(Image(data=disp_img))

def get_random_testimg_fpath(category: str = 'misc'):
    """Return the path to a random image in test_images/<category>."""
    testimg_dpath = os.path.join('test_images', category)
    assert os.path.isdir(testimg_dpath), f'Directory does not exist: {testimg_dpath}'
    return os.path.join(testimg_dpath, random.choice(os.listdir(testimg_dpath)))
    
# Part II of TP9:
class MetricsList():
    def __init__(self, **kwargs):
        self.metrics = kwargs.values()
        self.df = pd.DataFrame(columns=kwargs.keys())
    def update(self, logits, labels):
        _ = [metric.update((logits, labels)) for metric in self.metrics]
    def reset(self):
        _ = [metric.reset() for metric in self.metrics]
    def clear(self):
        self.df = self.df.iloc[0:0]  # Clear Dataframe
    def compute(self, mode):
        self.df.loc[mode] = [metric.compute() for metric in self.metrics]
    def __str__(self):
        return str(self.df)
        
class FashionMNISTDatasetLoader():
    def __init__(self, batch_size=128, Dataset=torchvision.datasets.FashionMNIST, transforms=None):
        transforms = [] if transforms is None else transforms
        # Get training data
        train_data = Dataset(root='./data/FashionMNIST', train=True, download=True,
            transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),  # Move data to a pytorch tensor
                *transforms                         # Apply other transformations
            ])
        )
        # Split train data into training set and validation set
        count = len(train_data)
        indices = list(range(count))
        split = count//10 # Use 10% for validation and 90% for training
        self.training_set = torch.utils.data.Subset(train_data, indices[split:])
        self.validation_set = torch.utils.data.Subset(train_data, indices[:split])

        # Get testing data
        self.testing_set = Dataset(root='./data/FashionMNIST', train=False, download=True,
            transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),  # Move data to a pytorch tensor
                *transforms                         # Apply other transformations
            ])
        )

        self.train_loader = torch.utils.data.DataLoader(self.training_set, batch_size=batch_size, drop_last=True)
        self.val_loader = torch.utils.data.DataLoader(self.validation_set, batch_size=batch_size, drop_last=True)
        self.test_loader = torch.utils.data.DataLoader(self.testing_set, batch_size=batch_size, drop_last=True)