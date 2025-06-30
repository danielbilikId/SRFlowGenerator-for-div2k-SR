import os
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from math import floor
from utils.preprocessing import preprocessing
from config import (
    RATIO, RESIZE_HEIGHT, RESIZE_WIDTH,
    TRAIN_VAL_SPLIT_PERC, VAL_TEST_SPLIT_PERC,
    TRAIN_HR_DIR, VALID_HR_DIR,
    TENSOR_X_PATH, TENSOR_Y_PATH, VAL_TENSOR_X_PATH, VAL_TENSOR_Y_PATH
)
from config import BATCH_SIZE

class DIV2KDataset(Dataset):
    def __init__(self, img_paths, data_type="train"):
        self.img_paths = img_paths
        self.data_type = data_type
        self.lr_images = []
        self.hr_images = []

        self._load_or_preprocess_data()

    def _load_or_preprocess_data(self):
        if self.data_type == "train":
            x_path = TENSOR_X_PATH
            y_path = TENSOR_Y_PATH
            target_resize_h = RESIZE_HEIGHT
            target_resize_w = RESIZE_WIDTH
        elif self.data_type == "val":
            x_path = VAL_TENSOR_X_PATH
            y_path = VAL_TENSOR_Y_PATH
            target_resize_h = RESIZE_HEIGHT * 2
            target_resize_w = RESIZE_WIDTH * 2
        else: 
            x_path = VAL_TENSOR_X_PATH
            y_path = VAL_TENSOR_Y_PATH
            target_resize_h = RESIZE_HEIGHT * 2
            target_resize_w = RESIZE_WIDTH * 2

        if os.path.exists(x_path) and os.path.exists(y_path):
            print(f"Loading preprocessed {self.data_type} data from {x_path} and {y_path}")
            self.lr_images = np.load(x_path)
            self.hr_images = np.load(y_path)
            self.lr_images = torch.from_numpy(self.lr_images).float()
            self.hr_images = torch.from_numpy(self.hr_images).float()
            if self.lr_images.shape[-1] == 3:
                self.lr_images = self.lr_images.permute(0, 3, 1, 2)
                self.hr_images = self.hr_images.permute(0, 3, 1, 2)
        else:
            print(f"Preprocessing {self.data_type} images...")
            img_lr_list = []
            img_hr_list = []
            for i in tqdm(range(len(self.img_paths))):
                x, y = preprocessing(self.img_paths[i], RATIO, target_resize_h, target_resize_w)
                img_lr_list.append(x)
                img_hr_list.append(y)

            self.lr_images = torch.stack(img_lr_list)
            self.hr_images = torch.stack(img_hr_list)

            np.save(x_path, self.lr_images.numpy())
            np.save(y_path, self.hr_images.numpy())
            print(f"Saved preprocessed {self.data_type} data to {x_path} and {y_path}")

    def __len__(self):
        return len(self.lr_images)

    def __getitem__(self, idx):
        return self.lr_images[idx], self.hr_images[idx]

def get_dataloaders():
    img_paths = []
    for dirname, _, filenames in os.walk(TRAIN_HR_DIR):
        for filename in filenames:
            img_paths.append(os.path.join(dirname, filename))

    for dirname, _, filenames in os.walk(VALID_HR_DIR):
        for filename in filenames:
            img_paths.append(os.path.join(dirname, filename))

    print('Dataset dimension: ', len(img_paths))

    num_total = len(img_paths)
    train_split_idx = floor(num_total * TRAIN_VAL_SPLIT_PERC)
    val_split_idx = floor(num_total * TRAIN_VAL_SPLIT_PERC + (num_total * (1 - TRAIN_VAL_SPLIT_PERC)) * VAL_TEST_SPLIT_PERC)

    train_img_paths = img_paths[:train_split_idx]
    val_img_paths = img_paths[train_split_idx:val_split_idx]
    test_img_paths = img_paths[val_split_idx:]

    print('Training: ', len(train_img_paths))
    print('Validation: ', len(val_img_paths))
    print('Test: ', len(test_img_paths))

    train_dataset = DIV2KDataset(train_img_paths, data_type="train")
    val_dataset = DIV2KDataset(val_img_paths, data_type="val")
    test_dataset = DIV2KDataset(test_img_paths, data_type="test")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=os.cpu_count() // 2, # Use half of CPU cores for data loading
        pin_memory=True # Speeds up data transfer to GPU
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=os.cpu_count() // 2,
        pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=os.cpu_count() // 2,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader