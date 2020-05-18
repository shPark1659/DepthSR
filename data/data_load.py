import os
import torch
import cv2
import numpy as np
from PIL import Image as Image
import tifffile
from data import PairCompose, PairRandomCrop, PairRandomHorizontalFilp, PairToTensor
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader


def train_dataloader(path, batch_size=64, crop_size=32, scale_factor=4, num_workers=0, use_transform=True):
    image_dir = os.path.join(path, 'train_x4_noc')
    # TODO: Add transform
    #transform = None
    #if use_transform:
    #    transform = PairCompose(
    #        [
    #            PairRandomCrop(256),
    #            PairRandomHorizontalFilp(),
    #            PairToTensor()
    #        ]
    #    )
    dataloader = DataLoader(
        DepthDataset(image_dir, crop_size, scale_factor, 2),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True
    )
    return dataloader


def test_dataloader(path, batch_size=1, scale_factor=4, num_workers=0):
    image_dir = os.path.join(path, 'test_x4')
    dataloader = DataLoader(
        DepthDataset(image_dir, None, scale_factor, 3),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    return dataloader


def valid_dataloader(path, batch_size=1, scale_factor=4, num_workers=0):
    image_dir = os.path.join(path, 'test_x4')
    dataloader = DataLoader(
        DepthDataset(image_dir, None, scale_factor, 3),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    return dataloader


class DepthDataset(Dataset):
    def __init__(self, image_dir, crop_size, scale_factor, mode):
        self.image_dir = image_dir
        self.image_list = os.listdir(os.path.join(image_dir, 'Df/'))
        self._check_image(self.image_list)
        self.image_list.sort()
        self.crop_size = crop_size
        self.scale_factor = scale_factor
        self.mode = mode #0: train 1: validation

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        if self.mode == 0:
            #train data
            img = cv2.imread(os.path.join(self.image_dir, 'RGB', self.image_list[idx]), cv2.IMREAD_COLOR)
            ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            y = ycbcr[:, :, 0] / 255.0
            label = tifffile.imread(os.path.join(self.image_dir, 'DF', self.image_list[idx]))
            label = label.astype(np.float32)

            # normalize depth image 0 to 1
            label_min = label.min()
            label_max = label.max()
            label = (label - label_min) / (label_max - label_min)

            f = np.random.randint(2)  # flip

            def img2patch(img, crop_size, flip, left=None, top=None):
                h, w = img.shape

                if left is None and top is None:
                    left = np.random.randint(0, w - crop_size)
                    top = np.random.randint(0, h - crop_size)

                patch = img[top:top + crop_size, left:left + crop_size]

                # flip image
                if flip == 0:
                    patch = patch
                else:
                    patch = cv2.flip(patch, 1)

                return patch, left, top

            colorHR, left, top = img2patch(y, self.crop_size, f)
            depthHR, _, _ = img2patch(label, self.crop_size, f, left, top)
            depthLR = cv2.resize(depthHR, dsize=(0, 0), fx=1/self.scale_factor, fy=1/self.scale_factor,
                                 interpolation=cv2.INTER_CUBIC)
            depthLRup = cv2.resize(depthLR, dsize=(0, 0), fx=self.scale_factor, fy=self.scale_factor,
                                   interpolation=cv2.INTER_CUBIC)

            sobelx = cv2.Sobel(depthLR, cv2.CV_32F, 1, 0, ksize=3)
            sobely = cv2.Sobel(depthLR, cv2.CV_32F, 0, 1, ksize=3)
            sobel = cv2.addWeighted(sobelx, 1, sobely, 1, 0)

            #sobel_min = sobel.min()
            #sobel_max = sobel.max()
            #sobel = (sobel - sobel_min) / (sobel_max - sobel_min)

            sobelup = cv2.resize(sobel, dsize=(0, 0), fx=self.scale_factor, fy=self.scale_factor,
                                 interpolation=cv2.INTER_CUBIC)


            colorHR = torch.from_numpy(colorHR).float().unsqueeze(0)
            depthHR = torch.from_numpy(depthHR).float().unsqueeze(0)
            depthLRup = torch.from_numpy(depthLRup).float().unsqueeze(0)
            sobelup = torch.from_numpy(sobelup).float().unsqueeze(0)

            return colorHR, depthLRup, sobelup, depthHR
        elif self.mode == 1:
            # validation
            img = cv2.imread(os.path.join(self.image_dir, 'y', self.image_list[idx]), cv2.IMREAD_GRAYSCALE) / 255.0
            depthup = cv2.imread(os.path.join(self.image_dir, 'Df', self.image_list[idx]), cv2.IMREAD_GRAYSCALE) / 255.0
            label = cv2.imread(os.path.join(self.image_dir, 'label', self.image_list[idx]),
                               cv2.IMREAD_GRAYSCALE) / 255.0



            colorHR = torch.from_numpy(img).float().unsqueeze(0)
            depthLRup = torch.from_numpy(depthup).float().unsqueeze(0)
            depthHR = torch.from_numpy(label).float().unsqueeze(0)

            return colorHR, depthLRup, depthHR
        elif self.mode == 2:
            # 미리 crop된 학습 영상 이용
            img = cv2.imread(os.path.join(self.image_dir, 'y', self.image_list[idx]), cv2.IMREAD_GRAYSCALE) / 255.0
            depthup = cv2.imread(os.path.join(self.image_dir, 'Df', self.image_list[idx]), cv2.IMREAD_GRAYSCALE) / 255.0
            label = cv2.imread(os.path.join(self.image_dir, 'label', self.image_list[idx]), cv2.IMREAD_GRAYSCALE) / 255.0

            r = np.random.randint(4)  # rotate
            f = np.random.randint(2)  # flip

            # rotate image
            if r == 0:
                img = img
                depthup = depthup
                label = label
            elif r == 1:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                depthup = cv2.rotate(depthup, cv2.ROTATE_90_CLOCKWISE)
                label = cv2.rotate(label, cv2.ROTATE_90_CLOCKWISE)
            elif r == 2:
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                depthup = cv2.rotate(depthup, cv2.ROTATE_90_COUNTERCLOCKWISE)
                label = cv2.rotate(label, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif r == 3:
                img = cv2.rotate(img, cv2.ROTATE_180)
                depthup = cv2.rotate(depthup, cv2.ROTATE_180)
                label = cv2.rotate(label, cv2.ROTATE_180)

            # flip image
            if f == 0:
                img = img
                depthup = depthup
                label = label
            else:
                img = cv2.flip(img, 1)
                depthup = cv2.flip(depthup, 1)
                label = cv2.flip(label, 1)

            colorHR = torch.from_numpy(img).float().unsqueeze(0)
            depthLRup = torch.from_numpy(depthup).float().unsqueeze(0)
            depthHR = torch.from_numpy(label).float().unsqueeze(0)

            return colorHR, depthLRup, depthHR

        elif self.mode == 3:
            # evaluation
            img = cv2.imread(os.path.join(self.image_dir, 'y', self.image_list[idx]), cv2.IMREAD_GRAYSCALE) / 255.0
            depthup = cv2.imread(os.path.join(self.image_dir, 'Df', self.image_list[idx]), cv2.IMREAD_GRAYSCALE) / 255.0
            label = cv2.imread(os.path.join(self.image_dir, 'label', self.image_list[idx]),
                               cv2.IMREAD_GRAYSCALE) / 255.0

            image_name = self.image_list[idx]



            colorHR = torch.from_numpy(img).float().unsqueeze(0)
            depthLRup = torch.from_numpy(depthup).float().unsqueeze(0)
            depthHR = torch.from_numpy(label).float().unsqueeze(0)

            return colorHR, depthLRup, depthHR, image_name
    @staticmethod
    def _check_image(lst):
        for x in lst:
            _, ext = x.split('.')
            if ext not in ['png', 'jpg', 'jpeg', 'tif']:
                raise ValueError
