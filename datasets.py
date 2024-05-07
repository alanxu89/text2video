import csv
import os
import random
import time

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets.folder import IMG_EXTENSIONS, pil_loader

from video_transforms import (
    ToTensorVideo,
    RandomHorizontalFlipVideo,
    UCFCenterCropVideo,
)


def get_transforms_video(resolution=256):
    video_trans = transforms.Compose([
        ToTensorVideo(),
        # RandomHorizontalFlipVideo(),
        UCFCenterCropVideo(resolution),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5],
                             inplace=True),
    ])

    return video_trans


class DatasetFromCSV(torch.utils.data.Dataset):

    def __init__(self,
                 csv_path,
                 num_frames=16,
                 frame_interval=1,
                 transform=None,
                 root=None):
        # import pandas
        self.samples = []
        self.csv_path = csv_path
        self.root = root
        if not os.path.exists(csv_path) and root is not None:
            self.csv_path = os.path.join(self.root, csv_path)

        with open(self.csv_path) as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                self.samples.append(row)
            print(len(self.samples))

        self.transform = transform

        self.num_frames = num_frames
        self.frame_interval = frame_interval
        self.num_real_frames = 1 + (num_frames - 1) * frame_interval

    def getitem(self, index):
        t0 = time.time()
        video_id, url, duration, page_dir, text = self.samples[index]
        if self.root:
            path = os.path.join(self.root, page_dir, f"{video_id}.mp4")

        vframes, aframes, info = torchvision.io.read_video(
            filename=path, pts_unit="sec", output_format="TCHW")

        # Sampling video frames
        total_frames = len(vframes)
        start_frame_ind = random.randint(0,
                                         total_frames - self.num_real_frames)
        end_frame_ind = start_frame_ind + self.num_real_frames
        frame_indice = np.arange(start_frame_ind,
                                 end_frame_ind,
                                 step=self.frame_interval,
                                 dtype=int)
        video = vframes[frame_indice]

        video = self.transform(video)  # T C H W

        video = video.permute(1, 0, 2, 3)  # C T H W, channel first convention
        # print(f"{t0:.3f}, {time.time() - t0:.3f}")

        return {"video": video, "text": text, "video_id": video_id}

    def __getitem__(self, index):
        for _ in range(5):
            try:
                return self.getitem(index)
            except Exception as e:
                print(e)
                index = np.random.randint(len(self))
        raise RuntimeError("Too many bad data.")

    def __len__(self):
        return len(self.samples)


class PreprocessedDatasetFromCSV(torch.utils.data.Dataset):

    def __init__(self, csv_path, root=None, preprocessed_dir=None):
        # import pandas
        self.samples = []
        self.csv_path = csv_path
        self.root = root
        if not os.path.exists(csv_path) and root is not None:
            self.csv_path = os.path.join(self.root, csv_path)

        self.preprocessed_dir = preprocessed_dir
        if not os.path.exists(preprocessed_dir) and root is not None:
            self.preprocessed_dir = os.path.join(self.root, preprocessed_dir)

        with open(self.csv_path) as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                self.samples.append(row)
            print(len(self.samples))

    def getitem(self, index):
        t0 = time.time()
        video_id, url, duration, page_dir, text = self.samples[index]

        preprocessed_data_path = os.path.join(self.preprocessed_dir,
                                              f"{video_id}.pt")
        data = torch.load(preprocessed_data_path)
        # x = data['x']
        # y = data['y']
        # mask = data['mask']

        return data

    def __getitem__(self, index):
        for _ in range(5):
            try:
                return self.getitem(index)
            except Exception as e:
                print(e)
                index = np.random.randint(len(self))
        raise RuntimeError("Too many bad data.")

    def __len__(self):
        return len(self.samples)
