import csv
import os

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets.folder import IMG_EXTENSIONS, pil_loader

from video_transforms import ToTensorVideo, RandomHorizontalFlipVideo, UCFCenterCropVideo


def get_transforms_video(resolution=256):
    video_trans = transforms.Compose([
        ToTensorVideo(),
        RandomHorizontalFlipVideo(),
        UCFCenterCropVideo(resolution),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[1.0, 1.0, 1.0],
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
        with open(csv_path) as f:
            spamreader = csv.reader(f)
            for row in spamreader:
                self.samples.append(row)
            print(len(self.samples))

        self.transform = transform

        self.num_frames = num_frames
        self.frame_interval = frame_interval
        # self.temporal_sample = video_transforms.TemporalRandomCrop(
        #     num_frames * frame_interval)
        self.root = root

    def getitem(self, index):
        sample = self.samples[index]
        path = sample[0]
        if self.root:
            path = os.path.join(self.root, path)
        text = sample[1]

        vframes, aframes, info = torchvision.io.read_video(
            filename=path, pts_unit="sec", output_format="TCHW")
        # total_frames = len(vframes)

        video = vframes
        video = self.transform(video)  # T C H W

        video = video.permute(1, 0, 2, 3)

        return {"video": video, "text": text}

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
