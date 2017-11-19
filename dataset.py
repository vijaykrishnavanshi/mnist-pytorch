import pandas as pd
import numpy as np
import cv2
import os
import csv
from torch.utils.data import Dataset

class MNISTDataset(Dataset):
    """MNIST dataset."""

    def __init__(self, csv_file, root_dir):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
        """
        self.image_frame = pd.read_csv(csv_file, skiprows=1)
        self.root_dir = root_dir

    def __len__(self):
        return len(self.image_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_frame.ix[idx, 0])
        image = cv2.imread(img_name, 0)
        image = image.reshape((1,28,28))
        #image = np.transpose(image, (2, 0, 1))
        label = self.image_frame.ix[idx, 1]
        sample = {'image': image, 'label': label}
        return sample
    