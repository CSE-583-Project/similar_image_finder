import unittest
import random
from dataLoader import Dataset
from dataLoader import LoadData

from torchvision import transforms
import torchvision
import pandas as pd

class TestLoader(unittest.TestCase):
    """Class to test Data Loader and Dataset modules.
    """
    def test_dataset_length_ds(self):
        csv_file_path = r"data/fashion.csv"
        data_dir = "data"
        image_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,))
    ])
        dataset = Dataset(csv_file_path, data_dir, image_transforms)
        df = pd.read_csv(csv_file_path)
        self.assertEqual(len(dataset), len(df))

    def test_image_size_ds(self):
        csv_file_path = r"data/fashion.csv"
        data_dir = "data"
        image_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,))
    ])
        dataset = Dataset(csv_file_path, data_dir, image_transforms)
        sample_number = random.randint(0, len(dataset))
        image, label = dataset[sample_number]
        self.assertEqual(image.shape, (3, 224, 224))

    def test_num_classes_ds(self):
        csv_file_path = r"data/fashion.csv"
        data_dir = "data"
        image_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,))
    ])
        dataset = Dataset(csv_file_path, data_dir, image_transforms)
        self.assertEqual(dataset._get_num_classes(), 31)