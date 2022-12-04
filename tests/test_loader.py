import unittest

from dataLoader import Dataset
from dataLoader import LoadData

from torchvision import transforms
import torchvision
import pandas as pd

class TestLoader(unittest.TestCase):
    """Class to test Data Loader and Dataset modules.
    """
    def test_dataset_length(self):
        csv_file_path = r"data\fashion.csv"
        data_dir = "data"
        image_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,))
    ])
        dataset = Dataset(csv_file_path, data_dir, image_transforms)
        df = pd.read_csv(csv_file_path)
        self.assertEqual(len(dataset), len(df))