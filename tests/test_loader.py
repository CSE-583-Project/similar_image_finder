import unittest
import random
from data_loader import Dataset
from data_loader import LoadData

from torchvision import transforms
import torchvision
import pandas as pd
import json

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
    
    def test_dataloader_batchsize_dl(self):
        csv_file_path = r"data/fashion.csv"
        data_dir = "data"
        image_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,))
    ])
        dataset = Dataset(csv_file_path, data_dir, image_transforms)
        json_path = "train_config.json"
        file = open(json_path)
        json_data = json.load(file)
        file.close()
        load_data = LoadData(dataset, json_path)
        train_loader, val_loader, test_loader = load_data.get_data_loaders()
        train_images, train_labels = next(iter(train_loader))
        self.assertEqual(json_data["train_batch_size"], train_images.size()[0])
        val_images, val_labels = next(iter(val_loader))
        self.assertEqual(json_data["val_batch_size"], val_images.size()[0])
        test_images, test_labels = next(iter(test_loader))
        self.assertEqual(json_data["test_batch_size"], test_images.size()[0])