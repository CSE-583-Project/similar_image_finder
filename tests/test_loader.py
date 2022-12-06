"""Test module to test dataset and dataloader.
"""
import unittest
import json
import random
import pandas as pd
import torchvision
from torchvision import transforms
from train import create_datasets
from train import create_dataloaders

class TestLoader(unittest.TestCase):
    """Class to test Data Loader and Dataset modules.
    """
    def test_dataset_length_ds(self):
        """Test function to test the length of the dataset created.
        """
        csv_file_path = r"data/fashion.csv"
        data_dir = "data"
        json_path = "train_config.json"

        image_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,))
    ])
        train_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(0.5),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5,), (0.5,))
    ])
        data_f = pd.read_csv(csv_file_path)
        train_ds, val_ds, test_ds = create_datasets(data_frame= data_f,
                                                    json_path= json_path,
                                                    data_dir= data_dir,
                                                    train_transform= train_transforms,
                                                    transform= image_transforms)
        self.assertEqual(len(train_ds)+ len(val_ds)+ len(test_ds), len(data_f))

    def test_image_size_ds(self):
        """Test function to check the size of image in the dataset.
        """
        csv_file_path = r"data/fashion.csv"
        data_dir = "data"
        json_path = "train_config.json"

        image_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,))
    ])
        train_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(0.5),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5,), (0.5,))
    ])
        data_f = pd.read_csv(csv_file_path)
        train_ds, val_ds, test_ds = create_datasets(data_frame= data_f,
                                                    json_path= json_path,
                                                    data_dir= data_dir,
                                                    train_transform= train_transforms,
                                                    transform= image_transforms)
        sample_number = random.randint(0, len(train_ds))
        image, _, _ = train_ds[sample_number]
        self.assertEqual(image.shape, (3, 224, 224))
        sample_number = random.randint(0, len(val_ds))
        image, _, _ = val_ds[sample_number]
        self.assertEqual(image.shape, (3, 224, 224))
        sample_number = random.randint(0, len(test_ds))
        image, _, _ = test_ds[sample_number]
        self.assertEqual(image.shape, (3, 224, 224))

    def test_num_classes_ds(self):
        """Test function to check the number of classes.
        """
        csv_file_path = "data/fashion.csv"
        data_dir = "data"
        json_path = "train_config.json"

        image_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,))
    ])
        train_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(0.5),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5,), (0.5,))
    ])
        data_f = pd.read_csv(csv_file_path)
        train_ds, val_ds, test_ds = create_datasets(data_frame= data_f,
                                                    json_path= json_path,
                                                    data_dir= data_dir,
                                                    train_transform= train_transforms,
                                                    transform= image_transforms)

        unique_classes = train_ds.get_unique_classes() +\
                         val_ds.get_unique_classes() +\
                         test_ds.get_unique_classes()
        unique_classes = set(unique_classes)
        self.assertEqual(len(unique_classes), 31)

    def test_dataloader_batchsize_dl(self):
        """Test function to check batch size of the data loader.
        """
        csv_file_path = "data/fashion.csv"
        data_dir = "data"
        json_path = "train_config.json"

        image_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,))
    ])
        train_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(0.5),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5,), (0.5,))
    ])

        data_f = pd.read_csv(csv_file_path)
        with open(json_path, encoding= "utf-8") as file:
            json_data = json.load(file)

        train_ds, val_ds, test_ds = create_datasets(data_frame= data_f,
                                                    json_path= json_path,
                                                    data_dir= data_dir,
                                                    train_transform= train_transforms,
                                                    transform= image_transforms)

        train_loader, val_loader, test_loader = create_dataloaders(train_dataset= train_ds,
                                                               val_dataset= val_ds,
                                                               test_dataset= test_ds,
                                                               json_path= json_path)

        train_images, _, _ = next(iter(train_loader))
        self.assertEqual(json_data["train_batch_size"], train_images.size()[0])
        val_images, _, _ = next(iter(val_loader))
        self.assertEqual(json_data["val_batch_size"], val_images.size()[0])
        test_images, _, _ = next(iter(test_loader))
        self.assertEqual(json_data["test_batch_size"], test_images.size()[0])
