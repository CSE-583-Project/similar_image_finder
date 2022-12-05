import unittest
import random
from data_loader import Dataset
from data_loader import LoadData
from train import create_datasets
from train import create_dataloaders

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
        df = pd.read_csv(csv_file_path)
        train_ds, val_ds, test_ds = create_datasets(df= df,
                                                    json_path= json_path,
                                                    data_dir= data_dir,
                                                    train_transforms= train_transforms,
                                                    transforms= image_transforms)
        self.assertEqual(len(train_ds)+ len(val_ds)+ len(test_ds), len(df))

    def test_image_size_ds(self):
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
        df = pd.read_csv(csv_file_path)
        train_ds, val_ds, test_ds = create_datasets(df= df,
                                                    json_path= json_path,
                                                    data_dir= data_dir,
                                                    train_transforms= train_transforms,
                                                    transforms= image_transforms)
        sample_number = random.randint(0, len(train_ds))
        image, label = train_ds[sample_number]
        self.assertEqual(image.shape, (3, 224, 224))
        sample_number = random.randint(0, len(val_ds))
        image, label = val_ds[sample_number]
        self.assertEqual(image.shape, (3, 224, 224))
        sample_number = random.randint(0, len(test_ds))
        image, label = test_ds[sample_number]
        self.assertEqual(image.shape, (3, 224, 224))

    def test_num_classes_ds(self):
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
        df = pd.read_csv(csv_file_path)
        train_ds, val_ds, test_ds = create_datasets(df= df,
                                                    json_path= json_path,
                                                    data_dir= data_dir,
                                                    train_transforms= train_transforms,
                                                    transforms= image_transforms)
        
        unique_classes = train_ds._get_unique_classes() +\
                         val_ds._get_unique_classes() +\
                         test_ds._get_unique_classes()
        unique_classes = set(unique_classes)
        self.assertEqual(len(unique_classes), 31)
    
    def test_dataloader_batchsize_dl(self):
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
        
        df = pd.read_csv(csv_file_path)
        f = open(json_path)
        json_data = json.load(f)
        f.close()
        train_ds, val_ds, test_ds = create_datasets(df= df,
                                                    json_path= json_path,
                                                    data_dir= data_dir,
                                                    train_transforms= train_transforms,
                                                    transforms= image_transforms)

        train_loader, val_loader, test_loader = create_dataloaders(train_dataset= train_ds,
                                                               val_dataset= val_ds,
                                                               test_dataset= test_ds,
                                                               json_path= json_path)

        train_images, train_labels = next(iter(train_loader))
        self.assertEqual(json_data["train_batch_size"], train_images.size()[0])
        val_images, val_labels = next(iter(val_loader))
        self.assertEqual(json_data["val_batch_size"], val_images.size()[0])
        test_images, test_labels = next(iter(test_loader))
        self.assertEqual(json_data["test_batch_size"], test_images.size()[0])