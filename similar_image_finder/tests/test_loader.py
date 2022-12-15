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
from inference import embeddings_loader, cosine_calc


class TestLoader(unittest.TestCase):
    """Class to test Data Loader and Dataset modules.
    """
    def test_dataset_length_ds(self):
        """Test function to test the length of the dataset created.
        """
        csv_file_path = "tests/test_data/fashion_test.csv"
        data_dir = "tests/test_data"
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
        csv_file_path = "tests/test_data/fashion_test.csv"
        data_dir = "tests/test_data"
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
        csv_file_path = "tests/test_data/fashion_test.csv"
        data_dir = "tests/test_data"
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
        csv_file_path = "tests/test_data/fashion_test.csv"
        data_dir = "tests/test_data"
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

    def test_img_cos_finder(self):
        """Test function to test if cosine distance calculation results 
        in correct calculation of scores.
        """
        img_embedding = [0.5, 0.6, 0.5, 0.7]
        all_embeddings = ['[0.7, 0.1, 0.9, 0.1]',
                        '[0.4, 0.6, 0.2, 0.7]',
                        '[0.5, 3.6, 0.1, 1.7]',
                        '[0.5, 2.6, 0.5, 0.7]',
                        '[0.5, 0.6, 0.1, 0.9]',
                        '[0.3, 0.6, 0.5, 1.7]',
                        '[0.3, 0.4, 0.5, 0.6]',
                        '[0.3, 0.4, 1.5, 0.5]',
                        '[0.5, 0.1, 0.6, 0.4]',
                        '[0.5, 0.4, 0.6, 0.9]',
                        '[0.1, 0.6, 2.7, 1.2]',
                        '[1.1, 0.6, 0.8, 0.4]',
                        '[0.5, 1.2, 0.8, 0.7]']
        file_paths = ['1.jpg', '2.jpg', '3.jpg', '4.jpg', '5.jpg', '6.jpg', '7.jpg', '8.jpg',
                        '9.jpg', '10.jpg', '11.jpg', '12.jpg', '13.jpg']

        # Testing with correct ordering of file names w.r.t. cosine scores.
        self.assertEqual(cosine_calc(img_embedding, all_embeddings, file_paths), \
            ['7.jpg', '10.jpg', '2.jpg', '13.jpg', '5.jpg', '12.jpg', '6.jpg', '9.jpg', \
                '4.jpg', '3.jpg'])

    def test_all_emb_loader(self):
        """Test if csv file containing all embeddings is loaded
        with correct values and correct data types.
        """
        all_emb_path = 'tests/test_data/small_all_embeddings.csv'
        embeddings, file_paths = embeddings_loader(all_emb_path)
        print(embeddings)
        print(file_paths)
        print(type(embeddings))
        print(type(file_paths))

        self.assertEqual(embeddings, ['[0.5, 0.6, 0.5, 0.7]', '[0.4, 0.6, 0.2, 0.7]', \
            '[0.5, 3.6, 0.1, 1.7]', '[0.5, 2.6, 0.5, 0.7]', '[0.5, 0.6, 0.1, 0.9]', \
                '[0.3, 0.6, 0.5, 1.7]'])
        self.assertEqual(file_paths, ['shoes1.jpg', 'shoes2.jpg', \
            'shirt1.jpg', 'shirt2.jpg', 'trouser1.jpg', 'trouser2.jpg'])
        self.assertEqual(type(embeddings), list)
        self.assertEqual(type(file_paths), list)
        self.assertEqual(type(embeddings[0]), str)
        self.assertEqual(type(file_paths[0]), str)
