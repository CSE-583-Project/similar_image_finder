"""Module that defines, creates and loads the dataset and data loader
for the purpose of training and inferencing the machine learning
model.
"""
import os
import json
import torch
from PIL import Image
from torch.utils.data import DataLoader

#open csv from data folder
class_map = {"Tops": 0,
             "Capris": 1,
             "Dresses": 2,
             "Shorts": 3,
             "Tshirts": 4,
             "Skirts": 5,
             "Jeans": 6,
             "Leggings": 7,
             "Innerwear Vests": 8,
             "Rompers": 9,
             "Lehenga Choli": 10,
             "Salwar": 11,
             "Booties": 12,
             "Clothing Set": 13,
             "Trousers": 14,
             "Shirts": 15,
             "Jackets": 16,
             "Kurtas": 17,
             "Sweatshirts": 18,
             "Kurta Sets": 19,
             "Churidar": 20,
             "Waistcoat": 21,
             "Blazers": 22,
             "Casual Shoes": 23,
             "Flip Flops": 24,
             "Sandals": 25,
             "Formal Shoes": 26,
             "Sports Shoes": 27,
             "Sports Sandals": 28,
             "Flats": 29,
             "Heels": 30}

class Dataset(torch.utils.data.Dataset):
    """Class Dataset to create the datset for machine learning and
    inferencing.
    """
    def __init__(self, data_frame, data_dir, transform=None, img=None):
        """Init function for dataset creation.
        Arguments:
        data_frame (pd.dataframe): dataframe with paths to images and labels.
        data_dir (str): directory where data is stored.
        transform (pytorch.transforms): Transformations to be applied to images.

        Returns None.
        """
        self.data_frame = data_frame
        self.transform = transform
        self.data_dir = data_dir
        self.img = img

    #dataset length
    def __len__(self):
        """Length of the dataset.
        Arguments:
        None.

        Returns length of the dataset
        """
        return len(self.data_frame)

    def get_unique_classes(self):
        """Gets the unique classes in the dataset.
        Arguments:
        None.

        Returns the unique classes of the dataset as a list.
        """
        return list(self.data_frame["ProductType"].unique())

    #load an one of images
    def __getitem__(self,idx):
        """Used to load an image in the dataset into Dataset module
        of PyTorch.
        Arguments:
        idx (int): Index of the data point.

        Returns:
        img_transformed (pytorch.Tensor): Transformed image.
        label (int): Class of the image.
        img_path (str): Path where the image is stored.
        """
        img_path = os.path.join(self.data_dir,
                                self.data_frame.iloc[idx]["Category"],
                                self.data_frame.iloc[idx]["Gender"],
                                "Images",
                                "images_with_product_ids",
                                self.data_frame.iloc[idx]["Image"])
        self.img = Image.open(img_path).convert("RGB")
        img_transformed = self.transform(self.img)
        self.img.close()

        class_name = self.data_frame.iloc[idx]["ProductType"]
        label = class_map[class_name]

        return img_transformed, label, img_path

class LoadData:
    """Class to create dataloaders for machine learning and
    inferencing.
    """
    def __init__(self, dataset, json_path, type_loader, inference, shuffle=True) -> None:
        """Init function for the data loader.

        Arguments:
        dataset (pytorch.Dataset): Dataset object that needs to be loaded.
        json_path (str): Path of the json file with parameters for loading.
        type_loader (str): Indicates which type of loader(train, test, val).
        inference (bool): Indicates whether to load as inference or training.
        shuffle (bool): Indicates whether to shuffle the data points.

        Returns None.
        """
        self.dataset = dataset
        self.shuffle = shuffle
        if not inference:
            with open(json_path, encoding= "utf-8") as file:
                self.json_data = json.load(file)
        self.type_loader = type_loader

    def get_data_loader(self):
        """Function to get data loader.

        Arguments:
        None.

        Returns:
        loader (pytorch.DataLoader): Data loader object with images, labels
        and paths.
        """
        #batch sizes
        batch_size = self.json_data[self.type_loader+"_batch_size"]

        loader = DataLoader(dataset = self.dataset,
                            batch_size=batch_size,
                            shuffle=self.shuffle)
        return loader
