import pandas as pd
import os
import torch
from PIL import Image
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import json
from torchvision import transforms

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
    def __init__(self, csv_file, data_dir,transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.data_dir = data_dir
        
    #dataset length
    def __len__(self):
        return len(self.df)

    def _get_num_classes(self):
        return len(self.df["ProductType"].unique())
    
    #load an one of images
    def __getitem__(self,idx):
        #path = data, Category, Gender, Images, images_with_product_ids
        img_path = os.path.join(self.data_dir,
                                self.df.iloc[idx]["Category"],
                                self.df.iloc[idx]["Gender"],
                                "Images",
                                "images_with_product_ids",
                                self.df.iloc[idx]["Image"])
        img = Image.open(img_path).convert("RGB")
        img_transformed = self.transform(img)
        img.close()
        
        class_name = self.df.iloc[idx]["ProductType"]
        label = class_map[class_name]
            
        return img_transformed, label

class LoadData:
    def __init__(self, dataset, json_path) -> None:
        self.dataset = dataset
        file = open(json_path)
        self.json_data = json.load(file)
        file.close()
        

    def get_data_loaders(self):

        #dataset sizes
        self.train_size = self.json_data["train_split"]
        self.val_size = self.json_data["val_split"]
        self.test_size = self.json_data["test_split"]

        #batch sizes
        self.train_batch_size = self.json_data["train_batch_size"]
        self.val_batch_size = self.json_data["val_batch_size"]
        self.test_batch_size = self.json_data["test_batch_size"]

        train_length = int(len(self.dataset) * self.train_size)
        val_length = int(len(self.dataset) * self.val_size)
        test_length = len(self.dataset) - train_length - val_length

        self.lengths = [train_length, val_length, test_length]
        train_data, val_data, test_data = random_split(self.dataset,
                                                       lengths= self.lengths)

        train_loader = DataLoader(dataset = train_data,
                                  batch_size=self.train_batch_size,
                                  shuffle=True)
        test_loader = DataLoader(dataset = test_data,
                                 batch_size=self.test_batch_size,
                                 shuffle=True)
        val_loader = DataLoader(dataset = val_data,
                                batch_size=self.val_batch_size,
                                shuffle=True)
        return train_loader, val_loader, test_loader