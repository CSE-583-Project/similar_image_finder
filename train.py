from data_loader.loader import Dataset, LoadData
from model.resnet_model import ResNetModel
from torchvision import transforms
from sklearn.model_selection import train_test_split
import pandas as pd
import json

def train_model(num_classes, train_loader, val_loader, test_loader, \
                model_destination_path, backbone_destination_path):
    print("Training Model...")
    model = ResNetModel(num_classes)
    model.train_model(train_config_path, train_loader, val_loader, test_loader)
    model.store_model(model_destination_path)
    model.generate_model_backbone()
    model.store_model_backbone(backbone_destination_path)

def create_datasets(df, json_path, data_dir, train_transforms = None, transforms = None):
    print("Creating Dataset...")
    f = open(json_path)
    json_data = json.load(f)
    f.close()
    train_df, inter_df = train_test_split(df, train_size= json_data["train_split"])
    val_df, test_df = train_test_split(inter_df, test_size=json_data["test_split"])
    train_dataset = Dataset(train_df, data_dir, train_transforms)
    val_dataset = Dataset(val_df, data_dir, transforms)
    test_dataset = Dataset(test_df, data_dir, transforms)

    return train_dataset, val_dataset, test_dataset

def create_dataloaders(train_dataset, val_dataset, test_dataset, json_path):
    print("Creating Data Loaders...")
    load_data_train = LoadData(train_dataset, json_path, "train", False)
    train_loader = load_data_train.get_data_loader()
    load_data_val = LoadData(val_dataset, json_path, "val", False)
    val_loader = load_data_val.get_data_loader()
    load_data_test = LoadData(test_dataset, json_path, "test", False)
    test_loader = load_data_test.get_data_loader()

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # Add data paths
    train_config_path = "train_config.json"
    csv_file_path = "data/fashion.csv"
    data_dir = "data"

    #read necessary files
    df = pd.read_csv(csv_file_path)

    # Destination Path for Model and Backbone
    model_destination_path = "./model/model.pt"
    backbone_destination_path = "./model/backbone.pt"

    #variables for model
    num_classes = len(df["ProductType"].unique())

    #transforms for image transformation
    image_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5,), (0.5,))])
    train_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(0.5),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5,), (0.5,))])

    #create dataset
    
    #dataset = create_dataset(csv_file_path, data_dir, image_transforms)
    train_dataset, val_dataset, test_dataset = create_datasets(df= df,
                                                               json_path= train_config_path,
                                                               data_dir= data_dir,
                                                               train_transforms= train_transforms,
                                                               transforms= image_transforms)
    
    #get loaders
    train_loader, val_loader, test_loader = create_dataloaders(train_dataset= train_dataset,
                                                               val_dataset= val_dataset,
                                                               test_dataset= test_dataset,
                                                               json_path= train_config_path)

    train_model(num_classes, train_loader, val_loader, test_loader, \
                model_destination_path, backbone_destination_path)