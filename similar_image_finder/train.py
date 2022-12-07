"""Python script to train machine learning model and save
to specified path.
"""
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from torchvision import transforms
from data_loader.loader import Dataset, LoadData
from model.resnet_model import ResNetModel

def train_model(num_classes, train_config_path, loaders_list, \
                model_destination_path, backbone_destination_path):
    """Function to train model.

    Arguments:
    num_classes (int): Number of classes in the dataset.
    train_loader (pytorch.DataLoader): Data loader for train data.
    val_loader (pytorch.DataLoader): Data loader for validation data.
    test_loader (pytorch.DataLoader): Data loader for test data.
    model_destination_path (str): Path where the model will be saved.
    backbone_destination_path (str): Path where the model backbone will
    be saved.

    Returns None.
    """
    print("Training Model...")
    model = ResNetModel(num_classes)
    train_loader, val_loader, test_loader = loaders_list
    model.train_model(train_config_path, train_loader, val_loader, test_loader)
    model.store_model(model_destination_path)
    model.generate_model_backbone()
    model.store_model_backbone(backbone_destination_path)

def create_datasets(data_frame, json_path, data_dir, train_transform = None, transform = None):
    """Function to create dataset.

    Arguments:
    data_frame (pd.dataframe): dataframe with paths to images and labels.
    json_path (str): Path of the json file with parameters for loading.
    data_dir (str): directory where data is stored.
    train_transforms (pytorch.transforms): Transformations to be applied to training
    images.
    transforms (pytorch.transforms): Transformations to be applied to val and test
    images.

    Returns:
    train_dataset (pytorch.Dataset): Dataset object with training images and labels.
    val_dataset (pytorch.Dataset): Dataset object with validation images and labels.
    test_dataset (pytorch.Dataset): Dataset object with test images and labels.
    """
    print("Creating Dataset...")
    with open(json_path, encoding= "utf-8") as file:
        json_data = json.load(file)
    train_df, inter_df = train_test_split(data_frame, train_size= json_data["train_split"])
    val_df, test_df = train_test_split(inter_df, test_size=json_data["test_split"])
    train_dataset = Dataset(train_df, data_dir, transform= train_transform)
    val_dataset = Dataset(val_df, data_dir, transform= transform)
    test_dataset = Dataset(test_df, data_dir, transform= transform)

    return train_dataset, val_dataset, test_dataset

def create_dataloaders(train_dataset, val_dataset, test_dataset, json_path):
    """Function to create dataloaders.

    Arguments:
    train_dataset (pytorch.Dataset): Dataset object with training images and labels.
    val_dataset (pytorch.Dataset): Dataset object with validation images and labels.
    test_dataset (pytorch.Dataset): Dataset object with test images and labels.
    json_path (str): Path to load configs for training and testing.

    Returns:
    train_loader (pytorch.DataLoader): DataLoader object with training images and
    labels.
    val_loader (pytorch.DataLoader): DataLoader object with validation images and
    labels.
    test_loader (pytorch.DataLoader): DataLoader object with test images and
    labels.
    """
    print("Creating Data Loaders...")
    load_data_train = LoadData(train_dataset, json_path, "train", False)
    train_loader = load_data_train.get_data_loader()
    load_data_val = LoadData(val_dataset, json_path, "val", False)
    val_loader = load_data_val.get_data_loader()
    load_data_test = LoadData(test_dataset, json_path, "test", False)
    test_loader = load_data_test.get_data_loader()
    loaders_list = train_loader, val_loader, test_loader

    return loaders_list

if __name__ == "__main__":
    # Add data paths
    TRAIN_CONFIG_PATH = "train_config.json"
    CSV_FILE_PATH = "data/fashion.csv"
    DATA_DIR = "data"

    #read necessary files
    df = pd.read_csv(CSV_FILE_PATH)

    # Destination Path for Model and Backbone
    MODEL_DEST_PATH = "./model/model.pt"
    BACKBONE_DEST_PATH = "./model/backbone.pt"

    #variables for model
    n_classes = len(df["ProductType"].unique())

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
    train_ds, val_ds, test_ds = create_datasets(data_frame= df,
                                                               json_path= TRAIN_CONFIG_PATH,
                                                               data_dir= DATA_DIR,
                                                               train_transform= train_transforms,
                                                               transform= image_transforms)

    #get loaders
    ldr_list = create_dataloaders(train_dataset= train_ds,
                                                               val_dataset= val_ds,
                                                               test_dataset= test_ds,
                                                               json_path= TRAIN_CONFIG_PATH)

    train_model(n_classes, TRAIN_CONFIG_PATH, ldr_list, \
                MODEL_DEST_PATH, BACKBONE_DEST_PATH)
