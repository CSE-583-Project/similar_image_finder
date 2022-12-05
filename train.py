from data_loader.loader import Dataset, LoadData
from model.resnet_model import ResNetModel
from torchvision import transforms

def train_model(num_classes, train_loader, val_loader, test_loader, \
                model_destination_path, backbone_destination_path):
    print("Training Model...")
    model = ResNetModel(num_classes)
    model.train_model(train_config_path, train_loader, val_loader, test_loader)
    model.store_model(model_destination_path)
    model.generate_model_backbone()
    model.store_model_backbone(backbone_destination_path)

def create_dataset(csv_file_path, data_dir, transforms = None):
    print("Creating Dataset...")
    dataset = Dataset(csv_file_path, data_dir, transforms)
    return dataset

def create_dataloaders(dataset, json_path):
    print("Creating Data Loaders...")
    load_data = LoadData(dataset, json_path)
    train_loader, val_loader, test_loader = load_data.get_data_loaders()
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # Add data paths
    train_config_path = "train_config.json"
    csv_file_path = "data/fashion.csv"
    data_dir = "data"

    # Destination Path for Model and Backbone
    model_destination_path = "./model/model.pt"
    backbone_destination_path = "./model/backbone.pt"

    # Transforms for image transformation
    image_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5,), (0.5,))])

    # Create dataset
    dataset = create_dataset(csv_file_path, data_dir, image_transforms)
    
    # Get number of classes
    num_classes = dataset._get_num_classes()
    train_loader, val_loader, test_loader = create_dataloaders(dataset, train_config_path)
    train_model(num_classes, train_loader, val_loader, test_loader, \
                model_destination_path, backbone_destination_path)