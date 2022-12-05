from data_loader.loader import Dataset, LoadData
from model.resnet_model import ResNetModel

def train_model(train_config_path):
    
    num_classes = None # TODO
    train_loader, val_loader, test_loader = None, None, None # TODO
    model = ResNetModel(num_classes)
    model.train_model(train_config_path, train_loader, val_loader, test_loader)
    model.store_model()
    model.generate_model_backbone()
    model.store_model_backbone()

if __name__ == "__main__":

    train_config_path = "train_config.json"
    train_model(train_config_path)