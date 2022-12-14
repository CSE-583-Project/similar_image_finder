""" Python script to train the model on data, store the model,
    create the model backbone and store it.
"""
import json
import tqdm
import torch
from torch import nn
from torchvision import models

class ResNetModel():

    """
    Class for training the model on data, model storing,
    backbone creation and backbone storing
    """

    def __init__(self, num_classes):

        """
        Inits ResNet Model Class with no. of classes.

        Arguments:
        num_classes (int): Number of classes in the dataset.

        Returns None.
        """

        self.resnet_model = models.resnet18(pretrained=True)
        num_features = self.resnet_model.fc.in_features
        self.resnet_model.fc = nn.Linear(num_features, num_classes)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.resnet_model.to(self.device)

    def train_model(self, train_config_path, train_loader, \
                    val_loader, test_loader):

        """
        Trains model on dataset

        Arguments:
        train_config_path (str): Path to the train config file
        train_loader (dataloader): Dataloader for the train set
        val_loader (dataloader): Dataloader for the val set
        test_loader (dataloader): Dataloader for the test set

        Returns Trained Model.
        """

        json_file = open(train_config_path)
        train_config = json.load(json_file)

        learning_rate = train_config["learning_rate"]

        if train_config["optimizer"] == "Adam":
            optimizer = torch.optim.Adam(self.resnet_model.parameters(), lr=learning_rate)
        elif train_config["optimizer"] == "SGD":
            optimizer = torch.optim.SGD(self.resnet_model.parameters(), lr=learning_rate)

        epochs = train_config["epochs"]

        train_losses, train_accuracies, val_losses, val_accuracies = [], [], [], []
        for epoch in range(epochs):
            print(f'\nEpoch {epoch + 1}:\n')
            self.resnet_model.train()
            with tqdm.tqdm(total=len(iter(train_loader))) as pbar:
                train_loss, correct = 0, 0
                for input, target, _ in train_loader:
                    input, target = input.to(self.device), target.to(self.device)
                    # Calculate training loss on model
                    optimizer.zero_grad()
                    output = self.resnet_model(input)
                    loss = nn.functional.cross_entropy(output, target)
                    train_loss += loss.item()
                    loss.backward()
                    optimizer.step()
                    correct += (torch.argmax(output, dim = 1) \
                                == target).type(torch.FloatTensor).sum().item()
                    pbar.update(1)
                pbar.close()

                train_loss /= len(train_loader)
                train_losses.append(train_loss)

                train_accuracy = 100. * correct / len(train_loader.dataset)
                train_accuracies.append(train_accuracy)

            self.resnet_model.eval()
            with torch.no_grad():

                val_loss, correct = 0, 0
                for input, target, _ in val_loader:
                    input, target = input.to(self.device), target.to(self.device)
                    # Calculate validation loss on model
                    optimizer.zero_grad()
                    output = self.resnet_model(input)
                    val_loss += nn.functional.cross_entropy(output, target).item()
                    correct += (torch.argmax(output, dim = 1) \
                                == target).type(torch.FloatTensor).sum().item()

            val_loss /= len(val_loader)
            val_losses.append(val_loss)

            val_accuracy = 100. * correct / len(val_loader.dataset)
            val_accuracies.append(val_accuracy)

            print(f'Train Loss: {train_loss:4f}, Train Accuracy: {train_accuracy:.2f}')
            print(f'Val Loss: {val_loss:4f}, Val Accuracy: {val_accuracy:.2f}\n')
            print("######################################################")


        #Calculate loss on test set
        test_loss, correct = 0, 0
        self.resnet_model.eval()
        with torch.no_grad():
            for input, target, _ in test_loader:
                input, target = input.to(self.device), target.to(self.device)
                #Calculate testing loss on model
                optimizer.zero_grad()
                output = self.resnet_model(input)
                test_loss += nn.functional.cross_entropy(output, target).item()
                correct += (torch.argmax(output, dim = 1) == target).type(torch.FloatTensor).sum().item()

        test_loss /= len(test_loader)
        print(f'\nTest Loss: {test_loss:.4f}, \
                Test Accuracy: {100. * correct / len(test_loader.dataset):.2f}\n')
        return self.resnet_model

    def store_model(self, destination_path):
        """
        Stores trained model

        Arguments:
        destination_path (str): Path for storing the trained model

        Returns None.
        """

        torch.save(self.resnet_model, destination_path)

    def generate_model_backbone(self):

        """
        Generates backbone for the trained model

        Returns None.
        """

        self.resnet_backbone = nn.Sequential(*(list(self.resnet_model.children())[:-1]))
        return self.resnet_backbone

    def store_model_backbone(self, destination_path):

        """
        Stores model Backbone

        Arguments:
        destination_path (str): Path for storing the model backbone

        Returns None.
        """

        torch.save(self.resnet_backbone, destination_path)


    def load_state_dict(self, destination_path, map_location):
        """
        Loading saved model.
        Arguments:
        destination_path - Path from which model is to be loaded.
        Returns:
        The RESNET model.
        """
        return torch.load(destination_path, map_location = map_location)