import torch
from torchvision import models
import json
import tqdm

class ResNetModel():

    def __init__(self, num_classes):

        self.resnet_model = models.resnet18(pretrained=True)
        num_features = self.resnet_model.fc.in_features
        self.resnet_model.fc = torch.nn.Linear(num_features, num_classes) 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.resnet_model.to(self.device)

    def train_model(self, train_config_path, train_loader, val_loader, test_loader):
        
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
            print('\nEpoch {}:\n'.format(epoch + 1))
            self.resnet_model.train()
            with tqdm.tqdm(total=len(iter(train_loader))) as pbar:
                train_loss, correct = 0, 0
                for input, target in train_loader:
                    input, target = input.to(self.device), target.to(self.device)
                    # Calculate training loss on model
                    optimizer.zero_grad()
                    output = self.resnet_model(input)
                    loss = torch.nn.functional.cross_entropy(output, target)
                    train_loss += loss.item()
                    loss.backward()
                    optimizer.step()
                    correct += (torch.argmax(output, dim = 1) == target).type(torch.FloatTensor).sum().item()
                    pbar.update(1)
                pbar.close()

                train_loss /= len(train_loader)
                train_losses.append(train_loss)

                train_accuracy = 100. * correct / len(train_loader.dataset)
                train_accuracies.append(train_accuracy)

            self.resnet_model.eval() 
            with torch.no_grad():

                val_loss, correct = 0, 0
                for input, target in val_loader:
                    input, target = input.to(self.device), target.to(self.device)
                    # Calculate validation loss on model
                    optimizer.zero_grad()
                    output = self.resnet_model(input)
                    val_loss += torch.nn.functional.cross_entropy(output, target).item()
                    correct += (torch.argmax(output, dim = 1) == target).type(torch.FloatTensor).sum().item()
      
            val_loss /= len(val_loader)
            val_losses.append(val_loss)

            val_accuracy = 100. * correct / len(val_loader.dataset)
            val_accuracies.append(val_accuracy)

            print('Train Loss: {:4f}, Train Accuracy: {:.2f}%'.format(
            train_loss, train_accuracy))
            print('Val Loss: {:4f}, Val Accuracy: {:.2f}%\n'.format(
            val_loss, val_accuracy))
            print("######################################################")


        #Calculate loss on test set
        test_loss, correct = 0, 0
        self.resnet_model.eval()
        with torch.no_grad():
            for input, target in test_loader:
                input, target = input.to(self.device), target.to(self.device)
                #Calculate testing loss on model
                optimizer.zero_grad()
                output = self.resnet_model(input)
        test_loss += torch.nn.functional.cross_entropy(output, target).item()
        correct += (torch.argmax(output, dim = 1) == target).type(torch.FloatTensor).sum().item()

        test_loss /= len(test_loader)
        print('\nTest Loss: {:.4f}, Test Accuracy: {:.2f}%\n'.format(
                test_loss, 100. * correct / len(test_loader.dataset)))
    
    def store_model(self, destination_path):
        
        torch.save(self.resnet_model, destination_path)

    def generate_model_backbone(self):

        self.resnet_backbone = torch.nn.Sequential(*(list(self.resnet_model.children())[:-1]))
        return self.resnet_backbone

    def store_model_backbone(self, destination_path):
        
        torch.save(self.resnet_backbone, destination_path)
