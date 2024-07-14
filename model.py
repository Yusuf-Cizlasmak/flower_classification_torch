from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

import timm 
import torch.nn as nn
from torchviz import make_dot


class CustomDataset(Dataset):
    """
    
    """
    def __init__(self, dataset_path, transform=None):
        self.dataset = ImageFolder(root=dataset_path, transform=transform)

    
    @property
    def classes(self):
        return self.dataset.classes
    
    # dunder methods

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]
    


class CustomModel(nn.Module):
    def __init__(self, num_classes):
        super(CustomModel, self).__init__()
        self.model = timm.create_model("resnet18", pretrained=True)
        
        self.features = nn.Sequential(*list(self.model.children())[:-1]) # son layerı çıkartıyoruz

        network_size=512

        self.classifier= nn.Sequential( nn.Flatten(),
            nn.Linear(network_size, num_classes)
        )


    def forward(self, x):
        x = self.features(x)
        output= self.classifier(x)
        return output
    
    def visualize(self, x, save_path):
        dot = make_dot(self.forward(x), params=dict(self.named_parameters()))
        dot.format = 'png'
        dot.render('model')

    