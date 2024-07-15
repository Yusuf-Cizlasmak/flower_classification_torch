import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image

from model import CustomDataset
import os

train_path = "model_dataset/train"
test_path = "model_dataset/test"
val_path = "model_dataset/test"
os.makedirs("model", exist_ok=True)


num_classes = 5
batch_size = 8
num_epochs = 50



transform = transforms.Compose([
    transforms.Resize((256, 256)),  
    transforms.ToTensor()])

train_dataset = CustomDataset(train_path, transform=transform)
test_dataset = CustomDataset(test_path, transform=transform)
val_dataset = CustomDataset(val_path, transform=transform)


train_dataset = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_dataset = DataLoader(test_dataset, batch_size=8, shuffle=True)
val_dataset = DataLoader(val_dataset, batch_size=8, shuffle=True)


transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Görüntüyü 256x256 boyutuna yeniden boyutlandırma
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Görüntüyü rastgele kırp ve yeniden boyutlandır
    transforms.RandomGrayscale(p=0.2),  # Görüntüyü %20 olasılıkla gri tonlamalıya çevir
    transforms.RandomHorizontalFlip(p=0.5),  # Görüntüyü yatayda %50 olasılıkla çevir
    transforms.RandomVerticalFlip(p=0.5),  # Görüntüyü dikeyde %50 olasılıkla çevir
    transforms.RandomRotation(30),  # Görüntüyü rastgele 30 derece döndür
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Görüntüyü rastgele %10 oranında kaydır
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),  # Görüntüye rastgele bulanıklık uygula
    transforms.ToTensor(),  # Görüntüyü tensor'a çevir
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Görüntüyü normalize et
])


# Transforms example
img_path = 'model_dataset/train/Lilly/1.jpg'
image = Image.open(img_path)
transformed_image = transform(image)

fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Orijinal görüntüyü göster
ax[0].imshow(image)
ax[0].set_title('Orijinal Görüntü')
ax[0].axis('off')

# Transform edilmiş görüntüyü göster
transformed_image_pil = transforms.ToPILImage()(transformed_image)
ax[1].imshow(transformed_image_pil)
ax[1].set_title('Transform Edilmiş Görüntü')
ax[1].axis('off')

plt.show()