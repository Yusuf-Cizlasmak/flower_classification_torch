import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image

from torchviz import make_dot
from model import CustomModel ,CustomDataset
import os

train_path = "model_dataset/train"
test_path = "model_dataset/test"
val_path = "model_dataset/test"
os.makedirs("model", exist_ok=True)


num_classes = 5
batch_size = 8
num_epochs = 50




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

transform = transforms.Compose([
    transforms.Resize((256, 256)),  
    transforms.ToTensor()])

train_dataset = CustomDataset(train_path, transform=transform)
test_dataset = CustomDataset(test_path, transform=transform)
val_dataset = CustomDataset(val_path, transform=transform)


train_dataset = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_dataset = DataLoader(test_dataset, batch_size=8, shuffle=True)
val_dataset = DataLoader(val_dataset, batch_size=8, shuffle=True)

# Model

model = CustomModel(num_classes=num_classes)
model.to(device)



#Example image

example_images, example_labels = next(iter(train_dataset))
example_images = example_images.to(device)

# Visualize
dot = make_dot(model(example_images), params=dict(model.named_parameters()))
dot.format = 'png'
dot.render("model", directory="model", cleanup=True)


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





# Params
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Training
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in tqdm(train_dataset, desc="Training loop"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)

    train_loss = running_loss / len(train_dataset.dataset)
    train_losses.append(train_loss)

    model.eval()

    with torch.no_grad():
        for images, labels in tqdm(val_dataset, desc="Validation loop"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, 1)


    val_loss = running_loss / len(val_dataset.dataset)
    val_losses.append(val_loss)


    print(
        f"Epoch {epoch+1}/{num_epochs} - Train loss: {train_loss}, Validation loss: {val_loss}"
    )
        # Save Model
    os.makedirs("model", exist_ok=True)
    torch.save(obj=model.state_dict(), f=f"model/flower_{epoch}.pth")

    # Plot losses
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('losses.png')
    plt.clf()


# Plot confusion matrix
model.eval()
conf_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int32)
class_names = train_dataset.dataset.classes
with torch.no_grad():
    for images, labels in tqdm(val_dataset, desc="Validation loop"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        for i, label in enumerate(labels):
            conf_matrix[label, predicted[i]] += 1

fig, ax = plt.subplots()
im = ax.matshow(conf_matrix, cmap='Blues')

for i in range(num_classes):
    for j in range(num_classes):
        ax.text(j, i, str(conf_matrix[i, j].item()), ha='center', va='center', color='black')

tick_marks = range(num_classes)
plt.title('Confusion Matrix')
plt.colorbar(im)
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.savefig('confusion_matrix.png')
plt.clf()