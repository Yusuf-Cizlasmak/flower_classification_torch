import torch
import torchvision.transforms as transforms

import gradio as gr

from model import CustomModel

# hangi modeli kullanacağımızı belirtiyoruz.
model_path = "./model/flower_40.pth"

labels = ["Lilly", "Lotus", "Orchid", "Sunflower", "Tulip"]

num_classes = len(labels)  # sınıf sayısı

device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)  # Cihazı belirler.

# Transform
transform = transforms.Compose(
    [transforms.Resize((256, 256)), transforms.ToTensor()]
)  # Görüntü ön işleme işlemleri.

# Model
model = CustomModel(num_classes=5)  # Modeli oluşturur.
model.load_state_dict(torch.load(model_path))  # Modeli yükler.
model.to(device)  # Modeli cihaza yükler.
model.eval()  # Modeli değerlendirme moduna alır.


def predict(image):
    img_tensor = (
        transform(image).unsqueeze(0).to(device)
    )  # Görüntüyü yükler ve cihaza yükler.

    with torch.no_grad():
        predictions = model(img_tensor)  # Modeli kullanarak tahmin yapar.

    predictions = torch.nn.functional.softmax(predictions, dim=1)[
        0
    ]  

    confidences = {
        labels[i]: float(predictions[i]) for i in range(num_classes)
    }  # Sınıf isimleri ve olasılıklarını bir sözlükte toplar.

    return confidences  


# ŞİMDİ GRADIO INTERFACE OLUŞTURUYORUZ.

gr.Interface(fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=num_classes),
    title="Flower Classifier",
).launch(
    debug=True,
    share=True
)  # Gradio arayüzünü oluşturur ve başlatır.
