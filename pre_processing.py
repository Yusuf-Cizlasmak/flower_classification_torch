import logging
import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


################################? Okuma ve yazma işlemleri için fonksiyonlar ############################


def check_dir(dir_path):
    """
    İlgili dizini kontrol eder .Eğer belirtilen dizin yoksa oluşturur
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        logging.info(f"{dir_path} created")
    else:
        logging.info(f"{dir_path} already exists")


def load_image(image_path):
    """
    Görüntülerin RGB formatında yüklenmesi için kullanılır
    """

    img = Image.open(image_path).convert("RGB")

    return img


def read_images_from_folder(folder_path):
    """
    Resimleri belirtilen klasörden okur ve bir liste olarak döndürür.

    output: [PIL.Image.Image, PIL.Image.Image, ...]
    """

    # Folder path control
    check_dir(folder_path)

    # Resimleri okuma
    images = [load_image(image_path) for image_path in glob(folder_path + "/*")]

    logging.info(f"{len(images)} images loaded from {folder_path}")

    return images


def change_resolution(image, max_width, max_height):
    """
    Resimlerin çözünürlüğünü değiştirmek için kullanılır. İleride crop'lama işlemi için ideal sonuçlar için.
    """
    width, height = image.size

    if width > max_width or height > max_height:
        # görüntünün oranını almak istioruz. Kare ve kenarları siyah olmaması adına

        ratio = min(max_width / width, max_height / height)

        # yeni boyutlarını vermek için

        new_width, new_height = int(width * ratio), int(height * ratio)

        image = image.resize((new_width, new_height), Image.LANCZOS)

        logging.info(
            f"Image resolution changed from {width}x{height} to {new_width}x{new_height}"
        )

        return image


def show_images(images):
    plt.figure(figsize=(20, 10))
    plt.axis("off")
    plt.imshow(images)
    plt.show()


def save_image(image, save_path):
    # Eğer resim numpy array ise PIL Image'e çevirme
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    # Eğer resim PIL Image değilse hata verme
    if not isinstance(image, Image.Image):
        raise ValueError("Image must be a numpy array or PIL Image")

    # Eğer resim RGB değilse RGB'ye çevirme
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Kaydetme işlemi
    if image.size[0] > 0 and image.size[1] > 0:  # Check if the image is not empty
        image.save(save_path)
        logging.info(f"Saved image to {save_path}")
    else:
        logging.warning(f"Cannot save empty image to {save_path}")
        


def save_images_to_dir(images, dir_path):
    os.makedirs(dir_path, exist_ok=True)

    # Dizini kontrol etme
    check_dir(dir_path)

    for i, image in tqdm(enumerate(images, 1)):
        # Resimleri kaydedileceği yeri ve dosya adını belirleme
        save_path = os.path.join(dir_path, f"{i}.jpg")

        # yukarıdaki fonksiyonu kullanarak resimleri kaydetme
        save_image(image, save_path)


def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        logging.info(f"Directory created: {directory}")


def get_images_from_dir(dir_path):
    # Dizin kontrolü
    check_dir(dir_path)

    # Dizin içindeki dosyaları listeleme
    files = os.listdir(dir_path)


    # Dosya uzantılarını filtreleme
    images = list(filter(lambda x: x.endswith(".jpg"), files))

    # Dosya yollarını birleştirme
    images_path = [os.path.join(dir_path, img) for img in images]

    return images_path
