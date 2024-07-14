import os
import shutil
import logging
from tqdm import tqdm
import glob
import random


def creating_dataset(classes):
    """
    ilgili dosya adları list şeklinde verilir ve ilgili cluster'lardan resimleri toplar.
    """
    os.makedirs("dataset",exist_ok=True)
    

    for class_ in tqdm(classes):

        os.makedirs(f"dataset/{class_}",exist_ok=True)

        for cluster in range(6):
            
            #Her bir resmin yolunu alma.
            image_paths= glob.glob(f"clustered_images/{class_}/cluster_{cluster}/*.jpg")

            for image_path in image_paths:
                shutil.copy(image_path,f"dataset/{class_}/")


        logging.info(f"{class_} is created")



classes = ["Lilly","Lotus","Orchid","Sunflower","Tulip"]


creating_dataset(classes)

# ? DOSYA YOLLARI BELİRLENMESİ
DATASET_PATH = "./dataset"
TRAIN_PATH = "model_dataset/train"
TEST_PATH = "model_dataset/test"
VAL_PATH = "model_dataset/val"
SPLIT_RATIO = 0.8
VAL_RATIO = 0.02 # 5 adet



# ? DATASET KLASÖRÜNÜN OLUŞTURULMASI
os.makedirs(TRAIN_PATH, exist_ok=True)
os.makedirs(TEST_PATH, exist_ok=True)
os.makedirs(VAL_PATH, exist_ok=True)

for class_folder in os.listdir(DATASET_PATH):
    # tam path oluşturulması
    class_path = os.path.join(DATASET_PATH, class_folder)

    # sadece klasörlerin alınması
    if not os.path.isdir(class_path):
        continue

    files = [
        f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))
    ]

    # ? VERİLERİN KARIŞTIRILMASI
    random.shuffle(files)  # bu şekilde belirli bir sıra olmadan karıştırılmış olur.

    split_index = int(len(files) * SPLIT_RATIO)
    val_index = int(len(files) * (SPLIT_RATIO + VAL_RATIO))

    train_files = files[:split_index]
    val_files = files[split_index:val_index]
    test_files = files[val_index:]

    # ? VERİLERİN dosya yolunu oluşturulması
    os.makedirs(os.path.join(TRAIN_PATH, class_folder), exist_ok=True)
    os.makedirs(os.path.join(VAL_PATH, class_folder), exist_ok=True)
    os.makedirs(os.path.join(TEST_PATH, class_folder), exist_ok=True)

    for f in train_files:
        shutil.copy(
            os.path.join(class_path, f), os.path.join(TRAIN_PATH, class_folder, f)
        )

    for f in val_files:
        shutil.copy(
            os.path.join(class_path, f), os.path.join(VAL_PATH, class_folder, f)
        )

    for f in test_files:
        shutil.copy(
            os.path.join(class_path, f), os.path.join(TEST_PATH, class_folder, f)
        )

print("Dataset is splitted")
