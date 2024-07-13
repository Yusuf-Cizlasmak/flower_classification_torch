from PIL import Image
from tqdm import tqdm
import pandas as pd
import os
from img2vec_pytorch import Img2Vec

from pre_processing import (change_resolution, check_dir,
                            read_images_from_folder, save_images_to_dir,
                            get_images_from_dir,load_image)



from unsupervisedBasedCrop_gray import unsupervised_crop_gray



classes = ["Lilly","Lotus","Orchid","Sunflower","Tulip"]
# Verileri kontrol etme.

for class_ in classes:

    check_dir(f"./flower_images/{class_}")


    # Resimleri okuma
    images = read_images_from_folder(f"./flower_images/{class_}")[:500]
    # Resimlerin boyutları olarak değiştirme

    size = 1024

    resized_images = [change_resolution(image, size, size) for image in images]



    # Crop işlemi
    crop_images = [unsupervised_crop_gray(image) for image in tqdm(images)]


    # ? Cropped image after

    save_images_to_dir(crop_images, f"./cropped_images/{class_}")


    # * Get Embeedings

    paths = get_images_from_dir(f"./cropped_images/{class_}")

    images= [load_image(path) for path in paths]


    img2vec=Img2Vec(cuda=False) #cuda=True ile GPU üzerinde çalıştırma

    embeddings= img2vec.get_vec(images) #Resimlerden embedding çıkarma

    print(embeddings.shape) # (n, 512) n: resim sayısı


    df = pd.DataFrame(embeddings)

    df['filepaths'] = paths

    #csv olarak kaydetme

    os.makedirs("./embeddings",exist_ok=True)

    df.to_csv(f"./embeddings/{class_}.csv",index=False)


