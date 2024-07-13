import cv2 
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
def plot_means(labels, pixel_values, centers):
    A = pixel_values[labels.ravel() == 0]
    B = pixel_values[labels.ravel() == 1]

    plt.scatter(A[:, 0], A[:, 0])
    plt.scatter(B[:, 0], B[:, 0], c='r')
    plt.scatter(centers[:, 0], centers[:, 0], s=80, c='y', marker='s')
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')
    plt.savefig("Classes.png")




def unsupervised_crop(pil_image, square=False):
    img = np.array(pil_image)
    
    #GRAYSCALE
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    pixel_values = img.reshape((-1,1))
    pixel_values = np.float32(pixel_values)

    # Define criteria and apply K-Means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    #k sınıf sayısı
    k = 2

    #kmeans
    _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Convert centers to uint8 and labels to the original image shape
    centers = np.uint8(centers)
    labels = labels.flatten()

    plot_means(labels,pixel_values,centers)


    # Segment the original image
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(pixel_values.shape)
    
    points = np.column_stack(np.where(segmented_image == np.min(centers)))
    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)
    
    min_side = min((x_max - x_min), (y_max - y_min))
    x_mean, y_mean = int((x_max + x_min) / 2), int((y_max + y_min) / 2)
    
    squared_x_min, squared_x_max = x_mean - int(min_side / 2), x_mean + int(min_side / 2)
    squared_y_min, squared_y_max = y_mean - int(min_side / 2), y_mean + int(min_side / 2)

    if not square:
        return img[y_min:y_max, x_min:x_max]
    else:
        return img[squared_y_min:squared_y_max, squared_x_min:squared_x_max]

