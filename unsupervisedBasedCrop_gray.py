import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image





def unsupervised_crop_gray(pil_image, square=False):
    img = np.array(pil_image)
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

    # Reshape image to a 2D array of pixels
    pixel_values = gray.reshape((-1, 1))
    pixel_values = np.float32(pixel_values)

    # Define criteria and apply K-Means clustering
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    k = 2
    _, labels, (centers) = cv.kmeans(
        pixel_values, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS
    )

    # Convert centers to uint8 and labels to the original image shape
    centers = np.uint8(centers)
    labels = labels.flatten()


    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(gray.shape)

    points = np.column_stack(np.where(segmented_image == np.min(centers)))
    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)

    min_side = min((x_max - x_min), (y_max - y_min))
    x_mean, y_mean = int((x_max + x_min) / 2), int((y_max + y_min) / 2)

    squared_x_min, squared_x_max = x_mean - int(min_side / 2), x_mean + int(
        min_side / 2
    )
    squared_y_min, squared_y_max = y_mean - int(min_side / 2), y_mean + int(
        min_side / 2
    )

    if not square:
        return img[y_min:y_max, x_min:x_max]
    else:
        return img[squared_y_min:squared_y_max, squared_x_min:squared_x_max]
