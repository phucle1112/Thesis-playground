from ast import Num
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.io import imread, imshow
from numpy import ndarray
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
from fastiecm import fastiecm


# ! Processing imported picture (local storage)
ikea_plant = cv2.imread('C:\\Users\\phucl\\Bosman van Zaal\\PEAK - General\\05-Users\\Phuc (Intern)\\THESIS\\WORKLOADS\\Phase - Implementation\\Code\\Thesis-playground\\src\\images\\ikea_plant\\park.png')
# plt.figure(num = None, figsize = (8, 6), dpi = 80)

# Display image function
def display(image, image_name):
    image = np.array(ikea_plant, dtype=float)/ float(255)
    shape = image.shape
    height = int(shape[0]/2)
    width = int(shape[1]/2)
    image = cv2.resize(image, (width, height))
    cv2.namedWindow(image_name)
    cv2.imshow(image_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def rgb_splitter(image):
    rgb_list = ['Reds', 'Greens', 'Blues']
    fig, ax = plt.subplots(1, 3, figsize=(17,7), sharey = True)
    for i in range(3):
        ax[i].imshow(image[:,:,i], cmap= rgb_list[i])
        ax[i].set_title(rgb_list[i], fontsize=22)
        ax[i].axis('off')
    fig.tight_layout()
    plt.show()

def contrast_stretch (img):
    in_min = np.percentile(img, 5)
    in_max = np.percentile(img, 95)
    out_min = 0.0
    out_max = 255.0
    out = img - in_min
    out *= ((out_min - out_max) / (in_min - in_max))
    out = out + in_min
    return out

def calc_ndvi(image):
    b, g, r = cv2.split(image)
    bottom = (r.astype(float) + b.astype(float))
    bottom[bottom == 0] = 0.01
    ndvi = (b.astype(float) - r) / bottom
    rounded_ndvi = np.around(ndvi, decimals = 1)
    return rounded_ndvi

# rgb_splitter(ikea_plant)
display(ikea_plant, 'Original ikea')

# contrasted image
contrasted = contrast_stretch(ikea_plant)
cv2.imwrite('C:\\Users\\phucl\\Bosman van Zaal\\PEAK - General\\05-Users\\Phuc (Intern)\\THESIS\\WORKLOADS\\Phase - Implementation\\Code\\Thesis-playground\\src\\images\\ikea_plant\\ikea_contrasted_original.png', contrasted)
display(contrasted, 'Contrasted original ikea')

# NDVI calculation
ndvi = calc_ndvi(contrasted)
cv2.imwrite('C:\\Users\\phucl\\Bosman van Zaal\\PEAK - General\\05-Users\\Phuc (Intern)\\THESIS\\WORKLOADS\\Phase - Implementation\\Code\\Thesis-playground\\src\\images\\ikea_plant\\ikea_ndvi.png', ndvi)
display(ndvi, 'NDVI ikea')

# contrasted ndvi pic
ndvi_contrasted = contrast_stretch(ndvi)
cv2.imwrite('C:\\Users\\phucl\\Bosman van Zaal\\PEAK - General\\05-Users\\Phuc (Intern)\\THESIS\\WORKLOADS\\Phase - Implementation\\Code\\Thesis-playground\\src\\images\\ikea_plant\\ikea_ndvi_contrasted.png', ndvi_contrasted)
display(ndvi_contrasted, 'NDVI Contrasted')

# color mapping
color_mapped_prep = ndvi_contrasted.astype(np.uint8)
color_mapped_image = cv2.applyColorMap(color_mapped_prep, fastiecm)
display(color_mapped_image, 'Color mapped')
cv2.imwrite('C:\\Users\\phucl\\Bosman van Zaal\\PEAK - General\\05-Users\\Phuc (Intern)\\THESIS\\WORKLOADS\\Phase - Implementation\\Code\\Thesis-playground\\src\\images\\ikea_plant\\ikea_color_mapped_image.png', color_mapped_image)