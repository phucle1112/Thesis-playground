# instruction 1: https://projects.raspberrypi.org/en/projects/astropi-ndvi/5

import cv2
import numpy as np
from fastiecm import fastiecm
from picamera import PiCamera
import picamera.array
import math

# original = cv2.imread('/home/pi/Computer-Vision/src/images/plant_vs_plant.png')
cam = PiCamera()
cam.rotation = 180
cam.resolution = (1920, 1080) # Uncomment if using a Pi Noir camera
# cam.resolution = (2592, 1952) # Comment this line if using a Pi Noir camera
stream = picamera.array.PiRGBArray(cam)
cam.capture(stream, format='bgr', use_video_port=True)
original = stream.array

def display(image, image_name):
    image = np.array(image, dtype=float)/float(255)
    shape = image.shape
    height = int(shape[0]/2)
    width = int(shape[1]/2)
    image = cv2.resize(image, (width, height))
    cv2.namedWindow(image_name)
    cv2.imshow(image_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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

contrasted = contrast_stretch(original)
cv2.imwrite('/home/pi/Computer-Vision/src/images/contrasted_plant.png', contrasted)

display(original, 'Original plants')
display(contrasted, 'Contrasted original plants')

ndvi = calc_ndvi(contrasted)
cv2.imwrite('/home/pi/Computer-Vision/src/images/ndvi_plants.png', ndvi)
display(ndvi, 'NDVI plants')
print('ndvi in this img is ', ndvi)

ndvi_contrasted_plant = contrast_stretch(ndvi)
cv2.imwrite('/home/pi/Computer-Vision/src/images/ndvi_contrasted_plants.png', ndvi_contrasted_plant)
display(ndvi_contrasted_plant, 'NDVI Contrasted plants')

color_mapped_prep = ndvi_contrasted_plant.astype(np.uint8)
color_mapped_image = cv2.applyColorMap(color_mapped_prep, fastiecm)
cv2.imwrite('/home/pi/Computer-Vision/src/images/color_mapped_image.png', color_mapped_image)
cv2.imwrite('/home/pi/Computer-Vision/src/images/original_stream.png', original)
display(color_mapped_image, 'Color mapped')


