import cv2
import os
from darkflow.net.build import TFNet
import matplotlib.pyplot as mp


def load_images(path='sample_img'):
    '''
    temp = os.listdir(path)

    images = []
    for i in temp:
        if i.endswith('jpg'):
            images.append(i)
    print(images)
    '''

    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]


# print(load_images())
filenames = load_images()

images = []
for file in filenames:
    images.append(cv2.imread(file, cv2.IMREAD_UNCHANGED))

# print(images)
for num, image in enumerate(images):
    cv2.imwrite(str(num)+'.png', image)

