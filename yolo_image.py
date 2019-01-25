import cv2
from darkflow.net.build import TFNet
import matplotlib.pyplot as mp

# %config InlineBackend.figure_format = 'svg'

options = {
    'model': 'cfg/yolo.cfg',
    'load': 'bin/yolo.weights',
    'threshold': 0.3,
    'gpu': 1.0
}

tfnet = TFNet(options)  # get engine ready

# read color image and convert to RGB
img = cv2.imread('sample_img/sample_dog.jpg', cv2.IMREAD_COLOR)       # put your image
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# now predict the image using YOLO
result = tfnet.return_predict(img)
# print(result)

img.shape  # now we find the shape of image

# now show the image information
for l, k in enumerate(result):
    print(k)
    tl = (result[l]['topleft']['x'], result[l]['topleft']['y'])
    br = (result[l]['bottomright']['x'], result[l]['bottomright']['y'])
    label = result[l]['label']
    confidence = result[l]['confidence']

    text = '{}: {:.0f}%'.format(label, confidence * 100)

    # now draw rectangle box and lavel and display
    img = cv2.rectangle(img, tl, br, (0, 255, 0), 2)
    img = cv2.putText(img, text, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

mp.imshow(img)
mp.show()
