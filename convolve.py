"""
* Read images
* Extract features from each image by convolving with filters
"""

import numpy as np
import matplotlib.pyplot as plt

image_size = 28 # width and length

"""
    Admin Functions
"""
# Read from our file
def read_images(filename):
    data = np.loadtxt(filename, delimiter=",", dtype='int')
    return data


# Dictionary to map each digit to its list of images
def make_digit_map(data):
    digit_map = {i:[] for i in range(10)}
    for row in data:
        digit_map[row[0]].append(row[1:].reshape((image_size, image_size)))
    return digit_map


# im is the target image
# k is the kernel
# returns the convolution image, without reversing k
def convolve(im, k):
    kh, kw = k.shape
    imh, imw = im.shape
    im_w_border = np.zeros((kh + imh - 1, kw + imw -1))
    im_w_border[kh-1:kh-1+imh, kw-1:kw-1+imw] += im
    new_img = np.array([[np.sum(k*im_w_border[j+kw-1:j+2*kw-1, i+kh-1:i+2*kh-1]) \
                for i in range(imw)] for j in range(imh)], dtype='int')
    new_img[new_img>255] = 255
    new_img[new_img<0] = 0
    
    return new_img[kh-1:kh-1+imh, kw-1:kw-1+imw]
    


    
kernel = np.array([[-1, 1],[-1, 1]])



data = read_images("mnist_small.csv")
digit_map = make_digit_map(data)

imgr = digit_map[4][0].reshape((image_size*1, image_size/1))
plt.imshow(imgr, cmap=plt.cm.binary)
plt.show()

imgrc = convolve(imgr, kernel)
plt.imshow(imgrc, cmap=plt.cm.binary)
plt.show()


