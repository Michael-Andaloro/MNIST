import numpy as np
import matplotlib.pyplot as plt

image_size = 28 # width and length

# Read from our file
data = np.loadtxt("mnist_small.csv", delimiter=",") 

# Take a look, make sure they are floats
print data


# Extract all but first column
imgs = np.asfarray(data[:, 1:])

# Grab first image
img = imgs[0].reshape((image_size, image_size))

# Draw it using matplotlib
plt.imshow(img, cmap="Greys")
plt.show()
