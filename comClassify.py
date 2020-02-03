"""
* Read images
* Extract features from each image
* Find center of mass of features by label
* Classify images by nearest center of mass
* Test accuracy of classification on training data
* Test accuracy on test data
"""

import numpy as np
import matplotlib.pyplot as plt

image_size = 28 # width and length

"""
    FEATURE FUNCTIONS
"""

# Number of b/w transitions along every other row
# 14 dimensions
# Rob Hochberg
def waviness(img):
    img2 = img.copy()
    img2[img2 > 0] = 255 # Any pixel not white becomes black
    return np.sum(abs(img2[:,1:] - img2[:,:-1])/255, axis=1)[::2].tolist()


# For each row, count number of non-white pixels.
# For each column, count number of non-white pixels.
# Total dimension: 28 x 2 = 56.
# Duy, Michael Bujard, Paul
def hv_weights(image):
    # row_nonzero_counts = np.count_nonzero(image, axis=1)
    # col_nonzero_counts = np.count_nonzero(image, axis=0)
    row_nonzero_counts = np.asarray([sum([0 if num == 0 else 1 for num in a_row]) for a_row in image])
    col_nonzero_counts = np.asarray([sum([0 if num == 0 else 1 for num in a_col]) for a_col in np.transpose(image)])
    #print "Hi", np.concatenate((row_nonzero_counts, col_nonzero_counts))
    return np.concatenate((row_nonzero_counts, col_nonzero_counts)).tolist()


# Anthony/Amelia/Sri
# Count the vertical straight lines in an image,
# taking an image which is not necessarily blocked in black beforehand
vertical_line_len = 10
def vertical_lines(image):
    lines = [0] # Have a 0 to the left of the first char in the list                                                                  
    for x in range(image_size - 1):
        counter = 0
        max = 0
        for y in range(image_size - 1):
            pixel = 1 if image[y][x] > 0 else 0
            if counter == 0:
                counter += pixel
            else:
                if pixel == 1:
                    counter += pixel
                else: #if there's a gap                                                                                                               
                    max = np.maximum(max, counter)
                    counter = 0
        max = np.maximum(max, counter)
        lines.append(1 if max >= vertical_line_len else 0)
    #print(lines)                                                                                                                                     
    line_count = 0
    for i in range(1, image_size):
        if lines[i] == 0 and lines[i-1] == 1:
            line_count += 1
    return [line_count]


# Minh/Matt/David
# Build the vertical waviness of the image, and sum it with the horizontal waviness.
# gets horizontal waviness, rotates matrix 90 degrees and runs formula for horizontal 
# waviness again for vertical waviness, add both waviness features together 
def combineWavy(img):
    x = np.sum(abs(img[1:]-img[:-1])/255,axis=1)[::2]
    img = np.rot90(img)
    y = np.sum(abs(img[1:]-img[:-1])/255,axis=1)[::2]
    z = np.add(x,y)
    return z.tolist()


# Michael, Michael, and Will
# Feature: top-heavy vs bottom-heavy 
# "bottom-heavy" means that there's more going on in the bottom of the picture/digit.  
#   For example: 6 is bottom-heavy because there are more transitions in the bottom of the digit.
# "top-heavy" means that there's more going on in the top of the picture/digit.  
#   For example: 4 and 9 are top-heavy as there are more transitions in the top of the digit
# Returns a list with the top-weight and bottom-weight as (top, bottom)
def top_bottom_balance(img) -> list:
    # Get the number of color transitions per row in the image
    transition_array = color_transition_array(img)
    midpoint = len(transition_array)//2 #Get the midpoint of the array
    # Split the transition array into top and bottom of the image
    top_array = transition_array[:midpoint]
    bottom_array = transition_array[midpoint:]
    # Sum the values for number of color transitions in the top and bottom of the picture
    top_value = np.sum(top_array)
    bottom_value = np.sum(bottom_array)
    return [top_value, bottom_value]

# Michael, Michael, and Will
# Split the image in half and compare the weights
# (# of color transitions, can be easily modified to do sum of non-white
# pixel values) of the two halves.
# The function returns a tuple in the form (top_half, bottom_half), 
# with the bigger number representing which part of the image has more going on
# Returns a single array with the number of color transistions per row, 
# corresponding to that index in the returned array
def color_transition_array(img) -> np.ndarray: 
    img2 = img.copy()
    img2[img2 > 0] = 255 # Any pixel not white becomes black
    return (np.sum(abs(img2[:, 1:] - img2[:, :-1])/255, axis=1))
 

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


# Extract features
# fnlist is a list of feature-generating functions, each of which should
#   take a 28x28 grayscale (0-255) image, 0=white, and return a 1-d array
#   of numbers
# Returns a map: digit -> nparray of feature vectors, one row per image
def build_feature_map(digit_map, fnlist):
    fmap = {i:[] for i in range(10)}
    for digit in fmap:
        for img in digit_map[digit]:
            feature_vector = []
            for f in fnlist:
                feature_vector += f(img)
            fmap[digit].append(feature_vector)
    return fmap


# Find center of mass of each digit's feature vectors
# feature_map is a map from each digit to a list of feature vectors
# Returns a map: digit -> Center of mass
def find_com(feature_map):
    com_map = {}
    for digit in feature_map:
        feature_matrix = np.asfarray(feature_map[digit])
        com_map[digit] = np.mean(feature_matrix, axis=0)
    return com_map


"""
    TESTS
"""
# Anthony/Michael A./David
# feature_map maps digits to list of list of features
# com_map maps digits to a com of that digit's features
# Returns the fraction of correctly-classified images
def testAMD(feature_map, com_map):
    ccom = None
    countr = 0.0
    countw = 0.0
    for key in feature_map:
        for ele in feature_map[key]:
            mdist = float("inf")
            for ckey in com_map:
                dist = abs(np.linalg.norm(com_map[ckey]-ele))
                if dist <= mdist:
                    mdist = dist
                    ccom = ckey
            if ccom == key:
                countr +=1  
            else:
                countw +=1
    per = countr/(countr +  countw)
    return per 


# Sri / Paul / Michael Bujard
# Test
# feature_map maps digits to list of list of features
# com_map maps digits to a com of that digit's features
def testSPM(feature_map, com_map):
    num_correct = 0
    num_total = 0
    for correct_digit, list_of_features in feature_map.items():
        for some_features in list_of_features:
            smallest_distance = float("inf")
            guess_digit = None
            for candidate_digit, a_com in com_map.items():
                distance = np.linalg.norm(a_com - some_features)
                if distance < smallest_distance:
                    smallest_distance = distance
                    guess_digit = candidate_digit
            if guess_digit == correct_digit:
                num_correct += 1
            num_total += 1
    return num_correct / num_total


# Rob Hochberg
# feature_map maps digits to list of list of features
# com_map maps digits to a com of that digit's features
def testR(feature_map, com_map):
    predictions = [[0 for i in range(10)] for j in range(10)]
    for correct_digit, list_of_features in feature_map.items():
        for some_features in list_of_features:
            smallest_distance = float("inf")
            guess_digit = None
            for candidate_digit, a_com in com_map.items():
                distance = np.linalg.norm(a_com - some_features)
                if distance < smallest_distance:
                    smallest_distance = distance
                    guess_digit = candidate_digit
            predictions[correct_digit][guess_digit] += 1
    return predictions

# List of implemented feature functions
features = [waviness]

# train
data = read_images("mnist_medium.csv")
digit_map = make_digit_map(data)
feature_map = build_feature_map(digit_map, features)
com = find_com(feature_map)

# Test on training data
print("AMD Test", testAMD(feature_map, com))
print("SPM Test", testSPM(feature_map, com))
print("R Test\n", np.array(testR(feature_map, com)))

# Test on test data
data = read_images("mnist_medium_test.csv")
digit_map = make_digit_map(data)
feature_map = build_feature_map(digit_map, features)
print("AMD Test", testAMD(feature_map, com))
print("SPM Test", testSPM(feature_map, com))
print("R Test\n", np.array(testR(feature_map, com)))




"""
for img in digit_map[4]:
    imgr = img.reshape((image_size*1, image_size/1))
    plt.imshow(imgr, cmap=plt.cm.binary)
    plt.show()
"""

