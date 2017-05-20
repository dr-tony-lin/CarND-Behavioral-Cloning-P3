'''
Utilities
'''
import math
from os.path import basename
import threading
import csv
import numpy as np
import cv2
from PIL import Image
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

def randomize(image, scale):
    '''
    Randomize the image

    Arguments:
    image: the image
    scale: the scale of randomization
    '''
    noise = cv2.randu(np.empty(image.shape, dtype=np.float32), 0, scale)
    return noise + image

def normalize(img):
    '''
    Normalize the grayscale image

    Arguments:
    a: the grayscale image to normalize
    Return: the normalized grayscale image
    '''
    if img.dtype != np.float32:
        img = np.array(img, dtype=np.float32)
    low = np.amin(img, axis=(0, 1))
    high = np.amax(img, axis=(0, 1))
    mid = (high + low) * 0.5
    dis = (high - low + 0.1) * 0.5  # +0.1 in case min = max
    return (img - mid) / dis

def grayscale(img):
    '''
    Convert the image to grayscale

    Arguments:
    img: the image to convert to grayscale
    Return: the converted grayscale image

    '''
    return cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))

def normal_gray(images, randomize_scale=None):
    '''
    Convert the images into normalized grayscale images for processing

    Arguments:
    images: an array of RGB images
    Returns: an array of normalized grayscale images
    '''
    if randomize_scale is not None:
        return np.array([normalize(randomize(grayscale(img), randomize_scale)) for img in images], dtype=np.float32)
    else:
        return np.array([normalize(grayscale(img)) for img in images], dtype=np.float32)

def scale_images(images, size):
    '''
    Scale the images

    Arguments:
    images: the array of image to scale
    size: the new size

    Return: an numpy array of scaled images
    '''
    return np.array([cv2.resize(image, size) for image in images])

def flip_image(image):
    '''
    Flip the image horizontally

    Arguments:
    image: the image to flip
    measurement: the directional measurement associated with the image
    '''
    image = np.fliplr(image)
    return image

def rand_visual(images, rows=4, columns=16, size=(2,2), fig=None, indices=None):
    '''
    Randomly show images from images

    Arguments:
        images: the image array
        rows: number of rows to show
        columns: number of columns to show
        fig: the matplotlib figure to show the images
        indices: the indices of images to show
    '''
    print(images.shape)
    nTypes = len(images)
    nImages = len(images[1])
    if fig is None:
        fig = plt.figure()
        fig.set_size_inches(columns*size[0], rows*nTypes*size[1])

    for i in range(rows):
        if indices is None:
            rands = np.random.randint(nImages-1, size=columns)
        else:
            rands = indices[i]
        for j in range(nTypes):
            for k in range(min(len(rands), columns)):
                plot_idx = (nTypes*i + j) * columns + k + 1
                plt.subplot(rows*nTypes, columns, plot_idx)
                if len(images[j][rands[k]].shape) < 3:
                    plt.imshow(images[j][rands[k]], cmap='gray')
                else:
                    plt.imshow(images[j][rands[k]])
    return plt

def convolution_output_size(size, kernel, stride, padding='SAME'):
    '''
    Comute the output image size after the convolution layer

    Arguments:
    size: the input size, tuple of two integers
    kernel: the kernel size, tuple of two integers
    stride: the stride size, tuple of two integers
    padding: the padding
    '''
    if padding == 'SAME':
        return (math.ceil(float(size[0]) / float(stride[0])),
                math.ceil(float(size[1]) / float(stride[1])))
    else:
        return (math.ceil(float(size[0] - kernel[0] + 1) / float(stride[0])),
                math.ceil(float(size[1] - kernel[1] + 1) / float(stride[1])))

def DrivingDataGenerator(images, data, batch_size=256, channel_first=True):
    """
    Driving data generator.
    Arguments:
        images - array of sample images, it vcan be an array of images or file names
        data - array of the corresponding images' sensor data
        batch_size - the batch size
    """
    while True:
        images, data = shuffle(images, data)
        for index in range(0, len(images), batch_size):
            img_out = []
            for i in range(batch_size):
                if index + i >= len(images):
                    break
                image = images[index + i]
                if isinstance(image, str): # file name
                    if 'flip-' in image:
                        image = image[5:]
                        img_out.append(np.array((Image.open(image).transpose(Image.FLIP_LEFT_RIGHT))))
                    else:
                        img_out.append(np.array(Image.open(image)))
                else: # An image
                    img_out.append(image)
            img_out = np.array(img_out)
            if channel_first:
                img_out = img_out.reshape((img_out.shape[0], img_out.shape[3], img_out.shape[1], img_out.shape[2]))
            yield (img_out, np.array(data[index:index+batch_size]))

def get_samples(dirs, flip=True, all_cameras=False, cr=[0.2, -0.2]):
    '''
    Get the samples from configured sample folders
    flip - include flipped images
    all_cameras - include images from left and right cameras
    cr - camera corrections for all cameras
    '''
    images = []
    data = []
    for folder in dirs:
        repeats = None
        percent = None
        if '*' in folder:
            folder, r = folder.split('*')
            value = float(r.strip())
            if value > 1.01:
                repeats = int(math.ceil(value))
                percent = value / repeats
            elif value < 0.99:
                percent = value

        folder = folder.strip() + "/"
        images_folder = folder + "IMG/"
        folder_images = []
        folder_data = []
        with open(folder + "driving_log.csv", 'r') as file:
            rows = csv.reader(file)
            first = True
            for row in rows:
                if len(row) < 7: # invalid row, skip it
                    continue
                if first: # skip caption
                    first = False
                    continue
                center = float(row[3])
                m = []
                imgs = []
                if all_cameras: # Include all images
                    # create adjusted steering measurements for the side camera images
                    left = center + cr[0]
                    right = center + cr[1]
                    imgs += [images_folder + basename(row[1].strip()),
                             images_folder + basename(row[0].strip()),
                             images_folder + basename(row[2].strip())]
                    m += [left, center, right]
                else: # Only center images
                    imgs.append(images_folder + basename(row[0].strip()))
                    m.append(center)
                if flip: # Also include flipped images
                    imgs += ['flip-' + im for im in imgs]
                    m += [-d for d in m]
                folder_images += imgs
                folder_data += m
        if repeats  is not None:
            folder_images = folder_images * repeats
            folder_data = folder_data * repeats
        elif percent is not None:
            folder_images, folder_data = shuffle(folder_images, folder_data)
            # Split the training samples with the given percentage
            size = int(len(folder_images) * percent)
            folder_images, folder_data = folder_images[:size], folder_data[:size]
        images += folder_images
        data += folder_data
    return images, data

def accept_inputs(callback):
    '''
    Accept user inputs
    callback -  the callback to call when received an input
    '''
    def _input():
        run = True
        while run:
            line = input()
            if line:
                callback(line.strip().lower())

    thread = threading.Thread(target=_input)
    thread.setDaemon(True)
    thread.start()
    return thread
