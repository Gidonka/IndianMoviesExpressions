from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from skimage.transform import resize
import glob
import os
import sys
import tarfile
from scipy import ndimage
import random
from random import shuffle
import tensorflow as tf

random.seed(42)

def setup(files, labels):
    filenames_and_labels = {}
        #empty dictionary containing filenames as keys and labels as values

    problem_line = 0
        #variable for counting how many lines (ergo how many files and labels) were problematic and will be unused
    problem_file = 0

    for labelfile in labels:
        #iterate through each labelfile in the collection of label files
        with open(labelfile, 'r') as searchfile:
            #open labelfile
            for line in searchfile:
                #iterate through each line in the labelfile
                split = line.split()
                    #split line by spaces (including tabs) into list of each item in line
                if len(split) <= 2 or split[2].endswith(".jpg") == False or "_" not in split[2]:
                    #if there are less than three items in the line (ergo item[2], filename, is missing) or it isn't a JPG or it doesn't have _
                    problem_line = problem_line + 1
                        #add 1 to problem line count
                    continue
                        #skip line (do not include in dictionary)
                removedpath = labelfile.rsplit("/", 1)[0]
                    #remove first element from the right after "/" from filepath of .txt file
                imagefilepath = removedpath + "/images/" + split[2]
                    #add to path "images" folder and filename from line in txt file
                filepath = os.path.abspath(imagefilepath)
                    #get path of image
                if not os.path.isfile(filepath):
                    problem_file += 1
                    continue
                    #if path doesn't exist, skip file and add to problem file counter
                if len(split) <= 11 or split[11] not in ["ANGER", "HAPPINESS", "SADNESS", "SURPRISE", "FEAR", "DISGUST", "NEUTRAL"]:
                    #if there are less than twelve items in the line (ergo item[11], expression label, is missing) or if the element does not match one of the seven labels
                    problem_line = problem_line + 1
                        #add 1 to problem line count
                    continue
                        #skip line (do not include in dictionary)
                label = split[11]
                    #label is the twelfth element in the split line
                filenames_and_labels[filepath] = label
                    #add filename and label as key-value pair to dictionary

    percent_problematic = float(problem_line)/len(files)*100
        #calculate percent problematic
    print("Percent of lines skipped because of problems:", percent_problematic)
        #print percent problematic

    percent_problem_files = float(problem_file)/len(files)*100
    print("Percent of files not found:", percent_problem_files)

    ##random.shuffle(filenames_and_labels)

    tr_amount = int(0.7 * len(filenames_and_labels))
    va_amount = int(0.6 * (len(filenames_and_labels)-tr_amount))
    te_amount = int(0.4 * (len(filenames_and_labels)-tr_amount))
        #define number of elements in each dataset

    ordered_filenames_and_labels_keys = sorted(filenames_and_labels.keys())
    train_dataset = list(ordered_filenames_and_labels_keys)[:tr_amount]
    train_dataset.sort()
    valid_dataset = list(ordered_filenames_and_labels_keys)[tr_amount:tr_amount+va_amount]
    valid_dataset.sort()
    test_dataset  = list(ordered_filenames_and_labels_keys)[tr_amount+va_amount:]
    test_dataset.sort()
        #create datasets (filepaths), sort them in set order

    return filenames_and_labels, train_dataset, valid_dataset, test_dataset

image_size = 28
num_labels = 7
batch_size = 128
    #define some variables

#crop and resize files
def square(image, image_size):
    if image.shape[1] > image.shape[0]:
        excess = (image.shape[1] - image.shape[0])/2
        excess = int(round(excess))
        image = image[:,(0 + excess):(image.shape[1] - excess)]
    elif image.shape[1] < image.shape[0]:
        excess = (image.shape[0] - image.shape[1])/2
        excess = int(round(excess))
        image = image[(0 + excess):(image.shape[0] -excess),:]
    else:
        excess = 0
        image = image
    image = resize(image, (image_size, image_size))
    return image

one_hot_dict = {"ANGER": [1, 0, 0, 0, 0, 0, 0], "HAPPINESS": [0, 1, 0, 0, 0, 0, 0], "SADNESS": [0, 0, 1, 0, 0, 0, 0], "SURPRISE": [0, 0, 0, 1, 0, 0, 0], "FEAR": [0, 0, 0, 0, 1, 0, 0], "DISGUST": [0, 0, 0, 0, 0, 1, 0], "NEUTRAL": [0, 0, 0, 0, 0, 0, 1]}
        #create dictionary for assigning one hot encoding to labels

def create_batch(dataset, position, image_size, filenames_and_labels):
    #define function for batch creation
    batch_of_filepaths = dataset[position:(position+batch_size)]
            #define breadth of batch for train or valid dataset
    batch_image_array = np.empty([len(batch_of_filepaths), image_size, image_size, 3])
        #define empty numpy array to be filled with image arrays. Each row is an image. Dimensions: batch_size (128) x image_size*image_size*channels (2352)
    batch_label_array = np.zeros([len(batch_of_filepaths), num_labels])
        #define empty numpy array to be filled with one hot encoded labels. Each row is a label. Dimensions: batch_size (128) x num_labels (7)
    for i, image in enumerate(batch_of_filepaths):
        #create loop to enumerate through the batch of filepaths
        face = io.imread(image)
                #read each image
        final_face = square(face, image_size)
                #call sqaure function on image
        batch_image_array[i,:] = final_face
                #fill each row of image array with an image array
            
        label = filenames_and_labels[image]
                #get matching label for each image using dictionary
        one_hot_label = one_hot_dict[label]
                #get one hot encoding for labels from one_hot_dict
        batch_label_array[i,:] = one_hot_label
                #fill each row of label array with a one hot encoded array
    batch_image_array -= 0.5
    return batch_image_array, batch_label_array

def network(tf_train_dataset, tf_train_labels):
    #####Convolution Network####

    #function for creating many weights
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)
    #function for creating many biasies
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    #function for convolving
    def conv2d(tf_train_dataset, W):
        return tf.nn.conv2d(tf_train_dataset, W, strides=[1, 1, 1, 1], padding='SAME')
    #function for pooling
    def max_pool_2x2(tf_train_dataset):
        return tf.nn.max_pool(tf_train_dataset, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    ###FIRST CONVOLUTION LAYER

    #create weight tensor ([dimension, dimension, number of inputs, number of outputs])
    W_conv1 = weight_variable([5, 5, 3, 32])
    #create bias tensor ([number of outputs])
    b_conv1 = bias_variable([32])

    #convolve x with weight, apply ReLU function, add bias
    h_conv1 = tf.nn.relu(conv2d(tf_train_dataset, W_conv1) + b_conv1)
    #max pool
    h_pool1 = max_pool_2x2(h_conv1)

    ###SECOND CONVOLUTION LAYER

    #create weight tensor ([dimension, dimension, number of inputs, number of outputs])
    W_conv2 = weight_variable([5, 5, 32, 64])
    #create bias tensor ([number of outputs])
    b_conv2 = bias_variable([64])

    #convolve x with weight, apply ReLU function, add bias
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    #max pool
    h_pool2 = max_pool_2x2(h_conv2)

    #FULLY CONNECTED LAYER

    #create weight tensor ([dimension, dimension, number of inputs, number of outputs])
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    #create bias tensor ([number of outputs])
    b_fc1 = bias_variable([1024])

    #reshape tensor from pooling layer
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    #apply ReLU, multiply by weight, add bias
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    #apply dropout to minimize overfitting- create placeholder for probability that neuron will be kept
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    #READOUT LAYER

    #apply softmax
    W_fc2 = weight_variable([1024, 7])
    b_fc2 = bias_variable([7])
    y_conv= tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    #train and evaluate: define cross entropy, minimize cross entropy, define ADAM optimizer as preferred method to do so
    #cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
    cross_entropy_array = tf.nn.softmax_cross_entropy_with_logits(y_conv, tf_train_labels)
    cross_entropy = tf.reduce_mean(cross_entropy_array)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(tf_train_labels,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.scalar_summary('accuracy', accuracy)
    tf.scalar_summary('loss', cross_entropy)
        #initialize SummaryWriter

    return accuracy, cross_entropy, y_conv, keep_prob
