#COMMENTS GUIDE:
#   = explination of line (located beneath and indented to line)
##  = temporarily removed (if located beneath and indented to line, signifies temporariness)
### = section header

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
import SharedFunctions
    #import modules

files  = glob.glob("/Users/Gidonka/Documents/Programming/NYU/MachineLearning/IndianMovies/IMFDB_Final/*/*/*/*.jpg", recursive=True)
    #files = filenames of image files 
labels = glob.glob("/Users/Gidonka/Documents/Programming/NYU/MachineLearning/IndianMovies/IMFDB_Final/*/*/*.txt", recursive=True)
    #labels = filenames of text files containing labels

filenames_and_labels, train_dataset, valid_dataset, test_dataset = SharedFunctions.setup(files, labels)
    #get entire dataset as well as individual datasets from setup function

image_size = 28
num_labels = 7
batch_size = 128
    #define some variables

tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, 3))
    #create placeholder for dataset
tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    #create placeholder for labels



sess = tf.InteractiveSession()
    #launch session

train_writer = tf.train.SummaryWriter('./summary', sess.graph)
    #create writer graph

accuracy, cross_entropy, y_conv, keep_prob = SharedFunctions.network(tf_train_dataset, tf_train_labels)
    #get accuracy, cross entropy (loss), y_conv, and dropout placeholder from network function (which uses batch creation function which uses square function)

train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
    #define Adam as preferred optimizer, define learning speed, and define what should be minimized (cross entropy)

saver = tf.train.Saver()
    #create saver

sess.run(tf.initialize_all_variables())
    #run session, initialize variables

merged_summaries = tf.merge_all_summaries()
    #merge graph summaries

#train network: repeat number of times, evaluate every number of steps, control dropout rate

    #define set position at dataset (to be used for creating valid batches) as 0
for i in range(20000):
    #create loop to iterate for number of steps wanted
    position = random.randint(0, len(train_dataset)-batch_size)
        #define new random position for taking batch (for use on train data not valid)
    dataset = train_dataset
        #define train dataset as dataset to be used
    batch_data, batch_labels = SharedFunctions.create_batch(dataset, position, image_size, filenames_and_labels)
        #get batch data and batch labels from create batch function
    sess.run(train_step, feed_dict={tf_train_dataset: batch_data, tf_train_labels: batch_labels, keep_prob: 0.5})
        #run session with defined settings, use feed dictionary created from batch data and batch labels, define probability that a neuron will be kept as 0.5
    print("step %d"%(i))
        #print step number
    if i%100 == 0:
        #every n steps
        loss, acc, summaries = sess.run([cross_entropy, accuracy, merged_summaries], feed_dict={tf_train_dataset: batch_data, tf_train_labels: batch_labels, keep_prob: 0.5})
            #calculate loss and accuracy
        print("loss %f, accuracy %f"%(np.average(loss), acc))
            #print loss and accuracy
        train_writer.add_summary(summaries, i)
            #add loss and accuracy to TensorBoard graph
        saver.save(sess, 'my-model', global_step=i)
            #save variables so that they can be restored later for validation testing
        
