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

tf_valid_dataset = tf.placeholder(tf.float32, shape=(None, image_size, image_size, 3))
    #create placeholder for dataset
tf_valid_labels = tf.placeholder(tf.float32, shape=(None, num_labels))
    #create placeholder for labels

sess = tf.InteractiveSession()

train_writer = tf.train.SummaryWriter('./summary', sess.graph)

accuracy, cross_entropy, y_conv, keep_prob = SharedFunctions.network(tf_valid_dataset, tf_valid_labels)

save_path = tf.train.latest_checkpoint('./')

saver = tf.train.Saver()

restored = saver.restore(sess, save_path)

train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)

sess.run(tf.initialize_all_variables())

merged_summaries = tf.merge_all_summaries()

position = 0
correct_samples = 0.0
for i in range(int(len(valid_dataset)/batch_size)):
    dataset = valid_dataset
    position += batch_size
    batch_data, batch_labels = SharedFunctions.create_batch(dataset, position, image_size, filenames_and_labels)
    #sess.run(train_step, feed_dict={tf_valid_dataset: batch_data, tf_valid_labels: batch_labels, keep_prob: 0.5})
    print("step %d"%(i))
    print(batch_data.shape)
    print(batch_labels.shape)
    loss, acc, summaries = sess.run([cross_entropy, accuracy, merged_summaries], feed_dict={tf_valid_dataset: batch_data, tf_valid_labels: batch_labels, keep_prob: 0.5})
    print("loss %f, accuracy %f"%(np.average(loss), acc))
    train_writer.add_summary(summaries, i)
    correct_samples += acc*len(batch_data)
print("validation accuracy:", correct_samples/(int(len(valid_dataset)/batch_size)))
    
    
