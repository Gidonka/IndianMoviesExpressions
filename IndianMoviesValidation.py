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
    #launch tensorflow session

train_writer = tf.train.SummaryWriter('./summary', sess.graph)
    #define writer for tensorboard

accuracy, cross_entropy, y_conv, keep_prob = SharedFunctions.network(tf_valid_dataset, tf_valid_labels)
    #call network function, return variable placeholders

save_path = tf.train.latest_checkpoint('./checkpoints')
#save_path = "./checkpoints/my-model-2000"
    #restore previous session- either last session or specified session

saver = tf.train.Saver()
    #define saver

restored = saver.restore(sess, save_path)
    #define restored session

merged_summaries = tf.merge_all_summaries()
    #merge summuries for tensorboard

position = 0
    #define starting position at dataset as 0
correct_samples = 0.0
    #define number of correct samples as 0
for i in range(int(len(valid_dataset)/batch_size)):
    #for each batch in the number of batches that can fit in the dataset
    dataset = valid_dataset
        #define dataset to be used as validation dataset
    position += batch_size
        #increase position in dataset by batch size
    batch_data, batch_labels = SharedFunctions.create_batch(dataset, position, image_size, filenames_and_labels)
        #call create batch function, return batch data and labels
    print("step %d"%(i))
        #print step number
    loss, acc, summaries = sess.run([cross_entropy, accuracy, merged_summaries], feed_dict={tf_valid_dataset: batch_data, tf_valid_labels: batch_labels, keep_prob: 1.0})
        #run session with placeholder variables, get values
    print("loss %f, accuracy %f"%(np.average(loss), acc))
        #print loss and accuracy
    train_writer.add_summary(summaries, i)
        #add summary to graph
    correct_samples += acc*len(batch_data)
        #add to number of correct samples
print("validation accuracy:", correct_samples/(int(len(valid_dataset))))
    #calculate percentage of correct samples
    
    
