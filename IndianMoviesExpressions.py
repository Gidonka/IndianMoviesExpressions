###COMMENTS GUIDE:
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
    #import modules

random.seed(42)


###get data, organize in dictionary
files  = glob.glob("/Users/Gidonka/Documents/Programming/NYU/MachineLearning/IndianMovies/IMFDB_Final/*/*/*/*.jpg", recursive=True)
    #files = filenames of image files 
labels = glob.glob("/Users/Gidonka/Documents/Programming/NYU/MachineLearning/IndianMovies/IMFDB_Final/*/*/*.txt", recursive=True)
    #labels = filenames of text files containing labels

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
            if not os.path.isfile(filepath):
                problem_file += 1
                continue
                #get filepath for image
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



print("Total dataset size:", len(filenames_and_labels.keys()))
print("Training dataset size:", len(train_dataset))
print("Validation dataset size:", len(valid_dataset))
print("Testing dataset size:", len(test_dataset))
    #print size of datasets


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

random.shuffle(train_dataset)
    #shuffle train dataset

one_hot_dict = {"ANGER": [1, 0, 0, 0, 0, 0, 0], "HAPPINESS": [0, 1, 0, 0, 0, 0, 0], "SADNESS": [0, 0, 1, 0, 0, 0, 0], "SURPRISE": [0, 0, 0, 1, 0, 0, 0], "FEAR": [0, 0, 0, 0, 1, 0, 0], "DISGUST": [0, 0, 0, 0, 0, 1, 0], "NEUTRAL": [0, 0, 0, 0, 0, 0, 1]}
        #create dictionary for assigning one hot encoding to labels

def create_batch(train_dataset, position_at_dataset, image_size, filenames_and_labels):
    #define function for batch creation
    batch_of_filepaths = train_dataset[position_at_dataset:(position_at_dataset+128)]
        #define breadth of batch from dataset
    #position_at_dataset = position_at_dataset+128
        #define how far into dataset batches have already gone
        #define variable for counting problematic images
    batch_image_array = np.empty([batch_size, image_size, image_size, 3])
        #define empty numpy array to be filled with image arrays. Each row is an image. Dimensions: batch_size (128) x image_size*image_size*channels (2352)
    batch_label_array = np.zeros([batch_size, num_labels])
        #define empty numpy array to be filled with one hot encoded labels. Each row is a label. Dimensions: batch_size (128) x num_labels (7)
    for i, image in enumerate(batch_of_filepaths):
        #create loop to enumerate through the batch of filepaths
        face = io.imread(image)
                #read each image
        final_face = square(face, image_size)
                #call sqaure function on image
            #flattened_face = final_face.flatten()
                #flatten 3 dimensional image (dim, dim, channels) into 1 dimension
        flattened_face = final_face
                #not actually flattening image because conv net needs it in 4D not 2D. 
        batch_image_array[i,:] = flattened_face
                #fill each row of image array with an image array
            
        label = filenames_and_labels[image]
                #get matching label for each image using dictionary
        one_hot_label = one_hot_dict[label]
                #get one hot encoding for labels from one_hot_dict
        batch_label_array[i,:] = one_hot_label
                #fill each row of label array with a one hot encoded array
                #add 1 to exception counter
                #skip image
        #print number of skipped images
    #print("Batch dataset array shape:", batch_image_array.shape)
    #print("Batch labels array shape:", batch_label_array.shape)
    #print(batch_image_array.dtype)
    #print("batch data min: %f, batch data max: %f"%(batch_image_array.min(), batch_image_array.max()))
    batch_image_array -= 0.5
    #print("batch data min: %f, batch data max: %f"%(batch_image_array.min(), batch_image_array.max()))
    return batch_image_array, batch_label_array
position_at_dataset = random.randint(0, len(train_dataset)-128)
batch_data, batch_labels = create_batch(train_dataset, position_at_dataset, image_size, filenames_and_labels)
    ##temporary function call- should be called within network init

"""

def valid_create(valid_dataset, image_size, filenames_and_labels):
    #define function for validation dataset creation
    problem_image = 0
    valid_image_array = np.empty([va_amount, image_size*image_size*3])
    valid_label_array = np.zeros([va_amount, num_labels])
    for i, image in enumerate(valid_dataset):
        try:
            face = io.imread(image)
            final_face = square(face, image_size)
            flattened_face = final_face.flatten()
            valid_image_array[i,:] = flattened_face

            label = filenames_and_labels[image]
            one_hot_label = one_hot_dict[label]
            valid_label_array[i,:] = one_hot_label
        except FileNotFoundError:
            problem_image = problem_image+1
            continue
    print("Valid dataset problematic images:", problem_image)
    print("Valid dataset array shape:", valid_image_array.shape)
    print("Valid labels array shape:", valid_label_array.shape)
    return valid_image_array, valid_label_array
#when to initialize function?
#what to pass returned arrays to?
valid_create(valid_dataset, image_size, filenames_and_labels)


def test_create(test_dataset, image_size, filenames_and_labels):
    problem_image = 0
    test_image_array = np.empty([te_amount, image_size*image_size*3])
    test_label_array = np.zeros([te_amount, num_labels])
    for i, image in enumerate(test_dataset):
        try:
            face = io.imread(image)
            final_face = square(face, image_size)
            flattened_face = final_face.flatten()
            test_image_array[i,:] = flattened_face

            label = filenames_and_labels[image]
            one_hot_label = one_hot_dict[label]
            test_label_array[i,:] = one_hot_label
        except FileNotFoundError:
            problem_image = problem_image+1
            continue
    print("Test dataset problematic images:", problem_image)
    print("Test dataset array shape:", test_image_array.shape)
    print("Test labels array shape:", test_labels_array.shape)
    return test_image_array, test_label_array
#when to initialize function?
#what to pass returned arrays to?
test_create(test_dataset, image_size, filenames_and_labels)
"""

#HIC SVNT DRACONES

"""
######Network V1######
###network architecture
graph = tf.Graph()
with graph.as_default():
    
    
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size*image_size*3))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    ##tf_valid_dataset = tf.constant(valid_image_array)
    ##tf_test_dataset = tf.constant(test_image_array)
        #create placeholders for input data

    
    ###hidden layer
    hidden_layer_size = 1024
        #define hidden layer size
    weights_h = tf.Variable(tf.truncated_normal([image_size * image_size * 3, hidden_layer_size]))
        #define weights for hidden layer
    biases_h  = tf.Variable(tf.zeros([hidden_layer_size]))
        #define biases for hidden layer
    hidden = tf.nn.relu(tf.matmul(tf_train_dataset, weights_h) + biases_h)
        #define how everything goes together in the ReLU function in the hidden layer

    ###output layer
    weights_o = tf.Variable(tf.truncated_normal([hidden_layer_size, num_labels]))
        #define weights for output layer
    biases_o = tf.Variable(tf.zeros([num_labels]))
        #define biases for output layer
    logits = tf.matmul(hidden, weights_o) + biases_o
        #define logits
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
        #define loss

    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
        #optimizer

    train_prediction = tf.nn.softmax(logits)
    ##tf_valid_dataset   valid_prediction = tf.nn.softmax(tf.matmul(tf.nn.relu(tf.matmul(tf_valid_dataset, weights_h) + biases_h), weights_o) + biases_o)
    ##test_prediction = tf.nn.softmax(tf.matmul(tf.nn.relu(tf.matmul(tf_test_dataset, weights_h) + biases_h), weights_o) + biases_o)
        #predictions for training, validation, and test data

num_steps = 100
    #define number of steps to be taken

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])
                #get accuracy

###network initialization
with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print("Initialized")
      #initialize TensorFlow graph with all variables which were set earlier
  for step in range(num_steps):
    position_at_dataset = random.randint(0, len(train_dataset)-128)
        #define how position at dataset progresses with every batch taken
    batch_data, batch_labels = create_batch(train_dataset, position_at_dataset, image_size, filenames_and_labels)
        #make batch data and batch labels equal to the returned array of func create_batch
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.
    print(l)
    ##if (step % 500 == 0):
    print("Minibatch loss at step %d: %f" % (step, l))
    print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      ##sprint("Validation accuracy: %.1f%%" % accuracy(
        #valid_prediction.eval(), valid_labels))
  #print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
"""
#####Convolution Network####
#launch TF session
sess = tf.InteractiveSession()

#create placeholder for answers
x = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, 3))
#create placeholder for correct answers
y_ = tf.placeholder(tf.float32, shape=(batch_size, num_labels))

#function for creating many weights
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)
#function for creating many biasies
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

#function for convolving
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
#function for pooling
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

###FIRST CONVOLUTION LAYER

#create weight tensor ([dimension, dimension, number of inputs, number of outputs])
W_conv1 = weight_variable([5, 5, 3, 32])
#create bias tensor ([number of outputs])
b_conv1 = bias_variable([32])

#reshape x as 4d tensor
#x_image = tf.reshape(x, [-1,28,28,1])

#convolve x with weight, apply ReLU function, add bias
h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
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

loss_results = open('LossResults', 'w')
accuracy_results = open('AccuracyResults', 'w')

#train and evaluate: define cross entropy, minimize cross entropy, define ADAM optimizer as preferred method to do so
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
cross_entropy_array = tf.nn.softmax_cross_entropy_with_logits(y_conv, y_)
cross_entropy = tf.reduce_mean(cross_entropy_array)
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.scalar_summary('accuracy', accuracy)
tf.scalar_summary('loss', cross_entropy)
    #initialize SummaryWriter
sess.run(tf.initialize_all_variables())


train_writer = tf.train.SummaryWriter('./summary', sess.graph)

merged_summaries = tf.merge_all_summaries()
#train network: repeat n times, evaluate every 100 steps, control dropout rate
for i in range(50000):
  position_at_dataset = random.randint(0, len(train_dataset)-128)
  #position_at_dataset = 0
  print("step %d"%(i))
  batch_data, batch_labels = create_batch(train_dataset, position_at_dataset, image_size, filenames_and_labels)
  #if i%100 == 0:
  #train_accuracy = accuracy.eval(feed_dict={x:batch_data, y_: batch_labels, keep_prob: 1.0})
  #print("training accuracy %f"%(100*train_accuracy))
  sess.run(train_step, feed_dict={x: batch_data, y_: batch_labels, keep_prob: 0.5})
  if i%50 == 0:
      loss, acc, summaries = sess.run([cross_entropy, accuracy, merged_summaries], feed_dict={x: batch_data, y_: batch_labels, keep_prob: 0.5})
      print("loss %f, accuracy %f"%(np.average(loss), acc))
      train_writer.add_summary(summaries, i)
  #loss_results.write(str(np.average(loss)))
  #loss_results.write(",")
  #accuracy_results.write(str(train_accuracy))
  #accuracy_results.write(",")
####for i in range(0, len(valid_dataset))...

#print results
##print("test accuracy %g"%accuracy.eval(feed_dict={
##    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

#TODO:
#change batch creation function to simultaniously create validation batch
#integrate validation accuracy check


#VALIDATION:
      #create valid batches with batch creation function
          #change batch_size to none
          #return valid_image_array and valid_label_array
      #initialize batch creation function
          #start at 0, create batches incrementally
          #create loop to run network and test on valid dataset
          #log accuracy results to tensorboard
          

