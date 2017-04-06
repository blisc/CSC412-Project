import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


class DataDistribution:
  def __init__(self):
    self.mnist = input_data.read_data_sets('data/', one_hot=True)
  
  def sample(self, numSamples):
    samples,labels = self.mnist.train.next_batch(numSamples)
    # samples.sort()
    return samples, labels

def linear(tensor_in, output_dim, scope = None, bias=0.0, summary=True, normal=False):
  with tf.variable_scope(scope or 'linear'):
    if normal:
      weights = tf.Variable(tf.truncated_normal([int(tensor_in.get_shape()[-1]), output_dim], stddev=0.001))
    else:
      weights = tf.get_variable("weights", [tensor_in.get_shape()[-1], output_dim], initializer=tf.contrib.layers.xavier_initializer(uniform=False))
    bias = tf.get_variable("biases", [output_dim], initializer=tf.constant_initializer(bias))
    
    if(summary):
      variable_summaries(weights, (scope or 'linear')+'_weights')
    
    return tf.matmul(tensor_in, weights) + bias

def optimizer(loss, learningRate, var_list=None):
  if var_list:
    optimizer = tf.train.AdamOptimizer(learningRate).minimize(loss, var_list = var_list)
  else:
    optimizer = tf.train.AdamOptimizer(learningRate).minimize(loss)
  return optimizer

def variable_summaries(var, name):
  """Attach a lot of summaries to a Tensor."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean/' + name, mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev/' + name, stddev)
    tf.summary.scalar('max/' + name, tf.reduce_max(var))
    tf.summary.scalar('min/' + name, tf.reduce_min(var))
    tf.summary.histogram(name, var)

def plot(data, trainStep, number, append=False):
  height, width = 28, 28 #in pixels
  spines = 'left', 'right', 'top', 'bottom'

  labels = ['label' + spine for spine in spines]

  tick_params = {spine : False for spine in spines}
  tick_params.update({label : False for label in labels})

  img = np.reshape(data, (-1, 28))

  img *= 255.

  desired_width = 4 #in inches
  scale = desired_width / float(width)

  fig, ax = plt.subplots(1, 1, figsize=(desired_width, height*scale))
  img = ax.imshow(img, cmap=cm.Greys_r, interpolation='none')

  #remove spines
  for spine in spines:
      ax.spines[spine].set_visible(False)

  #hide ticks and labels
  ax.tick_params(**tick_params)

  if append:
    name = './results/Pictures/Output_Step{}_{}_{}.png'.format(trainStep,number,append)
  else:
    name = './results/Pictures/Output_Step{}_{}.png'.format(trainStep,number)
  #save
  fig.savefig(name, dpi=300)

  plt.close(fig)

def plotMany(data, name, rows=10, cols=10):
  fig, ax = plt.subplots(rows,cols)
  for i in range(rows):
    for j in range(cols):
      img = data[i*cols+j]
      img = np.reshape(img, (-1, 28))
      img *= 255.
      ax[i,j].imshow(img, cmap=cm.Greys_r, interpolation='none')
      ax[i,j].axes.get_xaxis().set_visible(False)
      ax[i,j].axes.get_yaxis().set_visible(False)

  #save
  fig.savefig('./{}.png'.format(name), dpi=300)

  plt.close(fig)

def save_images(images, filename, **kwargs):
    fig = plt.figure(1)
    fig.clf()
    ax = fig.add_subplot(111)
    plot_images(images, ax, **kwargs)
    fig.patch.set_visible(False)
    ax.patch.set_visible(False)
    plt.savefig(filename)
    
def plot_comparison(data, generated_sample, trainStep, number, append=False, vmin=0, vmax=1):
  fig, (ax1, ax2) = plt.subplots(nrows=2)
  
  data = np.reshape(data, (-1, 28))
  generated_sample = np.reshape(generated_sample, (-1, 28))
  
  if vmin is None:
    vmin = min(np.min(data), np.min(generated_sample))
  if vmax is None:
    vmax = max(np.max(data), np.max(generated_sample))
  
  colour1 = ax1.imshow(data, cmap='gray', interpolation='nearest', aspect='auto', vmin=vmin, vmax=vmax)
  colour2 = ax2.imshow(generated_sample, cmap='gray', interpolation='nearest', aspect='auto', vmin=vmin, vmax=vmax)
  
  ax1.set_title('training data')
  
  ax2.set_title('generated data')
    
  fig.subplots_adjust(right=0.8)
  cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
  fig.colorbar(colour1, cax=cbar_ax)
  # fig.colorbar(colour2, ax=ax2)


  if append:
    name = './results/Pictures/Output_Step{}_{}_{}.png'.format(trainStep,number,append)
  else:
    name = './results/Pictures/Output_Step{}_{}.png'.format(trainStep,number)
  #save
  fig.savefig(name, dpi=300)

  plt.close(fig)