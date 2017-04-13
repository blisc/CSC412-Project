import os
import numpy as np
import tensorflow as tf

from BuildingBlocks import DataDistribution, linear, optimizer, plot_comparison, \
                           NormFlowLayer, lognormal, log_stdnormal

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('checkpoint_dir', 'results/', 'Directory for storing results')


class VariationalAutoencoder:
  '''Defines a VAE class that creates the model'''
  def __init__(self, batchSize, hiddenLayerSize, trainingEpochs, learningRate, latentDimension, flowLayers):
    self.batchSize = batchSize
    self.hiddenLayerSize = hiddenLayerSize
    self.trainingEpochs = trainingEpochs
    self.learningRate = learningRate
    self.latentDimension = latentDimension
    self.flowLayers = flowLayers # Length of normalizing flow
    
    # Loads MNIST dataset
    self.dataSamples = DataDistribution()
    self.trainingSize = 60000


  def createModel(self):
    print("Building Model")
    self.images = tf.placeholder(tf.float32, [self.batchSize, 784])
    
    with tf.variable_scope("Recognition") as scope:
      self.z_mean, self.z_log_var = self._recognition_network(self.images)
      epsilon = tf.random_normal((self.batchSize, self.latentDimension), 0, 1, dtype=tf.float32)

      # self.z are samples from q(z|x)
      # self.z shape = batch_size x latent_dimension
      self.z = tf.add(self.z_mean, tf.multiply(tf.sqrt(tf.exp(self.z_log_var)), epsilon))
    
    with tf.variable_scope("Generator") as scope:
      # self.reconstructon is the mean (i.e. output of VAE network, which is Bernoulli)
      self.reconstruction, self.sumLogDetJ, f_z = self._generator_network(self.z)

    # Compute the loss function
    self.log_q0_z0 = lognormal(self.z, self.z_mean, self.z_log_var)
    self.log_p_x_given_zk = tf.reduce_sum(self.images * tf.log(1e-10 + self.reconstruction) + \
                                     (1 - self.images) * tf.log(1e-10 + 1 - self.reconstruction), 1)
    self.log_p_x_given_zk = tf.reshape(self.log_p_x_given_zk, [128,1])
    self.log_p_zk = log_stdnormal(f_z)
    self.log_p_x_and_zk = self.log_p_x_given_zk + self.log_p_zk
    # self.sumLogDetJ = tf.reduce_sum(self.sumLogDetJ, axis=1)
    
    # print(self.log_q0_z0.get_shape())
    # print(self.log_p_x_given_zk.get_shape())
    # print(self.log_p_zk.get_shape())
    # print(self.log_p_x_and_zk.get_shape())
    # print(self.sumLogDetJ.get_shape())

    self.loss = -tf.reduce_mean(self.log_p_x_and_zk + self.sumLogDetJ - self.log_q0_z0)
    self.optimizer = optimizer(self.loss, self.learningRate)

    tf.summary.scalar(self.loss.op.name, self.loss)


  def train(self, restore=False):
    print("Starting training")
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)) as sess:
      summary = tf.summary.merge_all()
      summary_writer = tf.summary.FileWriter(FLAGS.checkpoint_dir, sess.graph)
      saver = tf.train.Saver(max_to_keep=2)

      startingEpoch = 0
      
      tf.global_variables_initializer().run()
      if restore:
        ckpt = tf.train.get_checkpoint_state(restore)
        if ckpt and ckpt.model_checkpoint_path:
          saver.restore(sess, ckpt.model_checkpoint_path)
          restore = tf.train.latest_checkpoint(restore)
          print("Restored :{}".format(restore))
          start_step = int(restore.split("-")[1])
          
        else:
          print("Saver error")
          return
          
      for epoch in range(startingEpoch, self.trainingEpochs):
        avg_loss = 0.
        total_batch = int(self.trainingSize / self.batchSize)
        for i in range(total_batch):
          batch, _ = self.dataSamples.sample(self.batchSize)
          _, loss = sess.run([self.optimizer, self.loss], feed_dict={self.images:batch})
          #DEBUG
          # log_q0_z0, log_p_x_given_zk, log_p_zk, sumLogDetJ = sess.run([self.log_q0_z0, self.log_p_x_given_zk, self.log_p_zk, self.sumLogDetJ], feed_dict={self.images:batch})
          # print("log_q0_z0: {}".format(log_q0_z0))
          # print("log_p_x_given_zk: {}".format(log_p_x_given_zk))
          # print("log_p_zk: {}".format(log_p_zk))
          # print("sumLogDetJ: {}".format(sumLogDetJ))
          avg_loss += loss / self.trainingSize * self.batchSize

        if epoch % 1 == 0:
          print("Epoch:", '%04d' % (epoch+1), "loss=", "{:.9f}".format(avg_loss))
          # print("last loss=", "{:.9f}".format(loss))
          summary_str = sess.run(summary, feed_dict={self.images:batch})
          summary_writer.add_summary(summary_str, epoch)
          summary_writer.flush()
        if epoch % 5 == 0 or (epoch+1) == self.trainingEpochs:
          checkpoint_file = os.path.join(FLAGS.checkpoint_dir, 'checkpoint')
          saver.save(sess, checkpoint_file, global_step=epoch)
          
          #Plot reconstruction
          output = sess.run(self.reconstruction, feed_dict={self.images:batch})
          plot_comparison(batch[0], output[0], epoch, 1)
          plot_comparison(batch[50], output[50], epoch, 2)
          plot_comparison(batch[100], output[100], epoch, 3)


  def _recognition_network(self, data):
    hid_1 = tf.nn.relu(linear(data,self.hiddenLayerSize,'hid_1'))
    hid_2 = tf.nn.relu(linear(hid_1,self.hiddenLayerSize,'hid_2'))
    
    #z_mean = tf.nn.relu(linear(hid_2,self.latentDimension,'z_mean'))
    #z_log_var = tf.nn.relu(linear(hid_2,self.latentDimension,'z_log_var'))
    z_mean = tf.nn.tanh(linear(hid_2,self.latentDimension,'z_mean'))
    z_log_var = tf.nn.tanh(linear(hid_2,self.latentDimension,'z_log_var'))
    
    return (z_mean, z_log_var)


  def _generator_network(self, sample):
    init_density_mean = self.z_mean
    init_density_std = self.z_log_var
    f_z = sample
    sumLogDetJ = None # shape = size of z x size of f = latent dim x num flow layers?
    # sumLogDetJ = []
    
    for i in range(0, self.flowLayers):
      # Set norm flow layer dimension to be same as latent dimension
      currScope = 'norm_flow_' + str(i+1)
      [f_z, logDetJ] = NormFlowLayer(f_z, self.latentDimension, scope=currScope)
      if sumLogDetJ is None:
        sumLogDetJ = logDetJ
      else:
        sumLogDetJ += logDetJ
        
      # sumLogDetJ.append(logDetJ)
      
    # print(len(sumLogDetJ))
    # print(sumLogDetJ[0].get_shape())
    # sumLogDetJ = tf.concat(1, sumLogDetJ)
    # print(sumLogDetJ.get_shape())
    # sumLogDetJ = tf.reduce_sum(sumLogDetJ

    hid_1 = tf.nn.relu(linear(f_z,self.hiddenLayerSize,'hid_1'))
    hid_2 = tf.nn.relu(linear(hid_1,self.hiddenLayerSize,'hid_2'))

    reconstruction = tf.nn.sigmoid(linear(hid_2,784,'z_mean'))

    return reconstruction, sumLogDetJ, f_z


if __name__ == '__main__':
  print("VAE Normalizing Flows")
  with tf.device('/gpu'):
    model = VariationalAutoencoder(batchSize=128, hiddenLayerSize=500, trainingEpochs=100, \
                                   learningRate=0.001, latentDimension=20, flowLayers=3)
    model.createModel()
    model.train()


