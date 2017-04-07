import os
import numpy as np
import tensorflow as tf

from BuildingBlocks import DataDistribution, linear, optimizer, plot_comparison

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('checkpoint_dir', 'results/', 'Directory for storing results')

class VariationalAutoencoder:
  '''Defines a VAE class that creates the model'''
  def __init__(self, batchSize, hiddenLayerSize, trainingEpochs, learningRate, latentDimension):
    self.batchSize = batchSize
    self.hiddenLayerSize = hiddenLayerSize
    self.trainingEpochs = trainingEpochs
    self.learningRate = learningRate
    self.latentDimension = latentDimension
    
    # Loads MNIST dataset
    self.dataSamples = DataDistribution()
    self.trainingSize = 60000
    
  def createModel(self):
    print("Building Model")
    self.images = tf.placeholder(tf.float32, [self.batchSize, 784])
    
    with tf.variable_scope("Recognition") as scope:
      # Log sigma squared version
      # self.z_mean, self.z_log_var = self._recognition_network(self.images)
      # epsilon = tf.random_normal((self.batchSize, self.latentDimension), 0, 1, dtype=tf.float32)
      # self.z = tf.add(self.z_mean, tf.mul(tf.sqrt(tf.exp(self.z_log_var)), epsilon))
      # Log sigma version
      self.z_mean, self.z_log_std = self._recognition_network(self.images)
      epsilon = tf.random_normal((self.batchSize, self.latentDimension), 0, 1, dtype=tf.float32)
      self.z = tf.add(self.z_mean, tf.mul(tf.exp(self.z_log_std), epsilon))
    
    with tf.variable_scope("Generator") as scope:
      self.reconstructrion = self._generator_network(self.z)
      
    # loss is -KL(q(z|x)||p(z)) + mean(log(p(x|z)))
    # KL + reconstructrion loss
    # q(z|x) is self.z
    self.reconstructrion_loss = -tf.reduce_sum(self.images * tf.log(1e-10 + self.reconstructrion)
                           + (1-self.images) * tf.log(1e-10 + 1. - self.reconstructrion),
                           1)
    # Log sigma squared version
    # self.latent_loss = -0.5 * tf.reduce_sum(1. + self.z_log_var 
                                           # - tf.square(self.z_mean) 
                                           # - tf.exp(self.z_log_var), 1)
    # Log sigma version
    self.latent_loss = -0.5 * tf.reduce_sum(1. + 2.*self.z_log_std
                                           - tf.square(self.z_mean) 
                                           - tf.exp(2.*self.z_log_std), 1)                       
    self.loss = tf.reduce_mean(self.reconstructrion_loss+self.latent_loss)
    
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
          _, loss, = sess.run([self.optimizer, self.loss], feed_dict={self.images:batch})
          # r_loss = np.mean(r_loss)
          # l_loss = np.mean(l_loss)
          # print("reconst loss {}".format(r_loss))
          # print("latent loss {}".format(l_loss))
          avg_loss += loss / self.trainingSize * self.batchSize

        if epoch % 5 == 0:
          print("Epoch:", '%04d' % (epoch+1), "loss=", "{:.9f}".format(avg_loss))
          # print("last loss=", "{:.9f}".format(loss))
          summary_str = sess.run(summary, feed_dict={self.images:batch})
          summary_writer.add_summary(summary_str, epoch)
          summary_writer.flush()
        if epoch % 10 == 0 or (epoch+1) == self.trainingEpochs:
          checkpoint_file = os.path.join(FLAGS.checkpoint_dir, 'checkpoint')
          saver.save(sess, checkpoint_file, global_step=epoch)
          
          #Plot reconstructrion
          output = sess.run(self.reconstructrion, feed_dict={self.images:batch})
          plot_comparison(batch[0], output[0], epoch, 1)
          plot_comparison(batch[50], output[50], epoch, 2)
          plot_comparison(batch[100], output[100], epoch, 3)
    
  def _recognition_network(self, data):
    hid_1 = tf.nn.relu(linear(data,self.hiddenLayerSize,'hid_1'))
    hid_2 = tf.nn.relu(linear(hid_1,self.hiddenLayerSize,'hid_2'))
    
    z_mean = tf.nn.relu(linear(hid_2,self.latentDimension,'z_mean'))
    z_log_std = tf.nn.relu(linear(hid_2,self.latentDimension,'z_log_std'))
    
    return (z_mean, z_log_std)
    
  def _generator_network(self, sample):
    hid_1 = tf.nn.relu(linear(sample,self.hiddenLayerSize,'hid_1'))
    hid_2 = tf.nn.relu(linear(hid_1,self.hiddenLayerSize,'hid_2'))
    
    reconstructrion = tf.nn.sigmoid(linear(hid_2,784,'z_mean'))
    
    return reconstructrion
    
if __name__ == '__main__':
  with tf.device('/gpu'):
    model = VariationalAutoencoder(batchSize=128, hiddenLayerSize=500, trainingEpochs=100, learningRate=0.001, latentDimension=20)
    model.createModel()
    model.train()