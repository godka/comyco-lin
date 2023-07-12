import numpy as np
import tensorflow.compat.v1 as tf
import tflearn

FEATURE_NUM = 128
ACTION_EPS = 1e-4

class Network():
    def CreateNetwork(self, inputs):
        with tf.variable_scope('actor'):
            split_0 = tflearn.fully_connected(inputs[:, 0:1, -1], FEATURE_NUM, activation='relu')
            split_1 = tflearn.fully_connected(inputs[:, 1:2, -1], FEATURE_NUM, activation='relu')
            split_2 = tflearn.conv_1d(inputs[:, 2:3, :], FEATURE_NUM, 1, activation='relu')
            split_3 = tflearn.conv_1d(inputs[:, 3:4, :], FEATURE_NUM, 1, activation='relu')
            split_4 = tflearn.conv_1d(inputs[:, 4:5, :self.a_dim], FEATURE_NUM, 1, activation='relu')
            split_5 = tflearn.fully_connected(inputs[:, 5:6, -1], FEATURE_NUM, activation='relu')

            split_2_flat = tflearn.flatten(split_2)
            split_3_flat = tflearn.flatten(split_3)
            split_4_flat = tflearn.flatten(split_4)

            merge_net = tflearn.merge(
                [split_0, split_1, split_2_flat, split_3_flat, split_4_flat, split_5], 'concat')

            pi_net = tflearn.fully_connected(merge_net, FEATURE_NUM, activation='relu')
            pi = tflearn.fully_connected(pi_net, self.a_dim, activation='softmax')
        return pi
            
    def get_network_params(self):
        return self.sess.run(self.network_params)

    def set_network_params(self, input_network_params):
        self.sess.run(self.set_network_params_op, feed_dict={
            i: d for i, d in zip(self.input_network_params, input_network_params)
        })

    def load_model(self, nn_model):
        saver = tf.train.Saver(max_to_keep=1000)  # save neural net parameters
        if nn_model is not None:  # nn_model is the path to file
            saver.restore(self.sess, nn_model)
            print("Model restored.")
            
    def __init__(self, sess, state_dim, action_dim, learning_rate):
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.H_target = 0.1
        
        self.lr_rate = learning_rate
        self.sess = sess
        self.inputs = tf.placeholder(tf.float32, [None, self.s_dim[0], self.s_dim[1]])

        self.acts = tf.placeholder(tf.float32, [None, self.a_dim])
        self.pi = self.CreateNetwork(inputs=self.inputs)
        self.real_out = tf.clip_by_value(self.pi, ACTION_EPS, 1. - ACTION_EPS)
        self.entropy = -tf.reduce_sum(tf.multiply(self.real_out, tf.log(self.real_out)), reduction_indices=1, keepdims=True)
        
        # Get all network parameters
        self.network_params = \
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor')

        # Set all network parameters
        self.input_network_params = []
        for param in self.network_params:
            self.input_network_params.append(
                tf.placeholder(tf.float32, shape=param.get_shape()))
        self.set_network_params_op = []
        for idx, param in enumerate(self.input_network_params):
            self.set_network_params_op.append(
                self.network_params[idx].assign(param))
        
        self.loss = tflearn.categorical_crossentropy(self.real_out, self.acts) \
            + 0.2 * tf.reduce_mean(self.entropy)
        self.opt = tf.train.AdamOptimizer(self.lr_rate).minimize(self.loss)

        self.sess.run(tf.global_variables_initializer())

    def predict(self, input):
        input_ = np.reshape(input, [-1, self.s_dim[0], self.s_dim[1]])
        action = self.sess.run(self.real_out, feed_dict={
            self.inputs: input_
        })
        return action[0]
    
    def train(self, s_batch, a_batch):
        self.sess.run([self.real_out, self.opt], feed_dict={
            self.inputs: s_batch,
            self.acts: a_batch
        })

    def save_model(self, nn_model='model'):
        saver = tf.train.Saver(max_to_keep=1000)  # save neural net parameters
        if nn_model is not None:  # nn_model is the path to file
            saver.save(self.sess, nn_model)
            print("Model saved.")
