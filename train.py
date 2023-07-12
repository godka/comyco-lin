import multiprocessing as mp
import numpy as np
import logging
import os
import sys
from env import ABREnv
import il as network
import tensorflow.compat.v1 as tf
import pool
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

S_DIM = [6, 8]
A_DIM = 6
ACTOR_LR_RATE = 1e-4
TRAIN_SEQ_LEN = 1000  # take as a train batch
TRAIN_EPOCH = 500
MODEL_SAVE_INTERVAL = 10
RANDOM_SEED = 42
SUMMARY_DIR = './comyco'
MODEL_DIR = './models'
TRAIN_TRACES = './train/'
TEST_LOG_FOLDER = './test_results/'
LOG_FILE = SUMMARY_DIR + '/log'

# create result directory
if not os.path.exists(SUMMARY_DIR):
    os.makedirs(SUMMARY_DIR)

NN_MODEL = None    

def testing(epoch, nn_model, log_file):
    # clean up the test results folder
    os.system('rm -r ' + TEST_LOG_FOLDER)
    #os.system('mkdir ' + TEST_LOG_FOLDER)

    if not os.path.exists(TEST_LOG_FOLDER):
        os.makedirs(TEST_LOG_FOLDER)
    # run test script
    os.system('python test.py ' + nn_model)

    # append test performance to the log
    rewards, entropies = [], []
    test_log_files = os.listdir(TEST_LOG_FOLDER)
    for test_log_file in test_log_files:
        reward, entropy = [], []
        with open(TEST_LOG_FOLDER + test_log_file, 'rb') as f:
            for line in f:
                parse = line.split()
                try:
                    entropy.append(float(parse[-2]))
                    reward.append(float(parse[-1]))
                except IndexError:
                    break
        rewards.append(np.mean(reward[1:]))
        entropies.append(np.mean(entropy[1:]))

    rewards = np.array(rewards)

    rewards_min = np.min(rewards)
    rewards_5per = np.percentile(rewards, 5)
    rewards_mean = np.mean(rewards)
    rewards_median = np.percentile(rewards, 50)
    rewards_95per = np.percentile(rewards, 95)
    rewards_max = np.max(rewards)

    log_file.write(str(epoch) + '\t' +
                   str(rewards_min) + '\t' +
                   str(rewards_5per) + '\t' +
                   str(rewards_mean) + '\t' +
                   str(rewards_median) + '\t' +
                   str(rewards_95per) + '\t' +
                   str(rewards_max) + '\n')
    log_file.flush()

    return rewards_mean, np.mean(entropies)
        
def main():
    env = ABREnv(random_seed=42)
    with tf.Session() as sess, open(LOG_FILE + '_test.txt', 'w') as test_log_file:
        summary_ops, summary_vars = build_summaries()

        actor = network.Network(sess,
                                state_dim=S_DIM, action_dim=A_DIM,
                                learning_rate=ACTOR_LR_RATE)

        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)  # training monitor
        saver = tf.train.Saver(max_to_keep=10000)  # save neural net parameters

        # restore neural net parameters
        nn_model = NN_MODEL
        if nn_model is not None:  # nn_model is the path to file
            saver.restore(sess, nn_model)
            print("Model restored.")
        
        actor_pool = pool.pool()
        for epoch in tqdm(range(TRAIN_EPOCH + 1)):
            obs = env.reset()
            last_bit_rate = 1
            for step in range(TRAIN_SEQ_LEN):
                action_prob = actor.predict(obs)

                # gumbel noise
                noise = np.random.gumbel(size=len(action_prob))
                bit_rate = np.argmax(np.log(action_prob) + noise)

                opt_bit_rate = env.net_env.get_optimal(last_bit_rate, 5000, 15)
                action_vec = np.zeros(A_DIM)
                action_vec[opt_bit_rate] = 1
                actor_pool.submit(obs, action_vec)
                
                s_batch, a_batch = actor_pool.get()
                actor.train(s_batch, a_batch)

                obs, rew, done, info = env.step(bit_rate)
                last_bit_rate = bit_rate

                if done:
                    break

            if epoch % MODEL_SAVE_INTERVAL == 0:
                # Save the neural net parameters to disk.
                save_path = saver.save(sess, SUMMARY_DIR + "/nn_model_ep_" +
                                       str(epoch) + ".ckpt")
                avg_reward, avg_entropy = testing(epoch,
                    SUMMARY_DIR + "/nn_model_ep_" + str(epoch) + ".ckpt", 
                    test_log_file)

                summary_str = sess.run(summary_ops, feed_dict={
                    summary_vars[0]: avg_reward,
                    summary_vars[1]: avg_entropy
                })
                writer.add_summary(summary_str, epoch)
                writer.flush()

def build_summaries():
    eps_total_reward = tf.Variable(0.)
    tf.summary.scalar("Reward", eps_total_reward)
    entropy = tf.Variable(0.)
    tf.summary.scalar("Entropy", entropy)

    summary_vars = [eps_total_reward, entropy]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars

if __name__ == '__main__':
    main()
