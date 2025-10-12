import os
import shutil
import subprocess
import sys
from typing import Tuple

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from env import ABREnv
import il as network
import pool
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

S_DIM = [6, 8]
A_DIM = 6
ACTOR_LR_RATE = 1e-4
TRAIN_SEQ_LEN = 1000  # take as a train batch
TRAIN_EPOCH = 500
MODEL_SAVE_INTERVAL = 10
RANDOM_SEED = 42
SUMMARY_DIR = './comyco'
TEST_LOG_FOLDER = './test_results/'
LOG_FILE = os.path.join(SUMMARY_DIR, 'log_test.txt')
MODEL_PREFIX = 'nn_model_ep_'
MODEL_SUFFIX = '.pth'

# create result directory
os.makedirs(SUMMARY_DIR, exist_ok=True)

NN_MODEL = None


def testing(epoch: int, nn_model: str, log_file) -> Tuple[float, float]:
    # clean up the test results folder
    if os.path.isdir(TEST_LOG_FOLDER):
        shutil.rmtree(TEST_LOG_FOLDER)
    os.makedirs(TEST_LOG_FOLDER, exist_ok=True)

    # run test script
    subprocess.run([sys.executable, 'test.py', nn_model], check=True)

    # append test performance to the log
    rewards, entropies = [], []
    test_log_files = os.listdir(TEST_LOG_FOLDER)
    for test_log_file in test_log_files:
        reward, entropy = [], []
        with open(os.path.join(TEST_LOG_FOLDER, test_log_file), 'rb') as f:
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
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    env = ABREnv(random_seed=RANDOM_SEED)
    actor = network.Network(state_dim=S_DIM, action_dim=A_DIM, learning_rate=ACTOR_LR_RATE)

    if NN_MODEL is not None:
        actor.load_model(NN_MODEL)
        print("Model restored.")

    writer = SummaryWriter(SUMMARY_DIR)
    actor_pool = pool.pool()

    with open(LOG_FILE, 'w') as test_log_file:
        for epoch in tqdm(range(TRAIN_EPOCH + 1)):
            obs = env.reset()
            last_bit_rate = 1
            for step in range(TRAIN_SEQ_LEN):
                action_prob = actor.predict(obs)

                noise = np.random.gumbel(size=len(action_prob))
                bit_rate = int(np.argmax(np.log(action_prob) + noise))

                opt_bit_rate = env.net_env.get_optimal(last_bit_rate, 5000, 15)
                action_vec = np.zeros(A_DIM)
                action_vec[opt_bit_rate] = 1
                actor_pool.submit(obs, action_vec)

                s_batch, a_batch = actor_pool.get()
                actor.train_step(s_batch, a_batch)

                obs, rew, done, info = env.step(bit_rate)
                last_bit_rate = bit_rate

                if done:
                    break

            if epoch % MODEL_SAVE_INTERVAL == 0:
                save_path = os.path.join(SUMMARY_DIR, f"{MODEL_PREFIX}{epoch}{MODEL_SUFFIX}")
                actor.save_model(save_path)
                avg_reward, avg_entropy = testing(epoch, save_path, test_log_file)

                writer.add_scalar("Reward", avg_reward, epoch)
                writer.add_scalar("Entropy", avg_entropy, epoch)
                writer.flush()

    writer.close()


if __name__ == '__main__':
    main()
