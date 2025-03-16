import os
import csv
import time
import math
import math
import numpy as np
import torch
import torch.multiprocessing as mp
import tensorflow as tf
# import logger
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
import redis
import json
import random
import subprocess
import sys
from collections import defaultdict
sys.path.append('/users/janechen/Genet/src')    
sys.path.append("/users/janechen/Genet/src/emulator/abr/pensieve")
sys.path.append("/users/janechen/Genet/src/emulator/abr/pensieve/agent_policy")
print("Pensieve sys path: ", sys.path)
from emulator.abr.pensieve import a3c
from emulator.abr.pensieve.utils import linear_reward
from emulator.abr.pensieve.a3c.a3c_jump import ActorNetwork as OriginalActorNetwork
from models import *
import rl_embedding
from rl_embedding import *
# from .models import create_mask

MODEL_SAVE_INTERVAL = 500
VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300]  # Kbps
# VIDEO_BIT_RATE = [300, 1200, 2850, 6500, 33000, 165000]
HD_REWARD = [1, 2, 3, 12, 15, 20]
M_IN_K = 1000.0
REBUF_PENALTY = 43  # 1 sec rebuffering -> 3 Mbps
# REBUF_PENALTY = 165  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 0  # default video quality without agent
BITRATE_DIM = 6
# DEVICE = 'cpu'
# EMBEDDING_SIZE = 16

BUCKET_BOUNDARIES = {
    1: [0.12, 0.2, 0.28, 0.43, 0.55, 0.83, 1.03, 1.29, 1.63, 2.12, 3.64, 4.02, 5.74, 8, 12, 14],
    2: [0.01, 0.3, 0.38, 0.44, 0.49, 0.54, 0.6, 0.68, 0.84, 1.41, 3, 5, 205, 395, 1206],
    3: [0.01, 0.08, 0.11, 0.15, 0.23, 0.45, 0.8, 0.9, 1, 1.75],
    4: [0.0002, 0.0047, 0.0361, 0.1, 0.2, 0.3],
    5: [0.75, 1, 1.001, 1.003, 1.012, 1.25, 3.52, 4.7, 5.39, 6.26]
}

# bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and
# time), chunk_til_video_end
# S_INFO = 6
# S_LEN = 6  # take how many frames in the past
A_DIM = 3
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
BUFFER_NORM_FACTOR = 10.0
RAND_RANGE = 1000
# log in format of time_stamp bit_rate buffer_size rebuffer_time chunk_size
# download_time reward
MILLISECONDS_IN_SECOND = 1000.0
B_IN_MB = 1000000.0
BITS_IN_BYTE = 8.0
VIDEO_CHUNK_LEN = 4000.0  # millisec, every time add this amount to buffer
PACKET_SIZE = 1500  # bytes
NOISE_LOW = 0.9
NOISE_HIGH = 1.1
TRAIN_SEQ_LEN = 100  # batchsize of pensieve training 100

# from pensieve.utils import compute_entropy

UP_LINK_SPEED_FILE="pensieve/data/12mbps"
VIDEO_SIZE_DIR="pensieve/data/video_sizes"

def setup_logger(logger_name, log_file, level=logging.INFO):
    """Create and return a logger with a file handler."""
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    # Avoid adding multiple handlers if logger already exists (happens in multiprocessing)
    if not logger.handlers:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(level)
        formatter = logging.Formatter(
            '%(asctime)s [%(name)s] %(levelname)s: %(message)s'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class Pensieve():
    """Pensieve Implementation.

    Args
        num_agents(int): number of processes to train pensieve models.
        log_dir(str): path where all log files and model checkpoints will be
            saved to.
        actor_path(None or str): path to a actor checkpoint to be loaded.
        critic_path(None or str): path to a critic checkpoint to be loaded.
        model_save_interval(int): the period of caching model checkpoints.
        batch_size(int): training batch size.
        randomization(str): If '', no domain randomization. All
            environment parameters will leave as default values. If 'udr',
            uniform domain randomization. If 'adr', active domain
            randomization.
    """

    def __init__(self, num_agents, log_dir, actor=None,
                 critic_path=None, model_save_interval=100, batch_size=100,
                 randomization='', randomization_interval=1, video_size_file_dir="", val_traces="", original_actor=None, adaptor_input=None, adaptor_hidden_layer=None):
        # https://github.com/pytorch/pytorch/issues/3966
        # mp.set_start_method("spawn")
        self.num_agents = num_agents


        self.net = actor
        self.original_actor = original_actor
        self.adaptor_input = adaptor_input
        self.adaptor_hidden_layer = adaptor_hidden_layer
        if self.adaptor_input == "original_action_prob":
            self.state_dim = 3 + EMBEDDING_SIZE
        elif self.adaptor_input == "original_selection":
            self.state_dim = 1 + EMBEDDING_SIZE
        elif self.adaptor_input == "original_bit_rate":
            self.state_dim = 1 + EMBEDDING_SIZE
        elif self.adaptor_input == "hidden_state":
            self.state_dim = HIDDEN_SIZE + EMBEDDING_SIZE
        else:
            self.state_dim = [S_INFO+EMBEDDING_SIZE, S_LEN] 
        # NOTE: this is required for the ``fork`` method to work
        # self.net.actor_network.share_memory()
        # self.net.critic_network.share_memory()

        #self.load_models(actor_path, critic_path)

        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.model_save_interval = model_save_interval
        self.epoch = 0  # track how many epochs the models have been trained
        self.batch_size = batch_size
        self.randomization = randomization
        self.randomization_interval = randomization_interval
        self.video_size_file_dir = video_size_file_dir
        self.val_traces = val_traces
        
        # Prepare a logger for the training
        self.train_logger = setup_logger(
            "trainLogger",
            "/mydata/logs/log_train"
        )

        # Prepare another logger for the "central_agent" function
        # (if you prefer a single file for central agent logs or a separate file altogether)
        self.central_logger = setup_logger(
            "centralLogger",
            "/mydata/logs/log_central"  # Or use os.path.join(log_dir, "log_central")
        )
        
        self.test_logger = setup_logger(
            "testLogger",
            "/mydata/logs/log_test"
        )


    def train(self, train_envs, save_dir, iters=1e5, use_replay_buffer=False, original_actor_path=None):
        """
        train_envs: list of env configs, each is a dict like
            {"trace_file": "path/to/trace", "delay": 20}
        """
        
        self.train_logger.info("Starting Pensieve.train() with %d agents.", self.num_agents)

        # Visdom Settings
        # vis = visdom.Visdom()
        # assert vis.check_connection()
        plot_color = 'red'
        # Visdom Logs
        val_epochs = []
        val_mean_rewards = []
        average_rewards = []
        average_entropies = []

        # inter-process communication queues
        net_params_queues = []
        exp_queues = []
        for i in range(self.num_agents):
            net_params_queues.append(mp.Queue(1))
            exp_queues.append(mp.Queue(1))

        # agent(agent_id, net_params_queue, exp_queue, train_envs,
        #   summary_dir, batch_size, randomization, randomization_interval,
        #   num_agents)
        agents = []
        for i in range(self.num_agents):
            agents.append(mp.Process(
                target=agent,
                args=(
                    i,
                    net_params_queues[i],
                    exp_queues[i],
                    train_envs,
                    self.log_dir,
                    self.batch_size,
                    self.randomization,
                    self.randomization_interval,
                    self.num_agents,
                    original_actor_path,
                    self.adaptor_input,
                    self.adaptor_hidden_layer,
                    self.state_dim
                )
            ))
            # agents.append(mp.Process(
            #     target=agent,
            #     args=(TRAIN_SEQ_LEN, S_INFO, S_LEN, A_DIM,
            #           save_dir, i, net_params_queues[i], exp_queues[i], trace_scheduler,
            #           video_size_file_dir, self.jump_action)))
        for i in range(self.num_agents):
            agents[i].start()
        with tf.Session() as sess, \
                open(os.path.join(save_dir, 'log_train'), 'w', 1) as log_central_file, \
                open(os.path.join(save_dir, 'log_val'), 'w', 1) as val_log_file:
            log_writer = csv.writer(log_central_file, delimiter='\t', lineterminator='\n')
            log_writer.writerow(['epoch', 'loss', 'avg_reward', 'avg_entropy'])
            val_log_writer = csv.writer(val_log_file, delimiter='\t', lineterminator='\n')
            val_log_writer.writerow(
                ['epoch', 'rewards_min', 'rewards_5per', 'rewards_mean',
                 'rewards_median', 'rewards_95per', 'rewards_max'])

            actor = a3c.ActorNetwork(sess,
                                     state_dim=self.state_dim,
                                     action_dim=A_DIM,
                                     bitrate_dim=BITRATE_DIM,
                                     hidden_dim=self.adaptor_hidden_layer)
                                     # learning_rate=args.ACTOR_LR_RATE)
            critic = a3c.CriticNetwork(sess,
                                       state_dim=self.state_dim,
                                       learning_rate=CRITIC_LR_RATE,
                                       bitrate_dim=BITRATE_DIM,
                                       hidden_dim=self.adaptor_hidden_layer)

            self.train_logger.info('actor and critic initialized')
            # summary_ops, summary_vars = a3c.build_summaries()

            sess.run(tf.global_variables_initializer())
            # writer = tf.summary.FileWriter(save_dir, sess.graph)  # training monitor
            saver = tf.train.Saver(max_to_keep=None)  # save neural net parameters

            # restore neural net parameters
            # if self.model_path:  # nn_model is the path to file
            #     saver.restore(sess, self.model_path)
            #     print("Model restored.")

            os.makedirs(os.path.join(save_dir, "model_saved"), exist_ok=True)

            epoch = 0

            # assemble experiences from agents, compute the gradients

            # val_rewards = [self._test(
            #     actor, trace, video_size_file_dir=self.video_size_file_dir,
            #     save_dir=os.path.join(save_dir, "val_logs")) for trace in self.val_traces]
            # val_mean_reward = np.mean(val_rewards)
            # max_avg_reward = val_mean_reward

            # val_log_writer.writerow(
            #         [epoch, np.min(val_rewards),
            #          np.percentile(val_rewards, 5), np.mean(val_rewards),
            #          np.median(val_rewards), np.percentile(val_rewards, 95),
            #          np.max(val_rewards)])
            # val_epochs.append(epoch)
            # val_mean_rewards.append(val_mean_reward)
            # bit_rate = DEFAULT_QUALITY
            # s_batch = [np.zeros((S_INFO, S_LEN))]
            # action_vec = np.zeros(A_DIM)
            # action_vec[bit_rate] = 1
            # a_batch = [action_vec]
            # r_batch = []
            while epoch < iters:
                start_t = time.time()
                # synchronize the network parameters of work agent
                actor_net_params = actor.get_network_params()
                critic_net_params = critic.get_network_params()
                for i in range(self.num_agents):
                    net_params_queues[i].put([actor_net_params, critic_net_params])

                # record average reward and td loss change
                # in the experiences from the agents
                total_batch_len = 0.0
                total_reward = 0.0
                total_td_loss = 0.0
                total_entropy = 0.0
                total_agents = 0.0

                # assemble experiences from the agents
                actor_gradient_batch = []
                critic_gradient_batch = []

                # linear entropy weight decay(paper sec4.4)
                entropy_weight = 0.5 #entropy_weight_decay_func(epoch)
                current_learning_rate =  0.0001 # learning_rate_decay_func(epoch)

                for i in range(self.num_agents):
                    # print(f"Initial s_batch: {s_batch}")
                    s_batch, a_batch, r_batch, terminal, info = exp_queues[i].get()
                    self.train_logger.info(f"Agent {i} got exp_queues with {len(s_batch)} samples")
                    # print(f"After getting exp_queues: {s_batch}")
                    print("s_batch size: ", len(s_batch))
                    # print(f"s_batch shape: {np.squeeze(np.stack(s_batch, axis=0), axis=1).shape}")
                    # print(f"a_batch shape: {np.vstack(a_batch).shape}")
                    # print(f"r_batch shape: {np.vstack(r_batch).shape}")
                    s_batch = np.stack(s_batch, axis=0)
                    if s_batch.shape[1] == 1:
                        s_batch = np.squeeze(s_batch, axis=1)
                    r_batch = np.vstack(r_batch)
                    a_batch = np.vstack(a_batch)
                    print("s_batch shape: ", s_batch.shape)
                    print("a_batch shape: ", a_batch.shape)
                    print("r_batch shape: ", r_batch.shape)
                    actor_gradient, critic_gradient, td_batch = \
                        a3c.compute_gradients(
                            s_batch=s_batch,
                            a_batch=a_batch,
                            r_batch=r_batch,
                            terminal=terminal, actor=actor,
                            critic=critic,
                            entropy_weight=entropy_weight)

                    actor_gradient_batch.append(actor_gradient)
                    critic_gradient_batch.append(critic_gradient)

                    total_reward += np.sum(r_batch)
                    total_td_loss += np.sum(td_batch)
                    total_batch_len += len(r_batch)
                    total_agents += 1.0
                    total_entropy += np.sum(info['entropy'])

                # compute aggregated gradient
                assert self.num_agents == len(actor_gradient_batch)
                assert len(actor_gradient_batch) == len(critic_gradient_batch)
                # assembled_actor_gradient = actor_gradient_batch[0]
                # assembled_critic_gradient = critic_gradient_batch[0]
                # for i in range(len(actor_gradient_batch) - 1):
                #     for j in range(len(assembled_actor_gradient)):
                #             assembled_actor_gradient[j] += actor_gradient_batch[i][j]
                #             assembled_critic_gradient[j] += critic_gradient_batch[i][j]
                # actor.apply_gradients(assembled_actor_gradient)
                # critic.apply_gradients(assembled_critic_gradient)
                for i in range(len(actor_gradient_batch)):
                    actor.apply_gradients(actor_gradient_batch[i], current_learning_rate)
                    critic.apply_gradients(critic_gradient_batch[i])

                # log training information
                epoch += 1
                avg_reward = total_reward / total_agents
                avg_td_loss = total_td_loss / total_batch_len
                avg_entropy = total_entropy / total_batch_len

                self.train_logger.info('Epoch: ' + str(epoch) +
                            ' TD_loss: ' + str(avg_td_loss) +
                            ' Avg_reward: ' + str(avg_reward) +
                            ' Avg_entropy: ' + str(avg_entropy))
                log_writer.writerow([epoch, avg_td_loss, avg_reward, avg_entropy])

                # summary_str = sess.run(summary_ops, feed_dict={
                #     summary_vars[0]: avg_td_loss,
                #     summary_vars[1]: avg_reward,
                #     summary_vars[2]: avg_entropy
                # })

                # writer.add_summary(summary_str, epoch)
                # writer.flush()

                if epoch % self.model_save_interval == 0:
                    # # Visdom log and plot

                    # val_rewards = [self._test(
                    #     actor, trace, video_size_file_dir=self.video_size_file_dir,
                    #     save_dir=os.path.join(save_dir, "val_logs")) for trace in self.val_traces]
                    # val_mean_reward = np.mean(val_rewards)

                    # val_log_writer.writerow(
                    #         [epoch, np.min(val_rewards),
                    #          np.percentile(val_rewards, 5), np.mean(val_rewards),
                    #          np.median(val_rewards), np.percentile(val_rewards, 95),
                    #          np.max(val_rewards)])
                    # val_epochs.append(epoch)
                    # val_mean_rewards.append(val_mean_reward)
                    # average_rewards.append(np.sum(avg_reward))
                    # average_entropies.append(avg_entropy)

                    # suffix = args.start_time
                    # if args.description is not None:
                    #     suffix = args.description
                    # curve = dict(x=val_epochs, y=val_mean_rewards,
                    #              mode="markers+lines", type='custom',
                    #              marker={'color': plot_color,
                    #                      'symbol': 104, 'size': "5"},
                    #              text=["one", "two", "three"], name='1st Trace')
                    # layout = dict(title="Pensieve_Val_Reward " + suffix,
                    #               xaxis={'title': 'Epoch'},
                    #               yaxis={'title': 'Mean Reward'})
                    # vis._send(
                    #     {'data': [curve], 'layout': layout,
                    #      'win': 'Pensieve_val_mean_reward'})
                    # curve = dict(x=val_epochs, y=average_rewards,
                    #              mode="markers+lines", type='custom',
                    #              marker={'color': plot_color,
                    #                      'symbol': 104, 'size': "5"},
                    #              text=["one", "two", "three"], name='1st Trace')
                    # layout = dict(title="Pensieve_Training_Reward " + suffix,
                    #               xaxis={'title': 'Epoch'},
                    #               yaxis={'title': 'Mean Reward'})
                    # vis._send(
                    #     {'data': [curve], 'layout': layout,
                    #      'win': 'Pensieve_training_mean_reward'})
                    # curve = dict(x=val_epochs, y=average_entropies,
                    #              mode="markers+lines", type='custom',
                    #              marker={'color': plot_color,
                    #                      'symbol': 104, 'size': "5"},
                    #              text=["one", "two", "three"], name='1st Trace')
                    # layout = dict(title="Pensieve_Training_Mean Entropy " + suffix,
                    #               xaxis={'title': 'Epoch'},
                    #               yaxis={'title': 'Mean Entropy'})
                    # vis._send(
                    #     {'data': [curve], 'layout': layout,
                    #      'win': 'Pensieve_training_mean_entropy'})

                    # if val_mean_reward > max_avg_reward:
                    # max_avg_reward = val_mean_reward
                    # Save the neural net parameters to disk.
                    save_path = saver.save(
                        sess,
                        os.path.join(save_dir, "model_saved", f"nn_model_ep_{epoch}.ckpt"))
                    self.train_logger.info("Model saved in file: " + save_path)

                end_t = time.time()
                # print(f'epoch{epoch-1}: {end_t - start_t}s')

            for tmp_agent in agents:
                tmp_agent.terminate()


    def calculate_from_selection(self, selected ,last_bit_rate):
        # selected_action is 0-5
        # naive step implementation
        if selected == 1:
            bit_rate = last_bit_rate
        elif selected == 2:
            bit_rate = last_bit_rate + 1
        else:
            bit_rate = last_bit_rate - 1
        # bound
        if bit_rate < 0:
            bit_rate = 0
        if bit_rate > 5:
            bit_rate = 5

        # print(bit_rate)
        return bit_rate

    def select_action(self, state, last_bit_rate, use_embedding=False, embeddings=None):
        if use_embedding:
            self.test_logger.info("use embedding")
            if self.adaptor_input is not None:
                if self.adaptor_input == "ACTION":
                    self.test_logger.info("action adaptor")
                    original_action_prob = self.original_actor.predict( np.reshape( state ,(1 ,S_INFO ,S_LEN) ) )
                    original_action_cumsum = np.cumsum( original_action_prob )
                    original_selection = (original_action_cumsum > np.random.randint(
                        1 ,RAND_RANGE ) / float( RAND_RANGE )).argmax()
                    original_bit_rate = calculate_from_selection( original_selection ,last_bit_rate )
                    
                    self.test_logger.info(f"Original action: {original_bit_rate}")                    
                    self.test_logger.info(f"embeddings type: {type(embeddings)}, shape: {embeddings.shape}")

                    adaptor_input = np.concatenate((np.array([original_bit_rate]), embeddings), axis=0)
                    self.test_logger.info(f"adaptor_input shape: {adaptor_input.shape}")                    
                    action_prob = self.net.predict(adaptor_input.reshape(1, -1))  # Expands to shape (1, 17)
                    action_cumsum = np.cumsum(action_prob)
                    selection = (action_cumsum > np.random.randint(
                        1, RAND_RANGE) / float(RAND_RANGE)).argmax()
                    bit_rate = calculate_from_selection(selection, original_bit_rate)
                    self.test_logger.info("Select action with actor adaptor - original_bit_rate: {}, bit_rate: {}".format(original_bit_rate, bit_rate))
                    return bit_rate
                elif self.adaptor_input == "HIDDEN":
                    self.test_logger.info("hidden adaptor")
                    original_hidden =  self.original_actor.get_hidden(np.reshape(state, (1, S_INFO, S_LEN)))

                    # Flatten the hidden state output
                    original_hidden_flat = original_hidden.flatten()  # Converts (1, 128) -> (128,)

                    self.test_logger.info(f"Original hidden: {original_hidden.shape}")
                    
                    # Flatten embeddings if necessary
                    embeddings_flat = embeddings.flatten()  # Converts (16,) -> (16,)
                    
                    # print(f"original_bit_rate type: {type(original_bit_rate)}, shape: {np.shape(original_bit_rate)}")
                    self.test_logger.info(f"embeddings type: {type(embeddings)}, shape: {embeddings.shape}")
                    adaptor_input = np.concatenate((original_hidden_flat, embeddings_flat))  # Shape: (128 + 16,) -> (144,)
                    # adaptor_input = np.concatenate((np.array([original_hidden]), embeddings), axis=0)
                    self.test_logger.info(f"adaptor_input shape: {adaptor_input.shape}")                    

                    action_prob = self.net.predict(adaptor_input.reshape(1, -1))  # Expands to shape (1, 17)
                else:
                    self.test_logger.info("no adaptor")
                    action_prob = self.net.predict( np.reshape( state ,(1, S_INFO+EMBEDDING_SIZE, S_LEN) ) )
        else:
            self.test_logger.info("no embedding")
            action_prob = self.net.predict( np.reshape( state ,(1, S_INFO, S_LEN) ) )
        action_cumsum = np.cumsum( action_prob )
        selection = (action_cumsum > np.random.randint(
            1 ,RAND_RANGE ) / float( RAND_RANGE )).argmax()
        bit_rate = self.calculate_from_selection( selection ,last_bit_rate )
        return bit_rate

    def evaluate(self, net_env, save_dir=None):
        torch.set_num_threads(1)
        net_env.reset()
        results = []
        time_stamp = 0
        bit_rate = DEFAULT_QUALITY
        while True:  # serve video forever
            # the action is from the last decision
            # this is to make the framework similar to the real
            state, reward, end_of_video, info = net_env.step(bit_rate)

            time_stamp += info['delay']  # in ms
            time_stamp += info['sleep_time']  # in ms

            results.append([time_stamp / M_IN_K, VIDEO_BIT_RATE[bit_rate],
                            info['buffer_size'], info['rebuf'],
                            info['video_chunk_size'], info['delay'], reward])

            state = torch.from_numpy(state).type('torch.FloatTensor')
            bit_rate, action_prob_vec = self.net.select_action(state)
            bit_rate = np.argmax(action_prob_vec)
            if end_of_video:
                break
        if save_dir is not None:
            # write to a file for the purpose of multiprocessing
            log_path = os.path.join(save_dir, "log_sim_rl_{}".format(
                net_env.trace_file_name))
            with open(log_path, 'w', 1) as f:
                csv_writer = csv.writer(f, delimiter='\t', lineterminator="\n")
                csv_writer.writerow(['time_stamp', 'bitrate', 'buffer_size',
                                     'rebuffer', 'video chunk_size', 'delay',
                                     'reward'])

                csv_writer.writerows(results)
        return results

    def evaluate_envs(self, net_envs):
        arguments = [(net_env, ) for net_env in net_envs]
        with mp.Pool(processes=8) as pool:
            results = pool.starmap(self.evaluate, arguments)
        return results

    def save_models(self, model_save_path):
        """Save models to a directory."""
        self.net.save_actor_model(os.path.join(model_save_path, "actor.pth"))
        self.net.save_critic_model(os.path.join(model_save_path, "critic.pth"))

    def load_models(self, actor_model_path, critic_model_path):
        """Load models from given paths."""
        if actor_model_path is not None:
            self.net.load_actor_model(actor_model_path)
        if critic_model_path is not None:
            self.net.load_critic_model(critic_model_path)

    def central_agent(self, net_params_queues, exp_queues, iters, train_envs,
                      val_envs, test_envs, use_replay_buffer):
        """Pensieve central agent.

        Collect states, rewards, etc from each agent and train the model.
        """
        torch.set_num_threads(2)

        self.central_logger.info(
            "Central agent started with %d agents for %d iterations.",
            self.num_agents, iters
        )

        assert self.net.is_central
        log_header = ['epoch', 'rewards_min', 'rewards_5per', 'rewards_mean',
                      'rewards_median', 'rewards_95per', 'rewards_max']
        test_log_writer = csv.writer(
            open(os.path.join(self.log_dir, 'log_test'), 'w', 1),
            delimiter='\t', lineterminator='\n')
        test_log_writer.writerow(log_header)

        train_e2e_log_writer = csv.writer(
            open(os.path.join(self.log_dir, 'log_train_e2e'), 'w', 1),
            delimiter='\t', lineterminator='\n')
        train_e2e_log_writer.writerow(log_header)

        val_log_writer = csv.writer(
            open(os.path.join(self.log_dir, 'log_val'), 'w', 1),
            delimiter='\t', lineterminator='\n')
        val_log_writer.writerow(log_header)

        t_start = time.time()
        for epoch in range(int(iters)):
            # synchronize the network parameters of work agent
            actor_net_params = self.net.get_actor_param()
            actor_net_params = [params.detach().cpu().numpy()
                                for params in actor_net_params]

            for i in range(self.num_agents):
                net_params_queues[i].put(actor_net_params)
                # Note: this is synchronous version of the parallel training,
                # which is easier to understand and probe. The framework can be
                # fairly easily modified to support asynchronous training.
                # Some practices of asynchronous training (lock-free SGD at
                # its core) are nicely explained in the following two papers:
                # https://arxiv.org/abs/1602.01783
                # https://arxiv.org/abs/1106.5730

            # record average reward and td loss change
            # in the experiences from the agents
            total_batch_len = 0.0
            total_reward = 0.0
            # total_td_loss = 0.0
            total_entropy = 0.0
            total_agents = 0.0

            # assemble experiences from the agents
            # actor_gradient_batch = []
            # critic_gradient_batch = []
            for i in range(self.num_agents):
                s_batch, a_batch, r_batch, terminal, info = exp_queues[i].get()
                entropy = info['entropy']
                # add s a r e into replay buffer and sample data out of buffer
                if use_replay_buffer:
                    for s, a, r, e in zip(s_batch, a_batch, r_batch, entropy):
                        self.replay_buffer.add((s, a, r, e))
                    s_batch, a_batch, r_batch, entropy = self.replay_buffer.sample(
                        self.batch_size)

                self.net.get_network_gradient(
                    s_batch, a_batch, r_batch, terminal=terminal,
                    epoch=self.epoch)
                total_reward += np.sum(r_batch)
                total_batch_len += len(r_batch)
                total_agents += 1.0
                total_entropy += np.sum(entropy)
            print('central_agent: {}/{}, total epoch trained {}'.format(
                epoch, int(iters), self.epoch))

            # log training information
            self.net.update_network()

            avg_reward = total_reward / total_agents
            avg_entropy = total_entropy / total_batch_len

            self.central_logger.info('Epoch: {} Avg_reward: {} Avg_entropy: {}'.format(
                self.epoch, avg_reward, avg_entropy))

            if (self.epoch+1) % self.model_save_interval == 0:
                # Save the neural net parameters to disk.
                print("Train epoch: {}/{}, time use: {}s".format(
                    epoch + 1, iters, time.time() - t_start))
                self.net.save_critic_model(os.path.join(
                    self.log_dir, "critic_ep_{}.pth".format(self.epoch + 1)))
                self.net.save_actor_model(os.path.join(
                    self.log_dir, "actor_ep_{}.pth".format(self.epoch + 1)))

                # tmp_save_dir = os.path.join(self.log_dir, 'test_results')
                if val_envs is not None:
                    val_results = self.evaluate_envs(val_envs)
                    vid_rewards = np.array(
                        [np.sum(np.array(vid_results)[1:, -1])
                         for vid_results in val_results])
                    val_log_writer.writerow([self.epoch + 1,
                                             np.min(vid_rewards),
                                             np.percentile(vid_rewards, 5),
                                             np.mean(vid_rewards),
                                             np.median(vid_rewards),
                                             np.percentile(vid_rewards, 95),
                                             np.max(vid_rewards)])
                if test_envs is not None:
                    test_results = self.evaluate_envs(test_envs)
                    vid_rewards = np.array(
                        [np.sum(np.array(vid_results)[1:, -1])
                         for vid_results in test_results])
                    test_log_writer.writerow([self.epoch + 1,
                                              np.min(vid_rewards),
                                              np.percentile(vid_rewards, 5),
                                              np.mean(vid_rewards),
                                              np.median(vid_rewards),
                                              np.percentile(vid_rewards, 95),
                                              np.max(vid_rewards)])
                t_start = time.time()
                # TODO: process val results and write into log
                # evaluate_envs(net, train_envs)
            self.epoch += 1

        # signal all agents to exit, otherwise they block forever.
        for i in range(self.num_agents):
            net_params_queues[i].put("exit")

def build_state(msg, agent_state):
    """
    msg: dict with {"last_quality":..., "rebuffer_time":..., "chunk_fetch_time":..., ...}
    agent_state: the [S_INFO, S_LEN] array you're updating
    """
    # SHIFT the existing columns left
    agent_state = np.roll(agent_state, -1, axis=-1)

    # For example, store last_quality normalized by max bit_rate=5
    agent_state[0, -1] = msg["last_quality"] / 5.0  
    agent_state[1, -1] = msg["rebuffer_time"] / 1000.0
    agent_state[2, -1] = msg["chunk_fetch_time"] / 1000.0
    # etc.

    return agent_state

def compute_reward(msg, bit_rate, last_quality):
    """
    msg: dict describing the chunk
    bit_rate: chosen action (0..5)
    last_quality: previous action
    """
    # e.g. you can do a "smoothness" penalty or rebuffer penalty:
    rebuffer_sec = msg["rebuffer_time"] / 1000.0
    video_quality = VIDEO_BIT_RATE[bit_rate]
    previous_quality = VIDEO_BIT_RATE[last_quality]
    reward = video_quality \
        - 4.3 * rebuffer_sec \
        - abs(video_quality - previous_quality)
    return reward



def agent(agent_id, net_params_queue, exp_queue, train_envs,
          summary_dir, batch_size, randomization, randomization_interval,
          num_agents, original_actor_path, adaptor_input_type, adaptor_hidden_layer, s_dim):
    """
    Each agent process picks an environment (delay, trace) from train_envs,
    starts a Mahimahi shell, runs the virtual_browser, collects data, etc.
    Then sends experiences to the central agent.
    """
    agent_logger = setup_logger(
        f"agent_{agent_id}",
        f"/mydata/logs/{agent_id}_agent.log"
    )
    agent_logger.info("Agent %d started!", agent_id)

    _, _, video_server_port = rl_embedding.launch_video_server_and_bftrace(agent_id, agent_logger)

    # 3) Create redis for state/action communication
    redis_client = redis.Redis(host="10.10.1.2", port=2666, decode_responses=True)
    redis_client.set(f"{agent_id}_action_flag", int(False))
    # redis_client.set(f"{agent_id}_stop_flag", int(False))

    with tf.compat.v1.Session() as sess:
        original_actor = OriginalActorNetwork(sess,
                                            state_dim=[S_INFO, S_LEN],
                                            action_dim=A_DIM,
                                            bitrate_dim=BITRATE_DIM)
        sess.run(tf.global_variables_initializer())
        original_saver = tf.train.Saver()
        if original_actor_path is not None:
            original_saver.restore(sess, original_actor_path)
            print("Original model restored.")
        
        actor = a3c.ActorNetwork(sess, state_dim=EMBEDDING_SIZE+1,
                                 action_dim=A_DIM, bitrate_dim=BITRATE_DIM,
                                 hidden_dim=adaptor_hidden_layer)
        critic = a3c.CriticNetwork(sess, state_dim=EMBEDDING_SIZE+1,
                                  learning_rate=CRITIC_LR_RATE,
                                  bitrate_dim=BITRATE_DIM,
                                  hidden_dim=adaptor_hidden_layer)

        # Initial synchronization of network parameters from the coordinator
        actor_net_params, critic_net_params = net_params_queue.get()
        actor.set_network_params(actor_net_params)
        critic.set_network_params(critic_net_params)

        last_bit_rate = DEFAULT_QUALITY
        selection = 0
        bit_rate = DEFAULT_QUALITY

        action_vec = np.zeros(A_DIM)
        action_vec[selection] = 1

        s_batch = [np.zeros(s_dim)]
        a_batch = [action_vec]
        r_batch = []
        entropy_record = []
        embeddings, tokens = rl_embedding.null_embedding_and_token()

        time_stamp = 0
        epoch = 0
        state = None
        reward = None

        # 2) Get its training environment with a random delay and a random synthetic trace
        # np.random.seed(agent_id)
        delay_val = random.choice(train_envs["delay_list"])
        scheduler = train_envs["train_scheduler"]
        abr_trace = scheduler.get_trace()
        mahimahi_trace_path = summary_dir+"/trace_"+str(agent_id)
        abr_trace.convert_to_mahimahi_format(mahimahi_trace_path)
        print(f"Agent {agent_id} using trace {mahimahi_trace_path}")

        # 3) Launch Mahimahi + virtual browser
        mahimahi_dir = "src/emulator/abr"
        mm_cmd = (
            f'mm-delay {delay_val} mm-loss uplink 0 mm-loss downlink 0 mm-link {UP_LINK_SPEED_FILE} {mahimahi_trace_path} -- bash -c \"python -m pensieve.virtual_browser.virtual_browser --ip \$MAHIMAHI_BASE --port {video_server_port} --abr RLTrain --video-size-file-dir {VIDEO_SIZE_DIR} --summary-dir {summary_dir}/pensieve_{agent_id}_{delay_val} --trace-file {mahimahi_trace_path} --abr-server-port=8322\"'
        )
        print(f"[Agent {agent_id}] Starting environment:\n{mm_cmd}")
        agent_logger.info(f"[Agent {agent_id}] Starting environment:\n{mm_cmd}")
        mm_proc = subprocess.Popen(mm_cmd, shell=True, cwd=mahimahi_dir)

        # 4) Main Loop
        while True:
            browser_active = redis_client.get(f"{agent_id}_browser_active")
            # print("browser_active", browser_active)
            agent_logger.info(f"Browser active: {browser_active}")
            if browser_active and int(browser_active) == 1:
                # Set action and flag in Redis
                redis_pipe = redis_client.pipeline(transaction=True)
                redis_pipe.set(f"{agent_id}_action", str(bit_rate))
                redis_pipe.set(f"{agent_id}_action_flag", int(True))
                try:
                    redis_pipe.execute()
                except Exception as e:
                    print(f"Exception {e}")
                    agent_logger.info(f"redis_pipe Exception {e}")
                # read from redis
                recv_state = False
                while not recv_state:
                    redis_pipe = redis_client.pipeline(transaction=True)
                    redis_pipe.get(f"{agent_id}_state")
                    redis_pipe.get(f"{agent_id}_reward")
                    redis_pipe.get(f"{agent_id}_state_flag")
                    try:
                        retval = redis_pipe.execute()
                    except Exception as e:
                        print(f"Exception {e}")
                        agent_logger.info(f"redis_pipe Exception {e}")
                    #print(f"Retval {retval}")
                    if retval[2] is not None:
                        if int(retval[2]) == 1:
                            agent_logger.info(f"[Agent {agent_id}] Received state flag: {retval[2]}")
                            redis_client.set(f"{agent_id}_state_flag", int(False))
                            state = json.loads(retval[0])
                            agent_logger.info(f"[Agent {agent_id}] Received state shape: {np.array(state).shape}.")
                            reward = float(retval[1])
                            # print(f"[Agent {agent_id}] Received state: {state}.")
                            recv_state = True
                            agent_logger.info(f"[Agent {agent_id}] Received state flag: {redis_client.get(f'{agent_id}_state_flag')}")
                    end_of_video = redis_client.get(f"{agent_id}_stop_flag")
                    if end_of_video and int(end_of_video) == 1:
                        agent_logger.info(f"[Agent {agent_id}] end_of_video received: {end_of_video}")
                        recv_state = True

                agent_logger.info(f"[Agent {agent_id}] recv_state: {recv_state}, end_of_video: {end_of_video}")
                # If state received, process it
                if recv_state:
                    state, embeddings, tokens = rl_embedding.transform_state_and_add_embedding(agent_id, state, embeddings, tokens)                    
                    r_batch.append(reward)
                    if (adaptor_input_type == "original_action_prob"):
                        original_action_prob = original_actor.predict(np.reshape(state, (1, S_INFO, S_LEN)))
                        adaptor_input = np.concatenate((original_action_prob, embeddings), axis=0)
                    elif (adaptor_input_type == "original_selection"):
                        # Get original actor model action
                        original_action_prob = original_actor.predict( np.reshape( state ,(1 ,S_INFO ,S_LEN) ) )
                        original_action_cumsum = np.cumsum( original_action_prob )
                        original_selection = (original_action_cumsum > np.random.randint(
                            1 ,RAND_RANGE ) / float( RAND_RANGE )).argmax()
                        adaptor_input = np.concatenate((np.array([original_selection]), embeddings), axis=0)
                    elif (adaptor_input_type == "original_bit_rate"):
                        original_action_prob = original_actor.predict( np.reshape( state ,(1 ,S_INFO ,S_LEN) ) )
                        original_action_cumsum = np.cumsum( original_action_prob )
                        original_selection = (original_action_cumsum > np.random.randint(
                            1 ,RAND_RANGE ) / float( RAND_RANGE )).argmax()
                        bit_rate = calculate_from_selection( original_selection ,last_bit_rate )
                        adaptor_input = np.concatenate((np.array([bit_rate]), embeddings), axis=0)
                    elif (adaptor_input_type == "hidden_state"):
                        original_hidden = original_actor.get_hidden(np.reshape(state, (1, S_INFO, S_LEN)))
                        # Flatten the hidden state output
                        original_hidden_flat = original_hidden.flatten()  # Converts (1, 128) -> (128,)
                        # Flatten embeddings if necessary
                        embeddings_flat = embeddings.flatten()  # Converts (16,) -> (16,)
                        # print(f"original_bit_rate type: {type(original_bit_rate)}, shape: {np.shape(original_bit_rate)}")
                        agent_logger.info(f"embeddings type: {type(embeddings)}, shape: {embeddings.shape}")
                        adaptor_input = np.concatenate((original_hidden_flat, embeddings_flat))

                    action_prob = actor.predict(adaptor_input.reshape(1, -1))  # Expands to shape (1, 17)
                    if np.isnan(action_prob[0, 0]) and agent_id == 0:
                        print(epoch)
                        print(state, "state")
                        print(action_prob, "action prob")
                        import pdb
                        pdb.set_trace()
                    action_cumsum = np.cumsum(action_prob)
                    selection = (action_cumsum > np.random.randint(
                        1, RAND_RANGE) / float(RAND_RANGE)).argmax()
                    bit_rate = calculate_from_selection(selection, last_bit_rate)
                    entropy_record.append(a3c.compute_entropy(action_prob[0]))

                    end_of_video = redis_client.get(f"{agent_id}_stop_flag")
                    agent_logger.info(f"[Agent {agent_id}] end_of_video check 1: {end_of_video}")
                    if len(r_batch) >= TRAIN_SEQ_LEN or (end_of_video and int(end_of_video) == 1):
                        exp_queue.put([s_batch[1:],  # ignore the first chuck
                                    a_batch[1:],  # since we don't have the
                                    r_batch[1:],  # control over it
                                    end_of_video,
                                    {'entropy': entropy_record}])
                        agent_logger.info(f"[Agent {agent_id}] sent experience to central agent.")
                        print(f"[Agent {agent_id}] sent experience to central agent.")
                        # print("s_batch shape: ", s_batch.shape)
                        # print("sent state shape: ", s_batch[1:].shape)

                        # synchronize the network parameters from the coordinator
                        actor_net_params, critic_net_params = net_params_queue.get()
                        actor.set_network_params(actor_net_params)
                        critic.set_network_params(critic_net_params)

                        # Reset batches
                        del s_batch[:]
                        del a_batch[:]
                        del r_batch[:]
                        del entropy_record[:]

                            # so that in the log we know where video ends

                # store the state and action into batches
                end_of_video = redis_client.get(f"{agent_id}_stop_flag")
                agent_logger.info(f"[Agent {agent_id}] end_of_video check 2: {end_of_video}")
                if end_of_video and int(end_of_video) == 1:
                    last_bit_rate = DEFAULT_QUALITY
                    bit_rate = DEFAULT_QUALITY  # use the default action here
                    #action_vec = np.array( [VIDEO_BIT_RATE[last_bit_rate] ,VIDEO_BIT_RATE[bit_rate] ,selection] )
                    action_vec = np.zeros(A_DIM)
                    selection = 0
                    action_vec[selection] = 1
                    s_batch.append(np.zeros((s_dim)))
                    a_batch.append(action_vec)
                    r_batch.append(0)
                    embeddings, tokens = rl_embedding.null_embedding_and_token()
                    epoch += 1
                    # reset virtual browser
                    agent_logger.info(f"[Agent {agent_id}] Resetting virtual browser.")
                    print(f"[Agent {agent_id}] Resetting virtual browser.")
                    # redis_client.flushdb() # Wrong, should only reset the agent's state
                    # only flush the agent's state
                    for key in redis_client.scan_iter(f"{agent_id}_*"):
                        redis_client.delete(key)
                    agent_logger.info(redis_client.keys(f"{agent_id}_*"))
                    redis_client.set(f"{agent_id}_browser_active", 0)
                    redis_client.set(f"{agent_id}_new_epoch", 1)
                    # store the state and action into batches
                    end_of_video = redis_client.get(f"{agent_id}_stop_flag")
                    agent_logger.info(f"[Agent {agent_id}] end_of_video check 2: {end_of_video}")
                    if end_of_video and int(end_of_video) == 1:
                        last_bit_rate = DEFAULT_QUALITY
                        bit_rate = DEFAULT_QUALITY  # use the default action here
                        #action_vec = np.array( [VIDEO_BIT_RATE[last_bit_rate] ,VIDEO_BIT_RATE[bit_rate] ,selection] )
                        action_vec = np.zeros(A_DIM)
                        selection = 0
                        action_vec[selection] = 1
                        s_batch.append(np.zeros(s_dim))
                        a_batch.append(action_vec)
                        # r_batch.append(0)
                        tokens = np.array([])
                        embeddings = np.zeros((EMBEDDING_SIZE), dtype=np.float32)
                        epoch += 1
                        # reset virtual browser
                        agent_logger.info(f"[Agent {agent_id}] Resetting virtual browser.")
                        print(f"[Agent {agent_id}] Resetting virtual browser.")
                        # redis_client.flushdb() # Wrong, should only reset the agent's state
                        # only flush the agent's state
                        for key in redis_client.scan_iter(f"{agent_id}_*"):
                            redis_client.delete(key)
                        agent_logger.info(redis_client.keys(f"{agent_id}_*"))
                        redis_client.set(f"{agent_id}_browser_active", 0)
                        redis_client.set(f"{agent_id}_new_epoch", 1)
                        redis_client.set(f"{agent_id}_stop_flag", int(False))

                    else:
                        s_batch.append(adaptor_input)
                        action_vec = np.zeros(A_DIM)
                        action_vec[selection] = 1
                        #print(action_vec)
                        a_batch.append(action_vec)
                        # r_batch.append(reward)
            else:
                # browser inactive
                time.sleep(10)

        # Wait for the environment to finish
        mm_proc.wait()
        # No token_proc in single-process approach
        print(f"[Agent {agent_id}] Finished.")

def calculate_from_selection(selected, last_bit_rate):
    # naive step implementation
    # action=0, bitrate-1; action=1, bitrate stay; action=2, bitrate+1
    if selected == 1:
        bit_rate = last_bit_rate
    elif selected == 2:
        bit_rate = last_bit_rate + 1
    else:
        bit_rate = last_bit_rate - 1
    # bound
    bit_rate = max(0, bit_rate)
    bit_rate = min(5, bit_rate)

    return bit_rate
