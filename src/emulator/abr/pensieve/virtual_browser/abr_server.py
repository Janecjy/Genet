import tensorflow as tf
import argparse
import csv
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
import sys
import time
import json
import redis
# print("Pong")
redis_client = redis.Redis(host="130.127.133.218", port=6379, decode_responses=True)
print("Ping Redis success")
redis_client.ping()
import numpy as np

from pensieve.agent_policy import Pensieve, RobustMPC, BufferBased, FastMPC, RLTrain
from pensieve.a3c.a3c_jump import ActorNetwork

from pensieve.constants import (
    A_DIM,
    BUFFER_NORM_FACTOR,
    DEFAULT_QUALITY,
    M_IN_K,
    S_INFO,
    S_LEN,
    TOTAL_VIDEO_CHUNK,
    VIDEO_BIT_RATE,
)
from pensieve.utils import construct_bitrate_chunksize_map, linear_reward

RANDOM_SEED = 42
RAND_RANGE = 1000


def parse_args():
    """Parse arguments from the command line."""
    parser = argparse.ArgumentParser("Video Server")
    parser.add_argument('--description', type=str, default=None,
                        help='Optional description of the experiment.')
    # ABR related
    parser.add_argument('--abr', type=str, required=True,
                        choices=['RobustMPC', 'RL', 'BufferBased', 'FastMPC'],
                        help='ABR algorithm.')
    parser.add_argument('--actor-path', type=str, default=None,
                        help='Path to RL model.')
    # data io related
    parser.add_argument('--summary-dir', type=str,
                        help='directory to save logs.')
    parser.add_argument('--trace-file', type=str, help='Path to trace file.')
    parser.add_argument("--video-size-file-dir", type=str, required=True,
                        help='Dir to video size files')

    # networking related
    parser.add_argument('--ip', type=str, default='localhost',
                        help='ip address of ABR/video server.')
    parser.add_argument('--port', type=int, default=8333,
                        help='port number of ABR/video server.')

    return parser.parse_args()


temp_test = "Available"
def make_request_handler(server_states):
    """Instantiate HTTP request handler."""

    class Request_Handler(BaseHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            self.server_states = server_states
            self.abr = server_states['abr']
            self.video_size = server_states['video_size']
            self.log_writer = server_states['log_writer']
            self.sess = server_states['sess']
            self.actor = server_states['actor']
            self.summary_dir = server_states['summary_dir']
            self.agent_id = server_states['agent_id'] #"0"#os.path.basename(self.summary_dir).split("_")[1]
            print("Agent ID {}".format(self.agent_id))
            print("Redis keys {}".format(redis_client.keys()))
            self.redis_client = redis.Redis(host="130.127.133.218", port=6379, decode_responses=True)
            print("Redis end")
            print("Redis keys {}".format(redis_client.keys()))
            BaseHTTPRequestHandler.__init__(self, *args, **kwargs)

        def do_POST(self):
            #self.redis_client = redis.Redis(host="130.127.133.218", port=6379, decode_responses=True)
            #print("New keys")
            #print(self.redis_client.keys())
            content_length = int(self.headers['Content-Length'])
            post_data = json.loads(self.rfile.read(
                content_length).decode('utf-8'))

            print("\tlastRequest: {}\n\tlastquality: {}\n\t"
                  "lastChunkStartTime: {}\n\tlastChunkEndTime: {}\n\t"
                  "lastChunkSize: {}\n\tRebufferTime: {}s\n\tbuffer: {}s\n\t"
                  "bufferAdjusted: {}\n\tbandwidthEst: {}\n\t"
                  "nextChunkSize: {}".format(
                      post_data['lastRequest'],
                      post_data['lastquality'],
                      post_data['lastChunkStartTime'],
                      post_data['lastChunkFinishTime'],
                      post_data['lastChunkSize'],
                      post_data['RebufferTime'] / M_IN_K,
                      post_data['buffer'],
                      post_data['bufferAdjusted'],
                      post_data['bandwidthEst'],
                      post_data['nextChunkSize'],
                  ))
            # print(f"Post Data {post_data}")
            if ('pastThroughput' in post_data):
                print("Summary: ", post_data)
            else:
                rebuffer_time = float(
                    post_data['RebufferTime'] -
                    self.server_states['last_total_rebuf'])

                reward = linear_reward(
                    VIDEO_BIT_RATE[post_data['lastquality']],
                    VIDEO_BIT_RATE[self.server_states['last_bit_rate']],
                    rebuffer_time / M_IN_K)

                self.server_states['last_bit_rate'] = post_data['lastquality']
                self.server_states['last_total_rebuf'] = post_data['RebufferTime']

                # compute bandwidth measurement
                video_chunk_fetch_time = post_data['lastChunkFinishTime'] - \
                    post_data['lastChunkStartTime']
                # print('video chunk fetch time:', video_chunk_fetch_time)
                video_chunk_size = post_data['lastChunkSize']

                # compute number of video chunks left
                self.server_states['video_chunk_count'] += 1
                video_chunk_remain = TOTAL_VIDEO_CHUNK - \
                    self.server_states['video_chunk_count']

                next_video_chunk_sizes = []
                for i in range(A_DIM):
                    if 0 <= self.server_states['video_chunk_count'] < TOTAL_VIDEO_CHUNK:
                        next_video_chunk_sizes.append(
                            self.video_size[i][self.server_states['video_chunk_count']])
                    else:
                        next_video_chunk_sizes.append(0)

                # this should be S_INFO number of terms
                try:
                    state0 = VIDEO_BIT_RATE[post_data['lastquality']
                                            ] / max(VIDEO_BIT_RATE)
                    state1 = post_data['buffer'] / BUFFER_NORM_FACTOR
                    # kilo byte / ms
                    state2 = video_chunk_size / video_chunk_fetch_time / M_IN_K
                    state3 = video_chunk_fetch_time / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec

                    # mega byte
                    state4 = np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K
                    state5 = min(video_chunk_remain,
                                 TOTAL_VIDEO_CHUNK) / TOTAL_VIDEO_CHUNK
                    # dequeue history record
                    self.server_states['state'] = np.roll(
                        self.server_states['state'], -1, -1)
                    self.server_states['state'][0, 0, -1] = state0
                    self.server_states['state'][0, 1, -1] = state1
                    self.server_states['state'][0, 2, -1] = state2
                    self.server_states['state'][0, 3, -1] = state3
                    self.server_states['state'][0, 4, :A_DIM] = state4
                    self.server_states['state'][0, 5, -1] = state5

                except ZeroDivisionError:
                    pass
                
                state_msg = self.server_states['state'].tolist()
                # print("Test sfas ")
                print(f"redis pipe get {self.redis_client.get(f'{self.agent_id}_action_flag')}")
                # print (f"Server State {state_msg}")
                # print("Pipe Set")
                redis_pipe = self.redis_client.pipeline(transaction=True)
                # print("Pipe Unset")
                redis_pipe.set(f"{self.agent_id}_state", json.dumps(state_msg))
                redis_pipe.set(f"{self.agent_id}_reward", reward)
                redis_pipe.set(f"{self.agent_id}_state_flag", int(True))
                try:
                    retval = redis_pipe.execute()
                except Exception as e:
                    print(f"Exception err 1{e}")

                self.log_writer.writerow(
                    [time.time(), VIDEO_BIT_RATE[post_data['lastquality']],
                     post_data['buffer'], rebuffer_time / M_IN_K,
                     video_chunk_size, video_chunk_fetch_time, reward,
                     post_data['bandwidthEst'] / 1000,
                     self.server_states['future_bandwidth']])

                if isinstance(self.abr, Pensieve):
                    bit_rate = self.abr.select_action(
                        self.server_states['state'], last_bit_rate=self.server_states['last_bit_rate'])
                elif isinstance(self.abr, RobustMPC):
                    last_index = int(post_data['lastRequest'])
                    future_chunk_cnt = min(self.abr.mpc_future_chunk_cnt,
                                           TOTAL_VIDEO_CHUNK - last_index - 1)
                    bit_rate, self.server_states['future_bandwidth'] = \
                        self.abr.select_action(
                        self.server_states['state'], last_index,
                        future_chunk_cnt, np.array(
                            [self.video_size[i]
                             for i in sorted(self.video_size)]),
                        post_data['lastquality'], post_data['buffer'])
                elif isinstance(self.abr, BufferBased):
                    bit_rate = self.abr.select_action(post_data['buffer'])
                elif isinstance(self.abr, FastMPC):
                    last_index = int( post_data['lastRequest'] )
                    future_chunk_cnt = min( self.abr.mpc_future_chunk_cnt ,
                                            TOTAL_VIDEO_CHUNK - last_index - 1 )
                    bit_rate ,self.server_states['future_bandwidth'] = \
                        self.abr.select_action(
                            self.server_states['state'] ,last_index ,
                            future_chunk_cnt ,np.array(
                                [self.video_size[i]
                                 for i in sorted( self.video_size )] ) ,
                            post_data['lastquality'] ,post_data['buffer'] )
                elif isinstance(self.abr, RLTrain):
                    # print("RLTrain selects action")
                    action_recv = False
                    while not action_recv:
                        
                        redis_pipe = self.redis_client.pipeline(transaction=True)
                        state = redis_pipe.get(f"{self.agent_id}_action")
                        flag = redis_pipe.get(f"{self.agent_id}_action_flag")
                        redis_pipe.set(f"{self.agent_id}_action_flag", int(False))
                        
                        try:
                            retval = redis_pipe.execute()
                        except Exception as e:
                            print(f"Exception execute {e}")
                        if retval[1] and int(retval[1]):
                            bit_rate = int(retval[0])
                            action_recv = True
                            print(f"RLTrain received action {bit_rate}")
                else:
                    raise TypeError("Unsupported ABR type.")

                send_data = str(bit_rate)

                end_of_video = post_data['lastRequest'] == TOTAL_VIDEO_CHUNK - 1
                # print("last request", post_data['lastRequest'])
                # print("Total video chunk", TOTAL_VIDEO_CHUNK)
                if end_of_video:
                    # send_data = "REFRESH"
                    print("End of video")
                    send_data = "stop"  # TODO: do not refresh the webpage and wait for timeout
                    self.server_states['last_total_rebuf'] = 0
                    self.server_states['last_bit_rate'] = DEFAULT_QUALITY
                    self.server_states['video_chunk_count'] = 0
                    self.server_states['state'] = np.zeros((1, S_INFO, S_LEN))
                    self.redis_client.set(f"{self.agent_id}_stop_flag", int(True))

                self.send_response(200)
                self.send_header('Content-Type', 'text/plain')
                self.send_header('Content-Length', str(len(send_data)))
                self.send_header('Access-Control-Allow-Origin', "*")
                self.end_headers()
                self.wfile.write(send_data.encode())


        def do_GET(self):
            print('GOT REQ')
            self.send_response(200)
            self.send_header('Cache-Control', 'max-age=3000')
            self.send_header('Content-Length', '20')
            self.end_headers()
            self.wfile.write(b"console.log('here');")

    return Request_Handler

def run_abr_server(abr, trace_file, summary_dir, actor_path,
                   video_size_file_dir, ip='localhost', port=8333,):
    print(f"Summary Directory {summary_dir}")
    os.makedirs(summary_dir, exist_ok=True)
    log_file_path = os.path.join(
        summary_dir, 'log_{}_{}'.format(abr, os.path.basename(trace_file)))
    agent_id = os.path.basename(summary_dir).split("_")[1]

    with tf.Session() as sess ,open( log_file_path ,'wb' ) as log_file:

        actor = ActorNetwork( sess ,
                              state_dim=[6 ,6] ,action_dim=6 ,
                              bitrate_dim=6)

        sess.run( tf.initialize_all_variables() )
        saver = tf.train.Saver()  # save neural net parameters

        # restore neural net parameters
        nn_model = actor_path
        if nn_model is not None:  # nn_model is the path to file
            saver.restore( sess ,nn_model )
            #print( "Model restored." )

        if abr == 'RobustMPC':
            abr = RobustMPC()
        elif abr == 'FastMPC':
            abr = FastMPC()
        elif abr == 'RL':
            # assert actor_path is not None, "actor-path is needed for RL abr."
            abr = Pensieve(16, summary_dir, actor=actor)
        elif abr == 'BufferBased':
            abr = BufferBased()
        elif abr == 'RLTrain':
            abr = RLTrain()
        else:
            raise ValueError("ABR {} is not supported!".format(abr))

        video_size = construct_bitrate_chunksize_map(video_size_file_dir)
        np.random.seed(RANDOM_SEED)

        assert len(VIDEO_BIT_RATE) == A_DIM

        # interface to abr_rl server

        log_writer = csv.writer(open(log_file_path, 'w', 1), delimiter='\t',
                                lineterminator='\n')
        log_writer.writerow(
            ['timestamp', 'bit_rate', 'buffer_size', 'rebuffer_time',
             'video_chunk_size', 'download_time', 'reward',
             'bandwidth_estimation','future_bandwidth'])

        # variables and states needed to track among requests
        server_states = {
            'sess': sess,
            'actor': actor,
            'log_writer': log_writer,
            'abr': abr,
            'video_size': video_size,
            'video_chunk_count': 0,
            "last_total_rebuf": 0,
            'last_bit_rate': DEFAULT_QUALITY,
            'state': np.zeros((1, S_INFO, S_LEN)),
            'future_bandwidth': 0,
            'summary_dir': summary_dir,
            'agent_id': agent_id,
            # <-- Add these two if you're controlling chunk step externally
            # 'chunk_req_queue': chunk_req_queue,
            # 'chunk_resp_queue': chunk_resp_queue,
        }

        handler_class = make_request_handler(server_states)
        server_address = (ip, port)
        httpd = HTTPServer(server_address, handler_class)
        print('Listening on ({}, {})'.format(ip, port))
        httpd.serve_forever()

# def run_training_server(trace_file, summary_dir, actor_path,
#                    video_size_file_dir, ip='localhost', port=8333):
#     os.makedirs(summary_dir, exist_ok=True)
#     log_file_path = os.path.join(
#         args.summary_dir, 'log_{}_{}'.format("RL", os.path.basename(trace_file)))

#     # Set up multiprocessing for training
#     exp_queues = []
#     net_params_queues = []
#     for _ in range(args.num_agents):
#         exp_queues.append(mp.Queue(1))
#         net_params_queues.append(mp.Queue(1))

#     with tf.Session() as sess, open(log_file_path, 'wb') as log_file:
#         # Initialize networks
#         actor = ActorNetwork(sess,
#                             state_dim=[S_INFO, S_LEN],
#                             action_dim=A_DIM,
#                             bitrate_dim=len(VIDEO_BIT_RATE))

#         sess.run(tf.global_variables_initializer())
#         saver = tf.train.Saver()

#         # Initialize ABR algorithm
#         abr = Pensieve(args.num_agents, args.summary_dir, 
#             actor=actor, model_save_interval=args.model_save_interval)

#         # Setup server states
#         video_size = construct_bitrate_chunksize_map(args.video_size_file_dir)
#         np.random.seed(RANDOM_SEED)

#         log_writer = csv.writer(open(log_file_path, 'w', 1),
#                               delimiter='\t', lineterminator='\n')
#         log_writer.writerow([
#             'timestamp', 'bit_rate', 'buffer_size', 'rebuffer_time',
#             'video_chunk_size', 'download_time', 'reward',
#             'bandwidth_estimation', 'future_bandwidth'
#         ])

#         server_states = {
#             'sess': sess,
#             'actor': actor,
#             'log_writer': log_writer,
#             'abr': abr,
#             'video_size': video_size,
#             'video_chunk_count': 0,
#             'last_total_rebuf': 0,
#             'last_bit_rate': DEFAULT_QUALITY,
#             'state': np.zeros((1, S_INFO, S_LEN)),
#             'future_bandwidth': 0,
#             'training': True,
#             'batch_size': 100,
#             'exp_queue': exp_queues[0]  # Use first queue for now
#         }

#         # Start training process
#         train_proc = mp.Process(video_sizetarget=run_training_loop, 
#                                args=(abr, net_params_queues, exp_queues, args))
#         train_proc.start()

#         # Start server
#         handler_class = make_request_handler(server_states)
#         server_address = (args.ip, args.port)
#         httpd = HTTPServer(server_address, handler_class)
#         print(f'Training server listening on ({args.ip}, {args.port})')
        
#         try:
#             httpd.serve_forever()
#         except KeyboardInterrupt:
#             print("\nStopping training...")
#             train_proc.terminate()
#             train_proc.join()
#             print("\nSaving model before exit...")
#             saver.save(sess, os.path.join(args.summary_dir, 'model_final.ckpt'))
#             print("Model saved. Exiting...")

# def run_training_loop(abr, net_params_queues, exp_queues, args):
#     """Separate process that runs the training loop."""
    
#     # Create environments for validation
#     val_envs = []  # You'll need to create validation environments
    
#     try:
#         # Call the actual training function
#         abr.train(
#             train_envs=None,  # Not used in emulation mode
#             val_envs=val_envs,
#             iters=args.num_epocvideo_sizes,
#             net_params_queues=net_params_queues,
#             exp_queues=exp_queues
#         )
#     except Exception as e:
#         print(f"Training error: {e}")
#         raise

def main():
    args = parse_args()
    run_abr_server(args.abr, args.trace_file, args.summary_dir,
                   args.actor_path, args.video_size_file_dir, args.ip,
                   args.port)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Capture Keyboard interrupted.")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
