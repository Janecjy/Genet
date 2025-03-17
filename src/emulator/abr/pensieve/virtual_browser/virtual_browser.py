import argparse
from json import dumps
import multiprocessing as mp
import os
import signal
from time import sleep
from urllib.parse import ParseResult, parse_qsl, unquote, urlencode, urlparse
import subprocess
import logging
from pyvirtualdisplay import Display
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
import redis
from  pensieve.agent_policy import rl_embedding

redis_client = redis.Redis(host="10.10.1.2", port=2666, decode_responses=True)

from pensieve.virtual_browser.abr_server import run_abr_server


def setup_logger(logger_name, log_file, level=logging.INFO):
    """Create and return a logger with a file handler."""
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    # Avoid adding multiple handlers if logger already exists (e.g. multiprocess)
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


def parse_args():
    """Parse arguments from the command line."""
    parser = argparse.ArgumentParser("Virtual Browser")
    parser.add_argument('--description', type=str, default=None,
                        help='Optional description of the experiment.')

    # ABR related
    parser.add_argument('--abr', type=str, required=True,
                        choices=['RobustMPC', 'RL', 'FastMPCchromedriver'
                                 'Default', 'FixedRate',
                                 'BufferBased', 'RateBased', 'Festive',
                                 'Bola', 'RLTrain'], help='ABR algorithm.')
    parser.add_argument('--actor-path', type=str, default=None,
                        help='Path to RL model.')
    parser.add_argument('--original-model-path', type=str, default=None,
                        help='Path to original RL model.')
    parser.add_argument('--adaptor-input', type=str, default=None,
                        help='Type of adaptor input.')
    parser.add_argument('--adaptor-hidden-size', type=int, default=128,
                        help='Hidden size of adaptor.')

    # data io related
    parser.add_argument('--summary-dir', type=str, required=True,
                        help='directory to save logs.')
    parser.add_argument('--trace-file', type=str, required=True,
                        help='Path to trace file.')
    parser.add_argument("--video-size-file-dir", type=str, required=True,
                        help='Dir to video size files')
    parser.add_argument('--run_time', type=int, default=240000,
                        help="Running time.")

    # networking related
    parser.add_argument('--ip', type=str, help='IP of HTTP video server.')
    parser.add_argument('--port', type=int,
                        help='Port number of HTTP video server.')
    parser.add_argument('--abr-server-ip', type=str, default='localhost',
                        help='IP of ABR server.')
    parser.add_argument('--abr-server-port', type=int, default=8333,
                        help='Port number of ABR server.')

    parser.add_argument('--buffer-threshold', type=int, default=60,
            help='Buffer threshold of Dash.js MediaPlayer. Unit: Second.')

    # New training arguments
    parser.add_argument('--train', action='store_true',
                        help='Enable training mode')
    parser.add_argument('--num-agents', type=int, default=16,
                        help='Number of training agents')
    parser.add_argument('--model-save-interval', type=int, default=100,
                        help='Save model every N iterations')
    parser.add_argument('--num-epochs', type=int, default=100000,
                        help='Number of training epochs')
    parser.add_argument('--use_embedding', action='store_true',
                        help='Use embedding during action prediciton')   
    
    return parser.parse_args()


def add_url_params(url, params):
    """Add GET params to provided URL being aware of existing.

    url = 'http://stackoverflow.com/test?answers=true'
    new_params = {'answers': False, 'data': ['some','values']}
    add_url_params(url, new_params)
    'http://stackoverflow.com/test?data=some&data=values&answers=false'

    Args
        url: string of target URL
        params: dict containing requested params to be added

    Return
        string with updated URL
    """
    # Unquoting URL first so we don't loose existing args
    url = unquote(url)
    # Extracting url info
    parsed_url = urlparse(url)
    # Extracting URL arguments from parsed URL
    get_args = parsed_url.query
    # Converting URL arguments to dict
    parsed_get_args = dict(parse_qsl(get_args))
    # Merging URL arguments dict with new params
    parsed_get_args.update(params)

    # Bool and Dict values should be converted to json-friendly values
    # you may throw this part away if you don't like it :)
    parsed_get_args.update(
        {k: dumps(v) for k, v in parsed_get_args.items()
         if isinstance(v, (bool, dict))}
    )

    # Converting URL argument to proper query string
    encoded_get_args = urlencode(parsed_get_args, doseq=True)
    # Creating new parsed result object based on provided with new
    # URL arguments. Same thing happens inside of urlparse.
    new_url = ParseResult(
        parsed_url.scheme, parsed_url.netloc, parsed_url.path,
        parsed_url.params, encoded_get_args, parsed_url.fragment
    ).geturl()

    return new_url


def timeout_handler(signum, frame):
    raise Exception("Timeout")

def launch_bpftrace(trace_output_file):
    """Launch bpftrace script and return the process."""
    cmd = "sudo bpftrace check.bt > bpftrace_output.txt"
    
    # with open(trace_output_file, 'w') as f:
    process = subprocess.Popen(cmd, shell=True, stderr=subprocess.PIPE)
    check_interval = 10

    # Start log rotation process
#     rotation_cmd = f"watch -n {check_interval} python3 -c '\
# import sys; from collections import deque; \
# lines = deque(open(\"{trace_output_file}\", \"r\"), maxlen=40000); \
# open(\"{trace_output_file}\", \"w\").writelines(lines)'"
    
#     rotation_process = subprocess.Popen(rotation_cmd, shell=True)

    return process

def main():
    args = parse_args()

    # Derive agent_id from summary dir (e.g. "pensieve_5_...")
    agent_id = os.path.basename(args.summary_dir).split("_")[1]
    print(agent_id)
    # Set up a logger that writes to a file with the agent_id in the name
    logger_name = f"{agent_id}_virtual_browser"
    log_file = f"/mydata/logs/{agent_id}_virtual_browser.log"
    logger = setup_logger(logger_name, log_file)

    redis_client.ping()
    logger.info("Redis ping successful")

    ip = args.ip
    port_number = args.port
    abr_algo = args.abr
    run_time = args.run_time
    embedding = None
    tokens = None
    if args.use_embedding:
        embedding, tokens = rl_embedding.null_embedding_and_token()
        video_server_proc, bpftrace_process = rl_embedding.launch_video_server_and_bftrace(agent_id, logger, run_video_server=False)
    else:
        # Start bpftrace before ABR server
        trace_output = "bpftrace_output.txt"
        bpftrace_process = launch_bpftrace(trace_output)
    
    logger.info("Starting ABR server in inference mode...")
    logger.info(f"Summary dir: {args.summary_dir}")

    # Launch the ABR server
    abr_server_proc = mp.Process(
        target=run_abr_server,
        args=(
            abr_algo,
            args.trace_file,
            args.summary_dir,
            args.actor_path,
            args.video_size_file_dir,
            args.abr_server_ip,
            args.abr_server_port,
            args.original_model_path,
            args.adaptor_input,
            args.adaptor_hidden_size,
            embedding,
            tokens,
        )
    )
    abr_server_proc.start()
    sleep(0.5)

    # abr_server_proc = mp.Process(target=run_abr_server, args=(
    #     abr_algo, args.trace_file, args.summary_dir, args.actor_path,
    #     args.video_size_file_dir, args.abr_server_ip, args.abr_server_port))
    # abr_server_proc.start()

    # sleep(0.5)

    # generate url
    url = 'http://{}:{}/index.html'.format(ip, port_number)
    # url_params = {'abr_id': ABR_ID_MAP[abr_algo]}
    url_params = {'abr_id': abr_algo,
                  'buffer_threshold': args.buffer_threshold,
                  'port': args.abr_server_port}
    url = add_url_params(url, url_params)
    logger.info(f"Open {url}")

    # ip = json.loads(urlopen("http://ip.jsontest.com/").read().decode('utf-8'))['ip']
    # url = 'http://{}/myindex_{}.html'.format(ip, abr_algo)
    redis_client.set(f"{agent_id}_new_epoch", 0)

    # timeout signal
    signal.signal(signal.SIGALRM, timeout_handler)
    # original code set up timeout if test not finished in run_time+30 seconds
    #signal.alarm(run_time + 30) 
    display = None
    driver = None
    try:
        # copy over the chrome user dir
        default_chrome_user_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'abr_browser_dir/chrome_data_dir')
        # chrome_user_dir = '/tmp/chrome_user_dir_id_' + process_id
        chrome_user_dir = '/tmp/chrome_user_dir'  # + process_id
        # os.system('rm -r ' + chrome_user_dir)
        os.system('cp -r ' + default_chrome_user_dir + ' ' + chrome_user_dir)

        # to not display the page in browser
        display = Display(visible=False, size=(800, 600))
        display.start()

        # initialize chrome driver
        options = Options()
        chrome_driver = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'abr_browser_dir/chromedriver')
        # options.add_argument('--user-data-dir=' + chrome_user_dir)
        # enable browser logging
        options.add_argument("--headless")
        options.add_argument("--disable-extensions")
        options.add_argument('--ignore-certificate-errors')
        options.add_argument( "--disable-web-security" )
        options.add_argument( "--disable-site-isolation-trials" )
        desired_caps = DesiredCapabilities.CHROME
        desired_caps['goog:loggingPrefs'] = {'browser': 'ALL'}
        #import pdb; pdb.set_trace()
        driver = webdriver.Chrome(chrome_driver, options=options,
                                  desired_capabilities=desired_caps)

        # run chrome
        #num_epochs = 75000
        num_epochs = args.num_epochs
        driver.set_page_load_timeout(10)
        redis_client.set(f"{agent_id}_browser_active", 1)
        redis_client.set(f"{agent_id}_stop_flag", int(False))
        driver.get(url)
        count = 1
        while count < num_epochs:
            sleep(10)
            logger.info(f"{agent_id} waiting for new epoch")
            new_epoch = redis_client.get(f"{agent_id}_new_epoch")
            logger.info(f"{agent_id} new_epoch: {new_epoch}")
            if new_epoch and int(new_epoch) == 1:
                count += 1
                logger.info(f"{agent_id} get new url with count: {count}")
                try:
                    logger.info(f"{agent_id} get url succeeded")
                    redis_client.set(f"{agent_id}_new_epoch", 0)
                    redis_client.set(f"{agent_id}_browser_active", 1)
                    logger.info(f"{agent_id} set new_epoch to 0")
                    driver.get(url)
                except Exception as e:
                    logs = driver.get_log('browser')
                    print("Browser logs on exception:\n", logs)
                    logger.info(f"Browser logs on exception:\n")
                    raise e

        if args.train:
            logger.info("Video streaming started. Training in progress...")
            logger.info(f"Will run for {run_time} seconds")
            logger.info(f"Model checkpoints saved every {args.model_save_interval} epochs")
            logger.info(f"Training logs will be saved to {args.summary_dir}")

        sleep(run_time)

        if args.train:
            abr_server_proc.join()
        if run_time == 0:
            stop_flag = redis_client.get(f"{agent_id}_stop_flag")
            while(stop_flag and int(stop_flag) == 0):
                sleep(60)
                stop_flag = redis_client.get(f"{agent_id}_stop_flag")

        driver.quit()
        display.stop()

        logger.info("done")
        abr_server_proc.terminate()

    except Exception as e:
        logger.exception("Exception in main loop")
        if display is not None:
            display.stop()
        if driver is not None:
            driver.quit()
        # try:
        #     proc.send_signal(signal.SIGINT)
        # except:
        #     pass
        print(e)
    finally:
        bpftrace_process.terminate()
        # rotation_process.terminate()
        abr_server_proc.terminate()
    abr_server_proc.terminate()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Keyboard Interrupted! Virtual browser exits!')
