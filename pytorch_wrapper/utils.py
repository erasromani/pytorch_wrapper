import os
import datetime
import dataclasses
import gin
import random
import argh
import logging
import sys
import torch
import numpy as np

from pathlib import Path
from contextlib import contextmanager
from logging import handlers

logger = logging.getLogger(__name__)

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError(f'{s} is not a valid boolean string')
    return s == 'True'

def get_output_dir(parent_dir):
    today = datetime.date.today()
    date = today.strftime('%Y-%m-%d')
    now = datetime.datetime.now()
    time = now.strftime('%H-%M-%S')
    output_dir = os.path.join(parent_dir, date, time)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info("Creating folder {}".format(output_dir))
    return output_dir

def split_dataset(split):
    pass

def show_sample():
    pass

def load_text_data(path, split_by=None):
    with open(path, "r") as f:
        data = f.read()
    if split_by is None:
        return data
    else:
        return data.split(split_by)

def set_seed(seed=None):
    # set seed
    if seed is None:
        seed = np.random.randint(low=0, high=99999)
    logger.info("seed = {}".format(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

def get_device(index=0):
    return torch.device("cuda:{}".format(index) if torch.cuda.is_available() else "cpu")

def repr_torchdict(torchdict):
    for i, (key, value) in enumerate(torchdict.items()):
        if i == 0:
            out = "{}={:.4f}".format(key, value) 
        else:
            out = "{}, {}={:.4f}".format(out, key, value)
    return out    

class Fork(object):
    def __init__(self, file1, file2):
        self.file1 = file1
        self.file2 = file2

    def write(self, data):
        self.file1.write(data)
        self.file2.write(data)

    def flush(self):
        self.file1.flush()
        self.file2.flush()


@contextmanager
def replace_logging_stream(file_):
    root = logging.getLogger()
    if len(root.handlers) != 1:
        logger.error(root.handlers)
        raise ValueError("Don't know what to do with many handlers")
    if not isinstance(root.handlers[0], logging.StreamHandler):
        raise ValueError
    stream = root.handlers[0].stream
    root.handlers[0].stream = file_
    try:
        yield
    finally:
        root.handlers[0].stream = stream


@contextmanager
def replace_standard_stream(stream_name, file_):
    stream = getattr(sys, stream_name)
    setattr(sys, stream_name, file_)
    try:
        yield
    finally:
        setattr(sys, stream_name, stream)

def run_with_redirection(stdout_path, stderr_path, func):
    def func_wrapper(*args, **kwargs):
        with open(stdout_path, 'a', 1) as out_dst:
            with open(stderr_path, 'a', 1) as err_dst:
                out_fork = Fork(sys.stdout, out_dst)
                err_fork = Fork(sys.stderr, err_dst)
                with replace_standard_stream('stderr', err_fork):
                    with replace_standard_stream('stdout', out_fork):
                        with replace_logging_stream(err_fork):
                            func(*args, **kwargs)
    return func_wrapper

def gin_wrap(fnc):
    def main(save_path, config, bindings=""):
        # You can pass many configs (think of them as mixins), and many bindings. Both ";" separated.
        gin.parse_config_files_and_bindings(config.split("#"), bindings.replace("#", "\n"))
        save_path = get_output_dir(save_path)
        os.system("cp " + config + " " + os.path.join(save_path, "config.gin"))
        os.system("cp " + sys.argv[0] + " " + os.path.join(save_path, Path(sys.argv[0]).name))
        run_with_redirection(os.path.join(save_path, "stderr.txt"),
                        os.path.join(save_path, "stdout.txt"),
                        fnc)(save_path)
    argh.dispatch_command(main)

def configure_logger(name='',
        console_logging_level=logging.INFO,
        file_logging_level=None,
        log_file=None):
    """
    Configures logger
    :param name: logger name (default=module name, __name__)
    :param console_logging_level: level of logging to console (stdout), None = no logging
    :param file_logging_level: level of logging to log file, None = no logging
    :param log_file: path to log file (required if file_logging_level not None)
    :return instance of Logger class
    """

    if file_logging_level is None and log_file is not None:
        logger.warning("Didnt you want to pass file_logging_level?")

    if len(logging.getLogger(name).handlers) != 0:
        logger.info("Already configured logger '{}'".format(name))
        return

    if console_logging_level is None and file_logging_level is None:
        return  # no logging

    logger = logging.getLogger(name)
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    if console_logging_level is not None:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(format)
        ch.setLevel(console_logging_level)
        logger.addHandler(ch)

    if file_logging_level is not None:
        if log_file is None:
            raise ValueError("If file logging enabled, log_file path is required")
        fh = handlers.RotatingFileHandler(log_file, maxBytes=(1048576 * 5), backupCount=7)
        fh.setFormatter(format)
        logger.addHandler(fh)

    logger.info("Logging configured!")

    return logger
