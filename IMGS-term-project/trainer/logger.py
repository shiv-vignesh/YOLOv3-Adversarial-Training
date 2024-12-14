import logging
import os
import sys
from typing import *

from datetime import datetime

class Logger:
    '''
    logger class
    '''
    def __init__(self, trainer_kwargs):
        self.log = logging.getLogger()
        self.log.setLevel(logging.INFO)

        # add log to file handler
        time_now = str(datetime.now()).split('.')[0]
        log_file_path = os.path.join(trainer_kwargs["output_dir"], f"training_log_{time_now}.txt")
        handler = logging.FileHandler(log_file_path, 'w+', 'utf-8')
        handler.setFormatter(
            logging.Formatter(
                fmt = '%(asctime)s %(message)s',
                datefmt ='%m/%d/%Y %I:%M:%S %p'
            )
        )
        self.log.addHandler(handler)

        # add log to stdout handler
        self.log.addHandler(logging.StreamHandler(sys.stdout))

    def log_message(self, message: str, error: bool=False):
        '''
        logging a message to stdout and `output_dir` arg in config file

        `message`: message to log
        `error`: a boolean to indicate if this message is an error
        '''
        if error:
            self.log.error(message)
        else:
            self.log.info(message)

    def log_line(self):
        '''
        log a dash-line
        '''
        self.log.info("-" * 100)

    def log_new_line(self):
        '''
        log a new line
        '''
        self.log.info("")

    def log_block(self, message: str):
        '''
        log a dash-line plus a message
        '''
        self.log_line()
        self.log_message(message)