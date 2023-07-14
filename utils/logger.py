import logging
import os

def get_logger(save_dir, task_name):
    LOG_FILE_NAME = '{}.log'.format(task_name)

    logger = logging.getLogger(task_name)
    logging.basicConfig(
        level=logging.DEBUG,
        filename=os.path.join(save_dir, LOG_FILE_NAME),
        filemode='a+',
        format='%(asctime)s [%(levelname)s]: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logger