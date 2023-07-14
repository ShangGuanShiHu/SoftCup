import os

import pandas as pd

def load_raw_data(args):
    raw_data_dir = os.path.join(args.data_dir, args.dataset, 'raw')
    train_raw_data = pd.read_csv(os.path.join(raw_data_dir, 'train.csv'))
    # test_raw_data = pd.read_csv(os.path.join(raw_data_dir, 'test_2000_x.csv'))
    test_raw_data = pd.read_csv(os.path.join(raw_data_dir, 'test.csv'))
    return train_raw_data, test_raw_data