from tqdm import tqdm

import torch
from core.preprocess.feature_extractor import *
from torch.utils.data import Dataset


class SoftCupDataset(Dataset):
    def __init__(self, data: DataFrame):
        super().__init__()
        self.data = self.to_data_list(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        # 在这里进行数据预处理、转换等操作
        return item

    def to_data_list(self, datas: DataFrame):
        data_list = []
        features = [col for col in datas.columns if col != 'sample_id' and col != 'label']
        for idx, data in tqdm(datas.iterrows()):
            # 故障类型
            ft = data['label']
            # 样本id
            sample_id = data['sample_id']

            # 故障特征feature0-feature106
            feature_data = torch.tensor(np.array(data[features].values, dtype=float),dtype=torch.float32)

            label = int(ft)
            tmp_data = (sample_id, feature_data, label)
            data_list.append(tmp_data)
        return data_list




