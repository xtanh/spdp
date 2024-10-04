import math
import functools

import torch
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from dataset import SkempiDataset

DEFAULT_PAD_VALUES = {
    'aa': 21, 
    'chain_nb': -1, 
    'chain_id': ' ', 
}

class PaddingCollate(object):
    def __init__(self, length_ref_key='aa', pad_values=DEFAULT_PAD_VALUES):
        super().__init__()
        self.length_ref_key = length_ref_key
        self.pad_values = pad_values

    @staticmethod
    def _pad_last(x, n, value=0):
        if isinstance(x, torch.Tensor):
            assert x.size(0) <= n
            if x.size(0) == n:
                return x
            pad_size = [n - x.size(0)] + list(x.shape[1:])
            pad = torch.full(pad_size, fill_value=value).to(x)
            return torch.cat([x, pad], dim=0)
        elif isinstance(x, list):
            pad = [value] * (n - len(x))
            return x + pad
        else:
            return x

    @staticmethod
    def _get_common_keys(list_of_dict):
        keys = set(list_of_dict[0].keys())
        for d in list_of_dict[1:]:
            keys = keys.intersection(d.keys())
        return keys

    def _get_pad_value(self, key):
        return self.pad_values.get(key, 0)

    def __call__(self, data_list):
        # 过滤掉 None 的数据
        data_list = [data for data in data_list if data is not None]

        # 如果所有数据都被过滤掉，返回 None
        if not data_list:
            print("Warning: All data in batch are None.")
            return None

        try:
            max_length = max([data[self.length_ref_key].size(0) for data in data_list])
        except KeyError:
            print(f"KeyError: Key '{self.length_ref_key}' not found in one of the data items.")
            return None
        except ValueError:
            print("ValueError: One of the data items is empty.")
            return None

        max_length = math.ceil(max_length / 8) * 8
        keys = self._get_common_keys(data_list)

        exclude_keys = {'esm_embeddings_wt','esm_embeddings_mut'}

        data_list_padded = []
        for data in data_list:
            if data is None:
                continue  # 再次检查并跳过 None 数据

            data_padded = {}
            for k, v in data.items():
                    if k in exclude_keys:
                        data_padded[k] = v  # 保持原样，不填充
                    else:
                        data_padded[k] = self._pad_last(v, max_length, value=self._get_pad_value(k))
            data_list_padded.append(data_padded)

        # 如果经过填充的数据仍然是空的，返回 None
        if not data_list_padded:
            print("Warning: No valid data to collate after padding.")
            return None

        # 过滤掉任何仍包含 None 的数据
        data_list_padded = [d for d in data_list_padded if all(v is not None for v in d.values())]

        if not data_list_padded:
            print("Warning: All data in batch are invalid after filtering.")
            return None

        batch = default_collate(data_list_padded)
        return batch



def inf_iterator(iterable):
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


class SkempiDatasetManager(object):
    def __init__(self,batch_size,num_workers=4):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.dataset = SkempiDataset(
            csv_path="/home/hongtan/pretrain_single/dataset/fireprot/output_file.csv",
            pdb_dir="/home/hongtan/pretrain_single/dataset/final_dataset/pdb",
            cache_dir="/home/hongtan/pretrain_single/dataset/final_dataset/cache",
            patch_size=64
        )

        self.pretrain_loader = DataLoader(
            self.dataset, 
            batch_size=self.batch_size,
            shuffle=True, 
            collate_fn=PaddingCollate(), 
            num_workers=self.num_workers
        )



    def get_pretrain_loader(self):
        return self.pretrain_loader
