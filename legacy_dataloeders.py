from dataclasses import dataclass, field
from functools import cached_property
import logging
from typing import Dict, List, Optional, Union

import numpy as np
import polars as pl
import torch

from torch.utils.data import DataLoader


@dataclass
class Data:
    train: pl.LazyFrame
    validation: pl.LazyFrame
    test: pl.LazyFrame
    item_id_to_idx: Dict[int, int]

    _train_user_ids: Optional[torch.Tensor] = field(init=False, default=None)

    @property
    def num_items(self) -> int:
        return len(self.item_id_to_idx)

    @cached_property
    def num_train_users(self) -> int:
        return self.train.select(pl.len()).collect(engine="streaming").item()

    def train_user_ids(self, device: torch.device) -> torch.Tensor:
        if self._train_user_ids is None or self._train_user_ids.device != device:
            self._train_user_ids = (
                self.train
                .select('uid')
                .collect(engine="streaming")['uid']
                .to_torch()
                .to(device)
            )
        return self._train_user_ids


class EvalDataset:
    def __init__(self, dataset: pl.DataFrame, max_seq_len: int):
        self._dataset = dataset
        self._max_seq_len = max_seq_len

    @property
    def dataset(self) -> pl.DataFrame:
        return self._dataset

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> Dict[str, Union[List[int], int]]:
        sample = self._dataset.row(index, named=True)

        item_sequence = sample['item_id_train'][-self._max_seq_len:]
        next_items = sample['item_id_valid']

        return {
            'user.ids': [sample['uid']],
            'user.length': 1,
            'item.ids': item_sequence,
            'item.length': len(item_sequence),
            'labels.ids': next_items,
            'labels.length': len(next_items),
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    processed_batch: Dict[str, List[int]] = {}

    for key in batch[0].keys():
        if key.endswith('.ids'):
            prefix = key.split('.')[0]
            assert f'{prefix}.length' in batch[0]

            processed_batch[f'{prefix}.ids'] = []
            processed_batch[f'{prefix}.length'] = []

            for sample in batch:
                processed_batch[f'{prefix}.ids'].extend(sample[f'{prefix}.ids'])
                processed_batch[f'{prefix}.length'].append(sample[f'{prefix}.length'])

    for part, values in processed_batch.items():
        processed_batch[part] = torch.tensor(values, dtype=torch.long)

    return processed_batch


logger = logging.getLogger(__name__)


class TrainDataset:
    def __init__(
        self,
        dataset: pl.DataFrame,
        num_items: int,
        num_neg_items: int,
        max_seq_len: int
    ):
        self._dataset = dataset
        self._num_items = num_items
        self._max_seq_len = max_seq_len
        self._num_neg_items = num_neg_items

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> Dict[str, Union[List[int], int]]:
        sample = self._dataset.row(index, named=True)

        item_sequence = sample['item_id'][:-1][-self._max_seq_len:]
        positive_sequence = sample['item_id'][1:][-self._max_seq_len:]

        ret_dict = {
            'user.ids': [sample['uid']],
            'user.length': 1,
            'item.ids': item_sequence,
            'item.length': len(item_sequence),
            'positive.ids': positive_sequence,
            'positive.length': len(positive_sequence),
        }

        if self._num_neg_items > 0:
            negative_sequence = np.random.randint(
                1,
                self._num_items + 1,
                size=(len(item_sequence), self._num_neg_items)
            ).tolist()

            ret_dict.update({
                'negative.ids': negative_sequence,
                'negative.length': len(negative_sequence),
            })

        return ret_dict


class EvalDatasetGTS(EvalDataset):
    def __init__(
        self,
        dataset: pl.DataFrame,
        max_seq_len: int,
        seed: int = 42,
        mode: str = 'random'
    ):
        super().__init__(dataset, max_seq_len)
        self.seed = seed
        self.mode = mode
        np.random.seed(seed)

        self.val_seq_len = np.array([len(seq) for seq in dataset['item_id_valid']])

        if mode == 'random':
            self.split_points = np.random.random(len(self))
            self.split_index = np.int32(np.ceil(self.val_seq_len * self.split_points))
        elif mode == 'last':
            self.split_index = np.ones_like(self.val_seq_len)
        elif mode == 'first':
            self.split_index = self.val_seq_len
        elif mode == 'successive':
            self.cum_sum_len = np.cumsum(self.val_seq_len)
        else:
            raise ValueError('undefined GTS evaluation mode')

    def __len__(self) -> int:
        if self.mode == 'successive':
            return int(self.val_seq_len.sum())
        return super().__len__()

    def __getitem__(self, index: int) -> Dict[str, Union[List[int], int]]:
        if self.mode == 'successive':
            user_index = np.searchsorted(self.cum_sum_len, index, side='right')
            val_index = index - self.cum_sum_len[user_index - 1] if user_index > 0 else index
            split_index = self.val_seq_len[user_index] - val_index
        else:
            user_index = index
            split_index = self.split_index[index]

        sample = self._dataset.row(user_index, named=True)

        train_items = sample['item_id_train']
        holdout_items = sample['item_id_valid']
        all_items = train_items + holdout_items

        item_sequence = all_items[-split_index - self._max_seq_len:-split_index]
        next_items = [all_items[-split_index]]
        seen_items = all_items[:-split_index]

        return {
            'user.ids': [sample['uid']],
            'user.length': 1,
            'item.ids': item_sequence,
            'item.length': len(item_sequence),
            'labels.ids': next_items,
            'labels.length': len(next_items),
            'seen.ids': seen_items,
            'seen.length': len(seen_items),
        }


class GPUSASRecDataloader:
    def __init__(self, base_loader: DataLoader, device: torch.device):
        self.device = device
        self.base_loader = base_loader
        self.batch_size = base_loader.batch_size

        all_items, all_users, all_lengths = [], [], []
        for batch in base_loader:
            all_items.append(batch['positive.ids'])
            all_lengths.append(batch['positive.length'])
            all_users.append(batch['user.ids'])

        self.items = torch.cat(all_items).long().to(device)
        self.users = torch.cat(all_users).long().to(device)
        self.lengths = torch.cat(all_lengths).long().to(device)

        self.cum_len = torch.cat([
            torch.zeros(1, device=device, dtype=torch.long),
            torch.cumsum(self.lengths, dim=0)
        ])

        self.seq_ids = torch.searchsorted(
            self.cum_len[1:],
            torch.arange(len(self.items), device=device),
            right=True
        )

        self._batch_index = 0
        self._permutation = None
        self._reshuffle()

    def _reshuffle(self) -> None:
        self._permutation = torch.randperm(len(self.lengths), device=self.device)
        self._batch_index = 0

    def __len__(self) -> int:
        return len(self.base_loader)

    def __iter__(self):
        self._reshuffle()
        return self

    def __next__(self) -> Dict[str, torch.Tensor]:
        if self._batch_index >= len(self):
            raise StopIteration

        start = self._batch_index * self.batch_size
        end = (self._batch_index + 1) * self.batch_size
        sampled_indices = self._permutation[start:end].sort().values

        mask = torch.isin(self.seq_ids, sampled_indices)

        input_mask = mask.clone()
        input_mask[self.cum_len[sampled_indices + 1] - 1] = False

        output_mask = mask.clone()
        output_mask[self.cum_len[sampled_indices]] = False

        input_items = self.items[input_mask]
        output_items = self.items[output_mask]

        sampled_lengths = self.lengths[sampled_indices] - 1
        sampled_users = self.users[sampled_indices]

        self._batch_index += 1

        return {
            'user.ids': sampled_users,
            'item.ids': input_items,
            'item.length': sampled_lengths,
            'positive.ids': output_items,
            'positive.length': sampled_lengths,
        }
