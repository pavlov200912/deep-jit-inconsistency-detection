import torch
from data_utils import DiffAST, DiffExample, DiffASTExample, CommentCategory
import os
from jit_constants import DATA_PATH
from data_loader import load_raw_data_from_path
import sys

sys.path.append('comment_update')


def get_collate(batch_transform=None):
    def mycollate(batch):
        if batch_transform is not None:
            collated = batch_transform(batch)
            return collated
        else:
            return batch
    return mycollate


class Dataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, partition):
        'Initialization'
        self.examples = []
        comment_types = [CommentCategory(category).name for category in
                         CommentCategory]
        for comment_type in comment_types:
            path = os.path.join(DATA_PATH, comment_type)
            loaded = load_raw_data_from_path(path)

            self.examples.extend(loaded[partition])

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]
