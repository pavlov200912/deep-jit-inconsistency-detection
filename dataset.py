import torch
import os
from jit_constants import DATA_PATH, AST_DATA_PATH
from data_loader import load_raw_data_from_path
import sys
from itertools import chain
import ijson

from data_utils import DiffAST, DiffExample, DiffASTExample, CommentCategory

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


class IterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, partition, ignore_ast=True):
        self.examples = []
        self.partition = partition
        self.ignore_ast = ignore_ast
        self.ast_files = []
        comment_types = [CommentCategory(category).name for category in
                         CommentCategory]
        for comment_type in comment_types:
            path = os.path.join(DATA_PATH, comment_type)
            loaded = load_raw_data_from_path(path)

            self.examples.extend(loaded[partition])

        print(f'{partition} examples loaded! Loading asts...')

        self.asts = []
        for comment_type in comment_types:
            path = os.path.join(AST_DATA_PATH, comment_type,
                                f'{partition}_ast.json')
            # todo: check for closed connection
            print(f'{comment_type}/{partition} asts loading...')
            ast_file = open(path, 'r')
            self.ast_files.append(ast_file)
            iter_ast = ijson.items(ast_file, 'item')
            self.asts.append(iter_ast)
        self.asts = chain(*self.asts)
        print(f'{partition} loaded!')

    def __iter__(self):
        if self.ignore_ast:
            return iter(self.examples)
        else:
            def ast_iter():
                for ex, ex_ast_info in zip(self.examples, self.asts):
                    old_ast = DiffAST.from_json(ex_ast_info['old_ast'])
                    new_ast = DiffAST.from_json(ex_ast_info['new_ast'])
                    diff_ast = DiffAST.from_json(ex_ast_info['diff_ast'])
                    ast_ex = DiffASTExample(ex.id, ex.label, ex.comment_type,
                                            ex.old_comment_raw,
                                            ex.old_comment_subtokens,
                                            ex.new_comment_raw,
                                            ex.new_comment_subtokens,
                                            ex.span_minimal_diff_comment_subtokens,
                                            ex.old_code_raw,
                                            ex.old_code_subtokens,
                                            ex.new_code_raw,
                                            ex.new_code_subtokens,
                                            ex.span_diff_code_subtokens,
                                            ex.token_diff_code_subtokens,
                                            old_ast, new_ast, diff_ast)

                    yield ast_ex

            return iter(ast_iter())

    def __len__(self):
        return len(self.examples)