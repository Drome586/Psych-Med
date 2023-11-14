import os
import json
from itertools import chain

import torch
from torch.utils.data import (Dataset, DataLoader)
from torch.nn.utils.rnn import pad_sequence


class WBDataset(Dataset):

    def __init__(self, data, tokenizer, max_history=15, batch_first=True, lm_labels=True):
        self.data = data
        self.tokenizer = tokenizer
        self.max_history = max_history
        self.pad = tokenizer.pad_token_id
        self.batch_first = batch_first
        self.lm_labels = lm_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.lm_labels:
            history = self.data[index][-2 * self.max_history:-1]
            response = self.data[index][-1]
        else:
            history = self.data[index][-2 * self.max_history:-1]
            response = []
        return self.process(history, response)

    def process(self, history, response):
        bos, eos = self.tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]"])

        encoder_input = [bos]
        for his in history:
            encoder_input += his + [eos]
        decoder_input = [bos] + response + [eos]

        instance = {}
        instance["input_ids"] = encoder_input
        instance["output_ids"] = decoder_input
        instance["input_mask"] = [1] * len(encoder_input)
        instance["output_mask"] = [1] * len(decoder_input)

        return instance

    def collate(self, batch):
        input_ids = pad_sequence(
            [torch.tensor(instance["input_ids"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=self.pad)
        output_ids = pad_sequence(
            [torch.tensor(instance["output_ids"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=self.pad)
        input_mask = pad_sequence(
            [torch.tensor(instance["input_mask"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=self.pad)
        output_mask = pad_sequence(
            [torch.tensor(instance["output_mask"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=self.pad)
        return input_ids, output_ids, input_mask, output_mask


def get_data(tokenizer, dataset_path='data/data.json'):
    with open(dataset_path, "r", encoding='utf-8') as f:
        data = json.loads(f.read())

    def tokenize(obj):
        if isinstance(obj, str):
            return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
        if isinstance(obj, dict):
            return dict((n, tokenize(o)) for n, o in obj.items())
        return list(tokenize(o) for o in obj)
    
    data = tokenize(data)
    return data


def build_train_dataloaders(tokenizer, batch_size):
    datasets = get_data(tokenizer)
    train_dataset= WBDataset(datasets["train"], tokenizer)
    train_loader = DataLoader(train_dataset, sampler=None, collate_fn=train_dataset.collate,
                              num_workers=8, batch_size=batch_size, shuffle=True)
    return train_dataset, train_loader


def build_valid_dataloaders(tokenizer, batch_size):
    datasets = get_data(tokenizer)
    valid_dataset = WBDataset(datasets["valid"], tokenizer)
    valid_loader = DataLoader(valid_dataset, sampler=None, collate_fn=valid_dataset.collate,
                              num_workers=8, batch_size=batch_size, shuffle=False)
    return valid_dataset, valid_loader


def build_test_dataloaders(tokenizer, batch_size):
    datasets = get_data(tokenizer)
    test_dataset = WBDataset(datasets["test"], tokenizer)
    test_loader = DataLoader(test_dataset, sampler=None, collate_fn=test_dataset.collate,
                              num_workers=8, batch_size=batch_size, shuffle=True)
    return test_dataset, test_loader
