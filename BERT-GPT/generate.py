import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
import json

import fire
import time
import numpy as np
import torch.multiprocessing as mp

from tqdm import tqdm

from collections import defaultdict
# uses allennlp modules
from allennlp.nn import util

# imports chinese gpt
from chinese_gpt import TransformerEncoder, TransformerDecoderLM

# uses bert chinese wordpiece tokenization
# from pytorch_pretrained_bert import OpenAIAdam

from inputter import build_test_dataloaders


def top_k_logits(logits, k):
    """Mask logits so that only top-k logits remain
    """
    values, _ = torch.topk(logits, k)
    min_values = values[:, -1].unsqueeze(1).repeat(1, logits.shape[-1])
    return torch.where(logits < min_values, torch.ones_like(logits, dtype=logits.dtype) * -1e10, logits)


def get_decoder(decoder_path, gpu_id):
    old_state_dict = torch.load(decoder_path, map_location="cpu")
    print(f'load from {decoder_path}')
    encoder = TransformerEncoder()
    decoder = TransformerDecoderLM()

    encoder_state_dict = encoder.state_dict()
    for i in encoder_state_dict.keys():
        encoder_state_dict[i] = old_state_dict['encoder.' + i]
    encoder.load_state_dict(encoder_state_dict)

    decoder_state_dict = decoder.state_dict()
    for i in decoder_state_dict.keys():
        decoder_state_dict[i] = old_state_dict['decoder.' + i]
    decoder.load_state_dict(decoder_state_dict)

    return encoder, decoder


def generate_sentences(tokenizer, decoder_path, rank, top_k, l):
    # make sure your model is on GPU
    device = torch.device("cuda:1")

    #------------------------LOAD MODEL-----------------
    encoder, decoder = get_decoder(decoder_path, gpu_id=0)
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    encoder.eval()
    decoder.eval()

    _, test_dataloader = build_test_dataloaders(tokenizer, batch_size=1)
    #------------------------END LOAD VALIDATE DATA--------------


    #------------------------START SAMPLE GENERETE-------------------
    if rank == 0:
        test_dataloader = tqdm(test_dataloader)
    for batch in test_dataloader:
        with torch.no_grad():
            batch = [item.to(device) for item in batch]

            encoder_input, decoder_input, mask, _ = batch

            _, past = encoder(encoder_input, mask)

            sentence = []

            prev_pred = decoder_input[:, :1]
            sentence.append(prev_pred)

            length = 1
            # decoding loop
            for i in range(100):
                mask = F.pad(mask, (0, 1), "constant", 1.0)
                logits, past = decoder(prev_pred, mask, past=past, past_length=length)
                logits = logits.squeeze(1)
                logits = top_k_logits(logits, k=top_k)
                probs = F.softmax(logits, dim=-1)
                prev_pred = torch.multinomial(probs, num_samples=1)
                sentence.append(prev_pred)
                if prev_pred[0][0] == 102:
                    break
                length += 1

            sentence = torch.cat(sentence, dim=-1)
            predict = tokenizer.convert_ids_to_tokens(sentence[0].tolist())

            target = decoder_input.squeeze(dim=0)
            target_num = (target != 0).sum()
            reference = tokenizer.convert_ids_to_tokens(target[:target_num].tolist())

            encoder_input = encoder_input.squeeze(dim=0)
            encoder_input_num = (encoder_input != 0).sum()
            inputs = tokenizer.convert_ids_to_tokens(encoder_input[:encoder_input_num].tolist())
            l.append(["".join(inputs[1:-1]), "".join(predict[1:-1]), "".join(reference[1:-1])])


    #------------------------END SAMPLE GENERETE-------------------


def sample_generate(top_k = 50, decoder_path='pretrained/bertGPT_pretrained_model.pth', process_num=1):
    
    torch.multiprocessing.set_start_method('spawn')
    # test_data = torch.load("test_data.pth")
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

    # length = len(test_data[0])

    mgr = mp.Manager()
    l = mgr.list()
    processes = []
    # for rank in range(process_num):
    #     if rank == process_num - 1:
    #         data = [d[int((rank / process_num) * length):] for d in test_data]
    #     else:
    #         data = [d[int((rank / process_num) * length) : int(((rank + 1) / process_num) * length)] for d in test_data]

    p = mp.Process(target=generate_sentences, args=(tokenizer, decoder_path, 0, top_k, l))  # data, 
    p.start()
    processes.append(p)

    for p in processes:
        p.join()


    Dialog_list = []
    with open('generate_sentences.txt', 'w', encoding='utf-8') as f:
        for s in l:
            cases = dict()
            cases['input'] = s[0]
            cases['predict'] = s[1]
            cases['reference'] = s[2]
            Dialog_list.append(cases)
        json.dump(Dialog_list, f, ensure_ascii = False, indent = 4)


if __name__ == '__main__':
    fire.Fire(sample_generate)

