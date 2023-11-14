import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, TensorDataset, DataLoader

import fire

# uses allennlp modules
from allennlp.nn import util

# imports chinese gpt
from chinese_gpt import TransformerEncoder, TransformerDecoderLM

from inputter import build_test_dataloaders


class BertGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = TransformerEncoder()
        self.decoder = TransformerDecoderLM()

    def forward(self, encoder_input, mask_encoder_input, decoder_input, mask_decoder_input):
        _, past = self.encoder(encoder_input, mask_encoder_input)

        mask = torch.cat([mask_encoder_input, mask_decoder_input], dim=1)
        logits, _ = self.decoder(decoder_input, mask, past=past, past_length=0)

        return logits


def calculate_perplexity(batch_size=4, decoder_path='pretrained/bertGPT_pretrained_model.pth'):

    device = torch.device("cuda:1")

    #------------------------LOAD MODEL-----------------
    print('load the model....')
    model = BertGPT()
    pretrained_model = torch.load(decoder_path, map_location=torch.device('cpu'))
    model.load_state_dict(pretrained_model)
    print(f'load from {decoder_path}')
    model = model.to(device)
    model.eval()
    print('load success')
    #------------------------END LOAD MODEL--------------
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    _, test_dataloader = build_test_dataloaders(tokenizer, batch_size=batch_size)
    
    #------------------------END LOAD VAL DATA--------------

    # perplexity = 0
    batch_count = 0
    print('start calculate the test perplexity....')

    with torch.no_grad():
        losses = 0
        for batch in tqdm(test_dataloader):
            batch = [item.to(device) for item in batch]
            encoder_input, decoder_input, mask_encoder_input, mask_decoder_input = batch

            logits = model(encoder_input, mask_encoder_input, decoder_input, mask_decoder_input)
  
            out = logits[:, :-1].contiguous()
            target = decoder_input[:, 1:].contiguous()
            target_mask = mask_decoder_input[:, 1:].contiguous()

            loss = util.sequence_cross_entropy_with_logits(out, target, target_mask, average="token")
            losses += loss.item()
            # perplexity += np.exp(loss.item())
            batch_count += 1
    print(f'losses / batch_count: {losses / batch_count}')
    perplexity = np.exp(losses / batch_count)
    print(f'test perplexity: {perplexity}')  #  / batch_count

    #------------------------END VAL-------------------


if __name__ == '__main__':
    fire.Fire(calculate_perplexity)

