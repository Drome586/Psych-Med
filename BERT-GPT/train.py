import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
import numpy as np

import fire
import time
import os
from tqdm import tqdm

# uses allennlp modules
from allennlp.nn import util

# imports chinese gpt
from chinese_gpt import TransformerEncoder, TransformerDecoderLM

# uses bert chinese wordpiece tokenization
# from pytorch_pretrained_bert import OpenAIAdam

from inputter import build_train_dataloaders, build_valid_dataloaders


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


def train_model(epochs=1, num_gradients_accumulation=4, batch_size=4, lr=1e-5, load_dir='decoder_model',
                decoder_model='pretrained/bertGPT_pretrained_model.pth'):

    device = torch.device("cuda:0")

    #------------------------LOAD MODEL-----------------
    print('load the model....')
    model = BertGPT()
    pretrained_model = torch.load(decoder_model, map_location=torch.device('cpu'))
    model.load_state_dict(pretrained_model)
    model = model.to(device)
    print('load success')
    #------------------------END LOAD MODEL--------------


    #------------------------LOAD TRAIN DATA------------------
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    train_dataset, train_dataloader = build_train_dataloaders(tokenizer, batch_size=batch_size)
    _, val_dataloader = build_valid_dataloaders(tokenizer, batch_size=batch_size)
    #------------------------END LOAD TRAIN DATA--------------
    

    #------------------------SET OPTIMIZER-------------------
    num_train_optimization_steps = int(
        len(train_dataset) / batch_size / num_gradients_accumulation) * epochs

    # param_optimizer = list(model.named_parameters())
    # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': 0.01},
    #     {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': 0.0}
    # ]

    # print(len(optimizer_grouped_parameters[0]['params']))

    # optimizer = OpenAIAdam(optimizer_grouped_parameters,
    #                        lr=lr,
    #                        warmup=0.01,
    #                        max_grad_norm=1.0,
    #                        weight_decay=0.01,
    #                        t_total=num_train_optimization_steps)

    optimizer = AdamW(model.parameters(), lr=lr, correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=num_train_optimization_steps)

    #------------------------END SET OPTIMIZER--------------


    #------------------------START TRAINING-------------------
    update_count = 0

    start = time.time()
    print('start training....')
    for epoch in range(epochs):
        #------------------------training------------------------
        model.train()
        losses = 0
        times = 0

        pbar_train = tqdm(train_dataloader, desc=f'epoch {epoch}')
        for batch in pbar_train:
            batch = [item.to(device) for item in batch]

            encoder_input, decoder_input, mask_encoder_input, mask_decoder_input = batch

            logits = model(encoder_input, mask_encoder_input, decoder_input, mask_decoder_input)
  
            out = logits[:, :-1].contiguous()
            target = decoder_input[:, 1:].contiguous()
            target_mask = mask_decoder_input[:, 1:].contiguous()
            loss = util.sequence_cross_entropy_with_logits(out, target, target_mask, average="token")
            loss.backward()

            losses += loss.item()
            times += 1
            
            update_count += 1

            if update_count % num_gradients_accumulation == num_gradients_accumulation - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            pbar_train.set_postfix(loss='{:.4f}'.format(loss.item()), lr=optimizer.param_groups[0]['lr'])

        end = time.time()
        print('-'*20 + f'epoch {epoch}' + '-'*20)
        print(f'time: {(end - start)}')
        print(f'loss: {losses / times}')
        start = end

        #------------------------validate------------------------
        model.eval()

        # perplexity = 0
        batch_count = 0
        print('start calculate the perplexity....')

        with torch.no_grad():
            losses = 0
            for batch in tqdm(val_dataloader):
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
        perplexity = np.exp(losses / batch_count)
        print(f'validate perplexity: {perplexity}')  #  / batch_count

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(os.path.abspath('.'), load_dir, str(epoch) + "decoder.pth"))

    #------------------------END TRAINING-------------------


if __name__ == '__main__':
    fire.Fire(train_model)

