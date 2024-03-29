import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from transformers import BertModel, BertConfig,BartForConditionalGeneration,BertTokenizer,BartConfig,BartModel
import json
import fire
from collections import defaultdict

from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.translate.nist_score import sentence_nist
from nltk.util import ngrams

from tqdm import tqdm

def bleu(predict, target, n):
    return sentence_bleu([target], predict, weights=tuple(1 / n for i in range(n)))

def nist(predict, target, n):
    if len(predict) < n or len(target) < n:
        return 0
    return sentence_nist([target], predict, n)

def cal_entropy(generated):
    etp_score = [0.0, 0.0, 0.0, 0.0]
    div_score = [0.0, 0.0, 0.0, 0.0]
    counter = [defaultdict(int), defaultdict(int),
               defaultdict(int), defaultdict(int)]
    for gg in generated:
        g = gg.rstrip().split()
        for n in range(4):
            for idx in range(len(g)-n):
                ngram = ' '.join(g[idx:idx+n+1])
                counter[n][ngram] += 1
    for n in range(4):
        total = sum(counter[n].values()) + 1e-10
        for v in counter[n].values():
            etp_score[n] += - (v+0.0) / total * (np.log(v+0.0) - np.log(total))
        div_score[n] = (len(counter[n].values())+0.0) / total
    return etp_score, div_score

def cal_length(sentences):
    sen_length = [len(s) for s in sentences]
    return np.mean(sen_length), np.var(sen_length)

def calculate_metrics(predict, reference):
    reference_len = len(reference)
    predict_len = len(predict)

    #-------------------bleu----------
    bleu_2 = bleu(predict, reference, 2)
    bleu_4 = bleu(predict, reference, 4)
    #-------------------nist----------
    nist_2 = nist(predict, reference, 2)
    nist_4 = nist(predict, reference, 4)
    #-------------------meteor----------
    predict = " ".join(predict)
    reference = " ".join(reference)
    meteor_scores = meteor_score([reference], predict)
    return bleu_2, bleu_4, nist_2, nist_4, meteor_scores

def top_k_logits(logits, k):
    """Mask logits so that only top-k logits remain
    """
    values, _ = torch.topk(logits, k)
    min_values = values[:, -1].unsqueeze(1).repeat(1, logits.shape[-1])
    return torch.where(logits < min_values, torch.ones_like(logits, dtype=logits.dtype) * -1e10, logits)



class BartMo(nn.Module):
    def __init__(self):
        super().__init__()
        encoder_config = BartConfig(
            vocab_size=51271,
            encoder_layers=6,
            encoder_attention_heads=12,
            d_model=768
        )
        self.encoder = BartModel(encoder_config)

        decoder_config = BartConfig(
            vocab_size=51271,
            encoder_layers=6,
            encoder_attention_heads=12,
            d_model=768
        )
        decoder_config.is_decoder = True
        self.decoder = BartModel(decoder_config)

        self.linear = nn.Linear(768, 51271, bias=False)

    def forward(self, input_ids, mask_encoder_input, output_ids, mask_decoder_input):
        encoder_hidden_states, _ = self.encoder(input_ids, mask_encoder_input)
        # out: [batch_size, max_length, hidden_size]
        out, _ = self.decoder(output_ids, mask_decoder_input, encoder_hidden_states=encoder_hidden_states)
        
        out = self.linear(out)
        return out


def sample_generate(
    top_k = 50,
    temperature = 1.0,
    decoder_path='./decoder_model/4model.pth',
    gpu_id=0
    ):
    # make sure your model is on GPU
    device = torch.device(f"cuda:{gpu_id}")

    #------------------------LOAD MODEL-----------------
    print('load the model....')
    tokenizer = BertTokenizer.from_pretrained("fnlp/bart-base-chinese")

    model = BartForConditionalGeneration.from_pretrained("fnlp/bart-base-chinese")
    model.load_state_dict(torch.load(decoder_path))

    # Linear = nn.Linear(768,51271,False)
    # for k,v in torch.load(decoder_path).items():
    #     if k == 'lm_head.weight':
    #         Linear.weight.data = v.data
    # print(Linear)
    # device = torch.device(f"cuda:6")

    model.to(device)
    model.eval()

    print('load success')
    #------------------------END LOAD MODEL--------------


    #------------------------LOAD VALIDATE DATA------------------
    
    val_data = torch.load("test_data.pth")
    
    val_dataset = TensorDataset(*val_data)
    val_dataloader = DataLoader(dataset=val_dataset, shuffle=False, batch_size=1)
    #------------------------END LOAD VALIDATE DATA--------------

    #------------------------START GENERETE-------------------
    update_count = 0

    bleu_2scores = 0
    bleu_4scores = 0
    nist_2scores = 0
    nist_4scores = 0
    
    meteor_scores = 0
    sentences = []
    print('start generating....')
    f = open("dialogue.txt", "w")
    Encoder = model.get_encoder()
    Decoder = model.get_decoder()
    # Linea = nn.Linear(768,51271,bias=False)
    # Linea.to(device)
    result = []
    for batch in tqdm(val_dataloader):
        with torch.no_grad():
            batch = [item.to(device) for item in batch]
            encoder_input,decoder_input,attention_mask,_ = batch
            templist = []
            out_puts = model.generate(input_ids = encoder_input,attention_mask = attention_mask,max_length = 100)
            out_puts_str = tokenizer.batch_decode(out_puts,skip_special_tokens=True)
            out_puts_str = [s.replace(" ","") for s in out_puts_str]
            inputs = tokenizer.batch_decode(encoder_input,skip_special_tokens=True)
            inputs = [s.replace(" ", "") for s in inputs]
            reference = tokenizer.batch_decode(decoder_input,skip_special_tokens=True)
            reference = [s.replace(" ", "") for s in reference]
            templist.append(inputs[0])
            templist.append(reference[0])
            templist.append(out_puts_str[0])
            result.append(templist)
    print("result insert success")
    Dialog_list = []
    with open('generate_sentences.txt', 'w', encoding='utf-8') as f:
        for s in result:
            cases = dict()
            cases['input'] = s[0]
            cases['reference'] = s[1]
            cases['predict'] = s[2]
            Dialog_list.append(cases)
        json.dump(Dialog_list, f, ensure_ascii=False, indent=4)
    print("Dialog_list insert success")
            # print(inputs[0])
            # print(reference)
            # print(out_puts_str[0])
            # print("------------")



#     for batch in tqdm(val_dataloader):
#         with torch.no_grad():
#             batch = [item.to(device) for item in batch]
#
#             encoder_input, decoder_input, mask_encoder_input, mask_decoder_input = batch
#             past= Encoder(input_ids = encoder_input, attention_mask = mask_encoder_input)[0]
#
#
#             prev_pred = decoder_input[:, :1]
#             sentence = prev_pred
#
#             # decoding loop
#             for i in range(100):
#
#                 logits= Decoder(input_ids = decoder_input,attention_mask = mask_decoder_input,
#                                  encoder_attention_mask = mask_encoder_input,encoder_hidden_states = past)[0]
#                # print (logits.size())
#                 logits = Linear(logits)
#                 logits = logits[:, -1]
#                 logits = logits.squeeze(1) / temperature
#
#                 logits = top_k_logits(logits, k=top_k)
#                 probs = F.softmax(logits, dim=-1)
#                 prev_pred = torch.multinomial(probs, num_samples=1)
#                 sentence= torch.cat([sentence, prev_pred], dim=-1)
#                 if prev_pred[0][0] == 102:
#                     break
#
#             predict = tokenizer.convert_ids_to_tokens(sentence[0].tolist())
#
#             encoder_input = encoder_input.squeeze(dim=0)
#             encoder_input_num = (encoder_input != 0).sum()
#             inputs = tokenizer.convert_ids_to_tokens(encoder_input[:encoder_input_num].tolist())
#
#             decoder_input = decoder_input.squeeze(dim=0)
#             decoder_input_num = (decoder_input != 0).sum()
#
#             reference = tokenizer.convert_ids_to_tokens(decoder_input[:decoder_input_num].tolist())
# #            print('-'*20 + f"example {update_count}" + '-'*20)
# #            print(f"input: {''.join(inputs)}")
# #            print(f"output: {''.join(reference)}")a
# #            print(f"predict: {''.join(predict)}")
#             f.write('-'*20 + f"example {update_count}" + '-'*20 + '\n')
#             f.write(f"input: {''.join(inputs)}" + "\n")
#             f.write(f"output: {''.join(reference)}" + "\n")
#             f.write(f"predict: {''.join(predict)}" + "\n\n")
#
#             temp_bleu_2, \
#             temp_bleu_4, \
#             temp_nist_2, \
#             temp_nist_4, \
#             temp_meteor_scores = calculate_metrics(predict[1:-1], reference[1:-1])
#
#             bleu_2scores += temp_bleu_2
#             bleu_4scores += temp_bleu_4
#             nist_2scores += temp_nist_2
#             nist_4scores += temp_nist_4
#
#             meteor_scores += temp_meteor_scores
#             sentences.append(" ".join(predict[1:-1]))
#             update_count += 1
#
#     entro, dist = cal_entropy(sentences)
#     mean_len, var_len = cal_length(sentences)
#
#     f = open("test_res.txt", "w")
#     print(f'avg: {mean_len}, var: {var_len}')
#     print(f'entro: {entro}')
#     print(f'dist: {dist}')
#     print(f'test bleu_2scores: {bleu_2scores / update_count}')
#     print(f'test bleu_4scores: {bleu_4scores / update_count}')
#     print(f'test nist_2scores: {nist_2scores / update_count}')
#     print(f'test nist_4scores: {nist_4scores / update_count}')
#     print(f'test meteor_scores: {meteor_scores / update_count}')
#     f.write(f'avg: {mean_len}, var: {var_len}\n')
#     f.write(f'entro: {entro}\n')
#     f.write(f'dist: {dist}\n')
#     f.write(f'test bleu_2scores: {bleu_2scores / update_count}\n')
#     f.write(f'test bleu_4scores: {bleu_4scores / update_count}\n')
#     f.write(f'test nist_2scores: {nist_2scores / update_count}\n')
#     f.write(f'test nist_4scores: {nist_4scores / update_count}\n')
#     f.write(f'test meteor_scores: {meteor_scores / update_count}\n')


if __name__ == '__main__':
    fire.Fire(sample_generate)

