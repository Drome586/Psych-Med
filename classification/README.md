### How to Run

1. You can download the pretrained models such as BERT, RoBERTa, and CPT, and then place them under the "pretrained" folder. Or you can load them using the parameter "--model_name_or_path".

2. You can run with the command `sh run.sh` and modify the parameters inside. 

|  Parameters  | Explanation  |
|  ----  | ----  |
|  model_name_or_path   | Local path of pretrained model or model name in [HuggingFace](https://huggingface.co/models)  |
| classification_task_name  | 'disease' |
| model_type  | 'bert' or 'roberta' or 'albert' or 'cpt', etc. |

### Requirements (for CPT)

- pytorch==1.8.1
- transformers==4.4.1

