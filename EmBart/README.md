# Chinese-Bart

Use "preprocess.py" to generate the input data.
Run "trans_train.py" for training, "trans_perplexity.py" to calculate the perplexity, and "trans_generate.py" to generate the response and calculate the other metrics.

The original data is "*.json", please run "preprocess" to generate the "*.pth", which is the input file of our model. 

We only provide a partial sample of the data, and the pre-trained model can be loaded from https://huggingface.co/fnlp/bart-base-chinese

Requirement:
Torch 1.8.1
Python 3.6.0 (or above)
Transformers 4.4.1 (or above)
nltk