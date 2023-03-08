# Chinese-Transformer

Use "preprocess.py" to generate the input data.
Run "trans_train.py" for training, "trans_perplexity.py" to calculate the perplexity, and "trans_generate.py" to generate the response and calculate the other metrics.

The original data is "*.json", please run "preprocess" to generate the "*.pth", which is the input file of our model. 

Requirement:
Torch 1.8.0 
Python 3.7.0 (or above)
Transformers 4.4.1 (or above)