### How to Run

1. Download the pretrained models and put them into corresponding folder.

> CDial-GPT can be downloaded [here](https://huggingface.co/thu-coai/CDial-GPT2_LCCC-base) (provided by [thu-coai/CDial-GPT](https://github.com/thu-coai/CDial-GPT)).


2. `pip install -r requirements.txt`

3. For training, run

    `python train.py --pretrained --model_checkpoint pretrained/GDial-GPT --data_path data/data.json --scheduler linear`

4. Generate conversation

For generating responses on test data:
    `python infer.py --model_checkpoint YOUR_MODEL_PATH --datapath data/test.json --out_path result.txt`
For interacting with the model from the command line:
    `python interact.py --model_checkpoint YOUR_MODEL_PATH`

