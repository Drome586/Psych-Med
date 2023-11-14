### How to Run

1. The pretrained model "bertGPT_pretrained_model.pth" can be downloaded [here](https://drive.google.com/file/d/1alyU4wEClpjj2-kGl45xxUal0dHZGZhI/view?usp=sharing) (provided by [UCSD-AI4H/Medical-Dialogue-System](https://github.com/UCSD-AI4H/Medical-Dialogue-System)). It should be placed under the "pretrained" folder.
2. `pip install -r requirements.txt`
3. When training, run `python train.py`
4. When calculate the perplexity, run `python perplexity.py`
5. When testing, run `python generate.py` to get the generated dialogues file,
  and run `python validate.py` to calculate the metrics.

