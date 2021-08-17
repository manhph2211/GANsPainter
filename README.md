GANsPainter :smile:
=====

In this project, I tried to implement CycleGANs to generate images in the style of Monet. My implemented model is based on what I learned in [this course](https://www.coursera.org/learn/apply-generative-adversarial-networks-gans) and [the official author](https://github.com/junyanz/CycleGAN)

# 1. Dateset

- Dataset can be found at [data](https://www.kaggle.com/c/gan-getting-started/overview). This is a kaggle competition and it hasn't finished yet at the time I push these code.

# 2. Dependencies

- You should create a virtual environment and then run `pip install -r requirements.txt` 

  - Pillow==8.1.0
  - matplotlib==3.3.4
  - numpy==1.20.0
  - opencv-python==4.5.1.48
  - scikit-learn==0.24.1
  - sklearn==0.0
  - torch==1.7.1
  - torchvision==0.8.2
  - tqdm==4.56.0

# 3. Usage

- If you want to use my pre-trained model, download [this](...) and put it int `./src`
- Use this command: 

```python train.py --batch_size [your option] --resume [your option] --epochs [your option] --lr [your option] # set resume True if using pretrained!!```

# 4. Results 

