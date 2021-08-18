GANsPainter :smile:
=====

In this project, I tried to implement CycleGANs to generate images in the style of Monet. My implemented model is based on what I learned in [this course](https://www.coursera.org/learn/apply-generative-adversarial-networks-gans) and [the official author](https://github.com/junyanz/CycleGAN)

# 1. Dateset

- Dataset can be found at [data](https://www.kaggle.com/c/gan-getting-started/overview). This is a kaggle competition and it hasn't finished yet at the time I push these code.

- Save `monet_jpg` and `photo_jpg` in `./data`


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

- If you want to use my pre-trained model, download [this](https://drive.google.com/drive/folders/1H_Kpp1tpNS8C2XsKln6bKJiUij1yEWxR?usp=sharing) and put it into `./weights`

- Use these command: 

```

python utils.py

python train.py --batch_size [your option] --resume [your option] --epochs [your option] --lr [your option] # set resume True if using pretrained!!

```

# 4. Results 

- In the very first steps of training ... I got

![](https://github.com/manhph2211/GANsPainter/blob/main/images/first_steps.png)

- This one after some epochs:

![](https://github.com/manhph2211/GANsPainter/blob/main/images/2_epoch.png)

- The final image(right):

![](https://github.com/manhph2211/GANsPainter/blob/main/images/m_epoch.png)

Well, I'm not gonna be the next Monet :)
