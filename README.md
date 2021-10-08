GANsPainter :smile:
=====

# 0. Introduction

In this project, I tried to implement CycleGANs to generate images in the style of Monet. My implemented model is based on what I learned in [this course](https://www.coursera.org/learn/apply-generative-adversarial-networks-gans) and [the official author](https://github.com/junyanz/CycleGAN)

There are several notes that you need to pay attention on when you're working with CycleGAN model

## 0.1 Generator 

- The code for a CycleGAN generator is much like Pix2Pix's U-Net with the addition of the residual block between the encoding (contracting) and decoding (expanding) blocks.
![image](https://user-images.githubusercontent.com/61444616/136531053-f414cf50-d89f-4b7e-89b1-b578463f241c.png)

- Residual block: In CycleGAN, after the expanding blocks, there are convolutional layers where the output is ultimately added to the original input so that the network can change as little as possible on the image. You can think of this transformation as a kind of skip connection, where instead of being concatenated as new channels before the convolution which combines them, it's added directly to the output of the convolution. In the visualization below, you can imagine the stripes being generated by the convolutions and then added to the original image of the horse to transform it into a zebra. These skip connections also allow the network to be deeper, because they help with vanishing gradients issues that come when a neural network gets too deep and the gradients multiply in backpropagation to become very small; instead, these skip connections enable more gradient flow. A deeper network is often able to learn more complex features
![image](https://user-images.githubusercontent.com/61444616/136531341-ac579642-6494-4def-b30c-8c20eea9c136.png)

## 0.2 PatchGAN Discriminator

- It is very similar to what you saw in Pix2Pix. Structured like the contracting path of the U-Net, the discriminator will
output a matrix of values classifying corresponding portions of the image as real or fake. 

## 0.3 Hyperpara
```
- adv_criterion: an adversarial loss function to keep track of how well the GAN is fooling the discriminator and how well the discriminator is catching the GAN
- recon_criterion: a loss function that rewards similar images to the ground truth, which "reconstruct" the image
- n_epochs: the number of times you iterate through the entire dataset when training
- dim_A: the number of channels of the images in pile A
- dim_B: the number of channels of the images in pile B (note that in the visualization this is currently treated as equivalent to dim_A)
- display_step: how often to display/visualize the images
- batch_size: the number of images per forward/backward pass
- lr: the learning rate
- target_shape: the size of the input and output images (in pixels)
- load_shape: the size for the dataset to load the images at before randomly cropping them to target_shape as a simple data augmentation
- device: the device type
```

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
