from dataset import *
from tqdm.notebook import tqdm
import os
import torch
import numpy as np
from discriminator import *
from generator import * 
from utils import plot
import torch.nn as nn
from loss import *
import cv2


class Trainer:
    def __init__(self, args):
        self.init_device()
        self.args = args
        
    def init_device(self):
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

    def init_model(self):
        self.gen_AB = Generator(input_channels = 3, output_channels = 3).to(self.device)
        self.gen_BA = Generator(output_channels =3, input_channels = 3).to(self.device)
        self.gen_opt = torch.optim.Adam(list(self.gen_AB.parameters()) + list(self.gen_BA.parameters()), lr=self.args.lr, betas=(0.5, 0.999))

        self.disc_A = Discriminator(input_channels = 3).to(self.device)
        self.disc_A_opt = torch.optim.Adam(self.disc_A.parameters(), lr=self.args.lr, betas=(0.5, 0.999))
        self.disc_B = Discriminator(input_channels = 3).to(self.device)
        self.disc_B_opt = torch.optim.Adam(self.disc_B.parameters(), lr=self.args.lr, betas=(0.5, 0.999))

        def weights_init(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.normal_(m.weight, 0.0, 0.02)
            if isinstance(m, nn.BatchNorm2d):
                torch.nn.init.normal_(m.weight, 0.0, 0.02)
                torch.nn.init.constant_(m.bias, 0)
        
        try:
            self.gen_AB.load_state_dict(torch.load('../weights/gen_AB.pth',map_location=self.device))
            self.gen_BA.load_state_dict(torch.load('../weights/gen_BA.pth',map_location=self.device))
            self.disc_A.load_state_dict(torch.load('../weights/disc_A.pth',map_location=self.device))
            self.disc_B.load_state_dict(torch.load('../weights/disc_B.pth',map_location=self.device))
        except:
            self.gen_AB = self.gen_AB.apply(weights_init)
            self.gen_BA = self.gen_BA.apply(weights_init)
            self.disc_A = self.disc_A.apply(weights_init)
            self.disc_B = self.disc_B.apply(weights_init)

    def test(self):
        self.init_model()
        transform = transforms.Compose([
            transforms.Resize(self.args.img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        if not os.path.isdir('../data/new_data'):
            os.mkdir('../data/new_data')

        for file in tqdm(glob.glob('../data/photo_jpg/*.jpg')):
            with torch.no_grad():
                real_B = transform(Image.open(file))
                fake_A = self.gen_BA(torch.unsqueeze(real_B, 0).to(self.device))
                fake_A = 255*(fake_A[0].permute(1,2,0).cpu().numpy() /2 + 0.5)
                cv2.imwrite(os.path.join('../data/new_data', file.split('/')[-1]), fake_A)

        assert len(os.listdir('../data/new_data')) >= 7000

    def train(self,save_model=True):

        mean_generator_loss = 0
        mean_discriminator_loss = 0

        self.init_model()

        transform = transforms.Compose([
            transforms.Resize(self.args.img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        dataset = ImageDataset('../data/data.json',transform)

        dataloader = DataLoader(dataset, batch_size=self.args.batch_size, num_workers=self.args.num_workers, shuffle=True)
        cur_step = 0
        adv_criterion = torch.nn.BCEWithLogitsLoss()
        recon_criterion = torch.nn.L1Loss()

        for epoch in range(self.args.n_epochs):
            for real_A, real_B, _ in tqdm(dataloader):
                # image_width = image.shape[3]
                cur_batch_size = len(real_A)
                real_A = real_A.to(self.device)
                real_B = real_B.to(self.device)

                ### Update discriminator A ###
                self.disc_A_opt.zero_grad() # Zero out the gradient before backpropagation
                with torch.no_grad():
                    fake_A = self.gen_BA(real_B)
                disc_A_loss = get_disc_loss(real_A, fake_A, self.disc_A, adv_criterion)
                disc_A_loss.backward(retain_graph=True) # Update gradients
                self.disc_A_opt.step() # Update optimizer

                ### Update discriminator B ###
                self.disc_B_opt.zero_grad() # Zero out the gradient before backpropagation
                with torch.no_grad():
                    fake_B = self.gen_AB(real_A)
                disc_B_loss = get_disc_loss(real_B, fake_B, self.disc_B, adv_criterion)
                disc_B_loss.backward(retain_graph=True) # Update gradients
                self.disc_B_opt.step() # Update optimizer

                ### Update generator ###
                self.gen_opt.zero_grad()
                gen_loss, fake_A, fake_B = get_gen_loss(
                    real_A, real_B, self.gen_AB, self.gen_BA, self.disc_A, self.disc_B, adv_criterion, recon_criterion, recon_criterion, lambda_identity=self.args.lambda_identity, lambda_cycle=self.args.lambda_cycle
                )
                gen_loss.backward() # Update gradients
                self.gen_opt.step() # Update optimizer

                # Keep track of the average discriminator loss
                mean_discriminator_loss += disc_A_loss.item() / self.args.display_step
                # Keep track of the average generator loss
                mean_generator_loss += gen_loss.item() / self.args.display_step

                ### Visualization code ###
                if cur_step % self.args.display_step == 0:
                    print(f"Epoch {epoch}: Step {cur_step}: Generator (U-Net) loss: {mean_generator_loss}, Discriminator loss: {mean_discriminator_loss}")
                    # plot(real_A[0].permute(1,2,0) / 2 + 0.5, fake_B[0].permute(1,2,0) / 2 + 0.5)
                    plot(real_B[0].permute(1,2,0)/2 + 0.5, fake_A[0].permute(1,2,0)/2 + 0.5)
                    mean_generator_loss = 0
                    mean_discriminator_loss = 0
                    # You can change save_model to True if you'd like to save the model
                    if save_model:
                        torch.save(self.gen_AB.state_dict(), '../weights/gen_AB.pth')
                        torch.save(self.gen_BA.state_dict(), '../weights/gen_BA.pth')
                        torch.save(self.disc_A.state_dict(), '../weights/disc_A.pth')
                        torch.save(self.disc_B.state_dict(), '../weights/disc_B.pth')
                cur_step += 1
