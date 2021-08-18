import argparse
from trainer import Trainer 


def main(mode = 'train'):
    parser = argparse.ArgumentParser(description='Painting ...')
    parser.add_argument('--batch_size', type=int, help='Batch size for training')
    parser.add_argument('--resume', type=bool, default = False, help='Resume from checkpoint')
    parser.add_argument('--n_epochs', type=int, help='Number of training epochs')
    parser.add_argument('--num_workers', type=int, default =2, help='Number of dataloader workers')
    parser.add_argument('--lr', type=float, help='initial learning rate')
    parser.add_argument('--img_size', type=int, default = 256, help ='image size')
    parser.add_argument('--display_step', type = int, default = 10, help='after display_step steps, show the imges')
    parser.add_argument('--lambda_identity', type = float, default = 0.1, help='the weight of the identity loss')
    parser.add_argument('--lambda_cycle', type = float, default = 10.0, help='the weight of the cycle-consistency loss')
    args = parser.parse_args()
    trainer = Trainer(args)
    if mode == 'train':
      trainer.train()
    elif mode == 'test':
      trainer.test()

if __name__ == '__main__':
    main()