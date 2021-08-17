import argparse
from trainer import Trainer 


def main():
    parser = argparse.ArgumentParser(description='Painting ...')
    parser.add_argument('--batch_size', type=int, default =1, help='Batch size for training')
    parser.add_argument('--resume', type=bool, default = True, help='Resume from checkpoint')
    parser.add_argument('--num_workers', type=int, default =2, help='Number of dataloader workers')
    parser.add_argument('--img_size', type=int, default = 256, help ='image size')
    args = parser.parse_args()
    trainer = Trainer(args)
    trainer.test()

if __name__ == '__main__':
    main()