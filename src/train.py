import torch
import os
import argparse
from trainer import Trainer 


def main():
    parser = argparse.ArgumentParser(description='Painting ...')
    parser.add_argument('--batch_size', type=int, help='Batch size for training')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--num_workers', type=int, help='Number of dataloader workers')
    parser.add_argument('--lr', type=float, help='initial learning rate')
    parser.add_argument('--optimizer', type=str, help='The optimizer want to use')
    parser.set_defaults(debug=False)
    parser.set_defaults(benchmark=True)
    args = parser.parse_args()
    trainer = Trainer(args)
    trainer.train()

if __name__ == '__main__':
    main()