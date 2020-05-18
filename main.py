import os
import torch
import argparse
from torch.backends import cudnn
from models.DepthSRnet import build_net
from train import _train
from eval import _eval


def main(config):
    # CUDNN
    cudnn.benchmark = True

    if not os.path.exists('results/'):
        os.makedirs(config.model_save_dir)
    if not os.path.exists('results/' + config.model_name + '/'):
        os.makedirs('results/' + config.model_name + '/')
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)

    model = build_net()
    print(model)
    if torch.cuda.is_available():
        model.cuda()
    if config.mode == 'train':
        _train(model, config)

    elif config.mode == 'test':
        _eval(model, config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Directories
    parser.add_argument('--model_name', type=str, default='dsr')
    parser.add_argument('--data_dir', type=str, default='C:\\DepthSRdata/')

    # Train
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--crop_size', type=int, default=64)
    parser.add_argument('--scale_factor', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--num_epoch', type=int, default=500)
    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--num_worker', type=int, default=4)
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--valid_freq', type=int, default=5)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--lr_steps', type=list, default=[900])

    # Test
    parser.add_argument('--test_model', type=str, default='results/dsr/weights/model_70.pkl')
    parser.add_argument('--mode', type=str, default='train')

    config = parser.parse_args()
    config.model_save_dir = os.path.join('results/', config.model_name, 'weights/')
    config.result_dir = os.path.join('results/', config.model_name, 'eval/')
    print(config)
    main(config)
