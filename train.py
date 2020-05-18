import os
import torch

from data import train_dataloader
from utils import Adder, Timer, WarmUpScheduler, check_lr, PolynomialScheduler
from torch.utils.tensorboard import SummaryWriter
from valid import _valid
import torch.nn.functional as F


def _train(model, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config.learning_rate,
                                 weight_decay=config.weight_decay)

    dataloader = train_dataloader(config.data_dir, config.batch_size, config.crop_size, config.scale_factor, config.num_worker)
    max_iter = len(dataloader)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, config.lr_steps, config.gamma)
    #scheduler = PolynomialScheduler(optimizer, config.learning_rate, 0, 0.3, config.num_epoch)
    #wu_scheduler = WarmUpSchduler(optimizer, config.learning_rate*0.01, config.learning_rate, max_iter)

    writer = SummaryWriter()
    epoch_adder = Adder()
    iter_adder = Adder()
    epoch_timer = Timer('m')
    iter_timer = Timer('m')

    for epoch_idx in range(1, config.num_epoch + 1):
        epoch_timer.tic()
        iter_timer.tic()
        for iter_idx, batch_data in enumerate(dataloader):
            # if epoch_idx == 1:
            #     wu_scheduler.step()
            input_img, input_depth, label_img = batch_data
            input_img = input_img.to(device)
            input_depth = input_depth.to(device)
            label_img = label_img.to(device)

            optimizer.zero_grad()

            pred_img = model(input_img, input_depth)

            loss = criterion(pred_img, label_img)
            loss.backward()
            optimizer.step()

            iter_adder(loss.item())
            epoch_adder(loss.item())

            if (iter_idx + 1) % config.print_freq == 0:
                lr = check_lr(optimizer)
                print("Time: %7.4f Epoch: %03d Iter: %4d/%4d LR: %.10f Loss: %7.4f" % (iter_timer.toc(), epoch_idx,
                                                                             iter_idx + 1, max_iter, lr,
                                                                             iter_adder.average()))
                writer.add_scalar('Loss', iter_adder.average(), iter_idx + (epoch_idx-1)* max_iter)
                iter_timer.tic()
                iter_adder.reset()
        if epoch_idx % config.save_freq == 0:
            save_name = os.path.join(config.model_save_dir, 'model_%d.pkl' % epoch_idx)
            torch.save({'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch_idx}, save_name)
        print("EPOCH: %02d\nElapsed time: %4.2f Epoch Loss: %7.4f" % (epoch_idx, epoch_timer.toc(), epoch_adder.average()))
        epoch_adder.reset()
        scheduler.step()
        if epoch_idx % config.valid_freq == 0:
            val = _valid(model, config, epoch_idx)
            print('%03d epoch \n Average PSNR %.2f dB' % (epoch_idx, val))
            writer.add_scalar('PSNR', val, epoch_idx)
    #save_name = os.path.join(config.model_save_dir, 'Model.pkl')
    #torch.save({'model': model.state_dict()}, save_name)
