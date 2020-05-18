import os
import torch
from torchvision.transforms import functional as F
import numpy as np
from utils import Adder, calculate_psnr
from data import test_dataloader
from utils import EvalTimer
from skimage.metrics import peak_signal_noise_ratio
import time

def _eval(model, config):
    state_dict = torch.load(config.test_model)
    model.load_state_dict(state_dict['model'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = test_dataloader(config.data_dir, batch_size=1, scale_factor=config.scale_factor, num_workers=0)
    adder = Adder()
#    timer = EvalTimer()
    model.eval()
    with torch.no_grad():
        psnr_adder = Adder()
        for iter_idx, data in enumerate(dataloader):
            input_img, input_depth, label_img, img_name = data

            if not os.path.exists(os.path.join(config.result_dir, 'EVAL')):
                os.mkdir(os.path.join(config.result_dir, 'EVAL'))

            save_name = os.path.join(config.result_dir, 'EVAL', '%d.png'%iter_idx)
            input_img = input_img.to(device)
            input_depth = input_depth.to(device)

            tm = time.time()
            pred = model(input_img, input_depth)
            elaps = time.time() - tm
            adder(elaps)

            p_numpy = pred.squeeze(0).cpu().numpy()
            in_numpy = label_img.squeeze(0).cpu().numpy()
            #p_numpy = np.clip(p_numpy, 0, 1)
            #pred = F.to_pil_image(pred[2].squeeze(0).cpu(), 'RGB')
            #label = F.to_pil_image(label_img, 'RGB')

            #pred.save(save_name)

            psnr = peak_signal_noise_ratio(p_numpy, in_numpy, data_range=1)
            psnr_adder(psnr)
            print('%s iter PSNR: %.2f time: %f' % (img_name, psnr, elaps))

        print('The average PSNR is %.2f dB' % (psnr_adder.average()))
        #timer.print_time()
        print("Average time: %f"%adder.average())
