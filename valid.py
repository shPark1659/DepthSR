import torch
from torchvision.transforms import functional as F
from data import valid_dataloader
from utils import Adder, calculate_psnr
import os
from skimage.metrics import peak_signal_noise_ratio


def _valid(model, config, ep):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = valid_dataloader(config.data_dir, batch_size=1, scale_factor=config.scale_factor, num_workers=0)
    model.eval()
    psnr_adder = Adder()

    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            input_img, input_depth, label_img, img_name = data
            input_img = input_img.to(device)
            input_depth = input_depth.to(device)

            #if not os.path.exists(os.path.join(config.result_dir, '%d' % (ep))):
            #    os.mkdir(os.path.join(config.result_dir, '%d' % (ep)))
            #save_name = os.path.join(config.result_dir, '%d' %ep, '%d' % (idx) + '.png')
            pred = model(input_img, input_depth)

            p_numpy = pred.squeeze(0).cpu().numpy()
            in_numpy = label_img.squeeze(0).cpu().numpy()

            psnr = peak_signal_noise_ratio(p_numpy, in_numpy, data_range=1)
            #label = F.to_pil_image(label_img.squeeze(0).cpu(), 'RGB')
            #label.save(save_name)

            print('%s PSNR: %.2f time: %f' % (img_name, psnr))

            psnr_adder(psnr)

    model.train()
    return psnr_adder.average()

