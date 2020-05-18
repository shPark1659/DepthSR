import torch
import torch.nn as nn
from models.cbam import *
from models.subNets import *

class DepthSRnet(nn.Module):
    def __init__(self):
        super(DepthSRnet, self).__init__()
        #input pyramid branch
        self.ipb1 = nn.Sequential( #result 48x48x64 Fpy_1
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.ipb2 = nn.Sequential( #result 24x24x128 Fpy_2
            nn.Conv2d(1, 128, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.ipb3 = nn.Sequential( #result 12x12x256 Fpy_3
            nn.Conv2d(1, 256, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.ipb4 = nn.Sequential( #result 6x6x512 Fpy_4
            nn.Conv2d(1, 512, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        # encoder branch
        self.ecb1 = nn.Sequential( #result 96x96x64 Fecb_1 pooling시 Decb_1
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.ecb2_1 = nn.Sequential( #result 48x48x64 Fecb_2
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.ecb2_2 = nn.Sequential( #result 48x48x128 Fecb_3 pooling시 Decb_2
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.ecb3_1 = nn.Sequential( #result 24x24x128 Fecb_4
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.ecb3_2 = nn.Sequential( #result 24x24x256 Fecb_5 pooling시 Decb_3
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.ecb4_1 = nn.Sequential( #result 12x12x256 Fecb_6
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.ecb4_2 = nn.Sequential( #result 12x12x512 Fecb_7 pooling시 Decb_4
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.ecb5_1 = nn.Sequential( #result 6x6x512 Fecb_8
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.ecb5_2 = nn.Sequential( #result 6x6x1024 Fecb_9
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        #y guidance branch
        self.y1 = nn.Sequential( #result 96x96x64 Fy_1 pooling시 Dy_1
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.y2 = nn.Sequential( #result 48x48x128 Fy_2 pooling시 Dy_2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.y3 = nn.Sequential( #result 24x24x256 Fy_3 pooling시 Dy_3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.y4 = nn.Sequential( #result 12x12x512 Fy_4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        #decoder branch
        self.dcb1 = nn.Sequential( #result 6x6x1024 Fdec_1
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.dcb2_1 = nn.Sequential( #result 12x12x512 Fdec_2
            nn.ConvTranspose2d(1024, 512, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True)
        )
        self.dcb2_2 = nn.Sequential( #result 12x12x512 Fdec_4
            nn.Conv2d(1536, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.dcb3_1 = nn.Sequential( #result 24x24x256 Fdec_5
            nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True)
        )
        self.dcb3_2 = nn.Sequential( #result 24x24x256 Fdec_7
            nn.Conv2d(768, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.dcb4_1 = nn.Sequential( #result 48x48x128 Fdec_8
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True)
        )
        self.dcb4_2 = nn.Sequential( #result 48x48x128 Fdec_10
            nn.Conv2d(384, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.dcb5_1 = nn.Sequential( #result 96x96x64 Fdec_11
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True)
        )
        self.dcb5_2 = nn.Sequential(  # result 96x96x1 Fdec_13
            nn.Conv2d(192, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 3, padding=1)
        )

        self.max_pool = nn.MaxPool2d(2, stride=2, padding=0)

        self.ipb1.apply(self.weight_init)
        self.ipb2.apply(self.weight_init)
        self.ipb3.apply(self.weight_init)
        self.ipb4.apply(self.weight_init)

        self.ecb1.apply(self.weight_init)
        self.ecb2_1.apply(self.weight_init)
        self.ecb2_2.apply(self.weight_init)
        self.ecb3_1.apply(self.weight_init)
        self.ecb3_2.apply(self.weight_init)
        self.ecb4_1.apply(self.weight_init)
        self.ecb4_2.apply(self.weight_init)
        self.ecb5_1.apply(self.weight_init)
        self.ecb5_2.apply(self.weight_init)

        self.y1.apply(self.weight_init)
        self.y2.apply(self.weight_init)
        self.y3.apply(self.weight_init)
        self.y4.apply(self.weight_init)

        self.dcb1.apply(self.weight_init)
        self.dcb2_1.apply(self.weight_init)
        self.dcb2_2.apply(self.weight_init)
        self.dcb3_1.apply(self.weight_init)
        self.dcb3_2.apply(self.weight_init)
        self.dcb4_1.apply(self.weight_init)
        self.dcb4_2.apply(self.weight_init)
        self.dcb5_1.apply(self.weight_init)
        self.dcb5_2.apply(self.weight_init)

    def forward(self, y, d):
        #input pyramid branch
        Dpy_1 = self.max_pool(d)
        Fpy_1 = self.ipb1(Dpy_1)
        Dpy_2 = self.max_pool(Dpy_1)
        Fpy_2 = self.ipb2(Dpy_2)
        Dpy_3 = self.max_pool(Dpy_2)
        Fpy_3 = self.ipb3(Dpy_3)
        Dpy_4 = self.max_pool(Dpy_3)
        Fpy_4 = self.ipb4(Dpy_4)

        #encoder branch
        Fecb_1 = self.ecb1(d)
        Decb_1 = self.max_pool(Fecb_1)
        Fecb_2 = self.ecb2_1(Decb_1)
        Fecb_3 = self.ecb2_2(torch.cat([Fecb_2, Fpy_1], 1))

        Decb_2 = self.max_pool(Fecb_3)
        Fecb_4 = self.ecb3_1(Decb_2)
        Fecb_5 = self.ecb3_2(torch.cat([Fecb_4, Fpy_2], 1))

        Decb_3 = self.max_pool(Fecb_5)
        Fecb_6 = self.ecb4_1(Decb_3)
        Fecb_7 = self.ecb4_2(torch.cat([Fecb_6, Fpy_3], 1))

        Decb_4 = self.max_pool(Fecb_7)
        Fecb_8 = self.ecb5_1(Decb_4)
        Fecb_9 = self.ecb5_2(torch.cat([Fecb_8, Fpy_4], 1))

        #y guidance branch
        Fy_1 = self.y1(y)
        Dy_1 = self.max_pool(Fy_1)
        Fy_2 = self.y2(Dy_1)
        Dy_2 = self.max_pool(Fy_2)
        Fy_3 = self.y3(Dy_2)
        Dy_3 = self.max_pool(Fy_3)
        Fy_4 = self.y4(Dy_3)

        #decoder branch
        Fdec_1 = self.dcb1(Fecb_9)

        Fdec_2 = self.dcb2_1(Fdec_1)
        Fdec_4 = self.dcb2_2(torch.cat([Fdec_2, Fy_4, Fecb_7], 1))

        Fdec_5 = self.dcb3_1(Fdec_4)
        Fdec_7 = self.dcb3_2(torch.cat([Fdec_5, Fy_3, Fecb_5], 1))

        Fdec_8 = self.dcb4_1(Fdec_7)
        Fdec_10 = self.dcb4_2(torch.cat([Fdec_8, Fy_2, Fecb_3], 1))

        Fdec_11 = self.dcb5_1(Fdec_10)
        Fdec_13 = self.dcb5_2(torch.cat([Fdec_11, Fy_1, Fecb_1], 1))

        result = d + Fdec_13

        return result

    def weight_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0.0, 0.5 * math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif classname.find('Linear') != -1:
            n = m.weight.size(1)
            m.weight.data.normal_(0, 0.01)
            m.bias.data = torch.ones(m.bias.data.size())

def build_net():
    return DepthSRnet()