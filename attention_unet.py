

import torch
import torch.nn as nn
torch.cuda.empty_cache()


class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x


class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x

class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi


class AttU_Net(nn.Module):

    def __init__(self,img_ch=3,output_ch=2):
        super(AttU_Net,self).__init__()

        filters = [16, 16*2, 16*4, 16*8, 16*16, 16*32] # = [16, 32, 64, 128, 256, 512]

        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=filters[0])
        self.Conv2 = conv_block(ch_in=filters[0], ch_out=filters[1])
        self.Conv3 = conv_block(ch_in=filters[1], ch_out=filters[2])
        self.Conv4 = conv_block(ch_in=filters[2], ch_out=filters[3])
        self.Conv5 = conv_block(ch_in=filters[3], ch_out=filters[4])
        self.Conv6 = conv_block(ch_in=filters[4], ch_out=filters[5])

        self.Up6 = up_conv(ch_in=filters[5], ch_out=filters[4])
        self.Att6 = Attention_block(F_g=filters[4], F_l=filters[4], F_int=filters[3])
        self.Up_conv6 = conv_block(ch_in=filters[5], ch_out=filters[4])

        self.Up5 = up_conv(ch_in=filters[4],ch_out=filters[3])
        self.Att5 = Attention_block(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.Up_conv5 = conv_block(ch_in=filters[4], ch_out=filters[3])

        self.Up4 = up_conv(ch_in=filters[3], ch_out=filters[2])
        self.Att4 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Up_conv4 = conv_block(ch_in=filters[3], ch_out=filters[2])

        self.Up3 = up_conv(ch_in=filters[2],ch_out=filters[1])
        self.Att3 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Up_conv3 = conv_block(ch_in=filters[2], ch_out=filters[1])

        self.Up2 = up_conv(ch_in=filters[1], ch_out=filters[0])
        self.Att2 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=filters[0])
        self.Up_conv2 = conv_block(ch_in=filters[1], ch_out=filters[0])


        self.Conv_1x1 = nn.Conv2d(filters[0], output_ch, kernel_size=1, stride=1, padding=0)


    def forward(self, x):

        # encoding path
        x1 = self.Conv1(x)
        x = self.Maxpool(x1)
        x2 = self.Conv2(x)
        x = self.Maxpool(x2)
        x3 = self.Conv3(x)
        x = self.Maxpool(x3)
        x4 = self.Conv4(x)
        x = self.Maxpool(x4)
        x5 = self.Conv5(x)
        x = self.Maxpool(x5)
        x = self.Conv6(x) # Root of the encoder

        x = self.Up6(x)
        x5 = self.Att6(g=x,x=x5)
        x = torch.cat((x5, x),dim=1)
        x = self.Up_conv6(x)

        x = self.Up5(x)
        x5 = self.Att5(g=x,x=x4)
        x = torch.cat((x4, x),dim=1)
        x = self.Up_conv5(x)

        x = self.Up4(x)
        x3 = self.Att4(g=x,x=x3)
        x = torch.cat((x3, x),dim=1)
        x = self.Up_conv4(x)

        x = self.Up3(x)
        x2 = self.Att3(g=x,x=x2)
        x = torch.cat((x2,x),dim=1)
        x = self.Up_conv3(x)

        x = self.Up2(x)
        x1 = self.Att2(g=x,x=x1)
        x = torch.cat((x1, x),dim=1)
        x = self.Up_conv2(x)

        x = self.Conv_1x1(x)

        del(x1, x2, x3, x4, x5)

        return x
