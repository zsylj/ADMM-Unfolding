import torch
import torch.nn as nn
from utils import A, At

class myCNN(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(myCNN, self).__init__()
        self.relu = nn.ReLU()
        self.l1 = nn.Conv2d(in_ch, 16, (3,3), padding=1)
        self.l2 = nn.Conv2d(16, 32, (3,3), padding=1)
        self.l3 = nn.Conv2d(32, 64, (3,3), padding=1)
        self.l4 = nn.Conv2d(64, 32, (3,3), padding=1)
        self.l5 = nn.Conv2d(32, 16, (3,3), padding=1)
        self.l6 = nn.Conv2d(16, out_ch, (3,3), padding=1)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        out = self.relu(out)
        out = self.l4(out)
        out = self.relu(out)
        out = self.l5(out)
        out = self.relu(out)
        out = self.l6(out)
        return out

class myUNet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(myUNet, self).__init__()
        self.relu = nn.ReLU()

        self.l1_1 = nn.Conv2d(in_ch, 32, (3,3), padding=1)
        self.l1_2 = nn.Conv2d(32, 32, (3,3), padding=1)
        self.l1_l2 = nn.MaxPool2d((2,2))

        self.l2_1 = nn.Conv2d(32, 64, (3,3), padding=1)
        self.l2_2 = nn.Conv2d(64, 64, (3,3), padding=1)
        self.l2_l3 = nn.MaxPool2d((2,2))

        self.l3_1 = nn.Conv2d(64, 128, (3,3), padding=1)
        self.l3_2 = nn.Conv2d(128, 128, (3,3), padding=1)
        

        self.l3_l2 = nn.ConvTranspose2d(128, 64, (4,4), stride=2, padding=1)
        self.l2_r1 = nn.Conv2d(128, 64, (3,3), padding=1)
        self.l2_r2 = nn.Conv2d(64, 64, (3,3), padding=1)

        self.l2_l1 = nn.ConvTranspose2d(64, 32, (4,4), stride=2, padding=1)
        self.l1_r1 = nn.Conv2d(64, 32, (3,3), padding=1)
        self.l1_r2 = nn.Conv2d(32, 32, (3,3), padding=1)

        self.l1_r3 = nn.Conv2d(32, out_ch, (3,3), padding=1)

        self.last = nn.Tanh()

    def forward(self, x):
        out_l1dw = self.l1_1(x)
        out_l1dw = self.relu(out_l1dw)
        out_l1dw = self.l1_2(out_l1dw)
        out_l1dw = self.relu(out_l1dw)
        
        out_l2dw = self.l1_l2(out_l1dw)
        
        out_l2dw = self.l2_1(out_l2dw)
        out_l2dw = self.relu(out_l2dw)
        out_l2dw = self.l2_2(out_l2dw)
        out_l2dw = self.relu(out_l2dw)

        out_l3dw = self.l2_l3(out_l2dw)

        out_l3dw = self.l3_1(out_l3dw)
        out_l3dw = self.relu(out_l3dw)
        out_l3dw = self.l3_2(out_l3dw)
        out_l3dw = self.relu(out_l3dw)

        out_l2up = self.l3_l2(out_l3dw)
        out_l2up = self.relu(out_l2up)

        out_l2up = torch.cat((out_l2up, out_l2dw), 1)

        out_l2up = self.l2_r1(out_l2up)
        out_l2up = self.relu(out_l2up)
        out_l2up = self.l2_r2(out_l2up)
        out_l2up = self.relu(out_l2up)

        out_l1up = self.l2_l1(out_l2up)
        out_l1up = self.relu(out_l1up)

        out_l1up = torch.cat((out_l1up, out_l1dw), 1)
        out_l1up = self.l1_r1(out_l1up)
        out_l1up = self.relu(out_l1up)
        out_l1up = self.l1_r2(out_l1up)
        out_l1up = self.relu(out_l1up)

        out = self.l1_r3(out_l1up)

        out = self.last(out)

        out = out + x

        return out
     
###############################################################################
class GAPNet(nn.Module):
    def __init__(self):
        super(GAPNet, self).__init__()
        self.myNet01 = myUNet(in_ch=8, out_ch=8)
        self.myNet02 = myUNet(in_ch=8, out_ch=8)
        self.myNet03 = myUNet(in_ch=8, out_ch=8)
        self.myNet04 = myUNet(in_ch=8, out_ch=8)
        self.myNet05 = myUNet(in_ch=8, out_ch=8)
        self.myNet06 = myUNet(in_ch=8, out_ch=8)
        self.myNet07 = myUNet(in_ch=8, out_ch=8)
        self.myNet08 = myUNet(in_ch=8, out_ch=8)
        self.myNet09 = myUNet(in_ch=8, out_ch=8)
        self.myNet10 = myUNet(in_ch=8, out_ch=8)
        self.myNet11 = myUNet(in_ch=8, out_ch=8)
        self.myNet12 = myUNet(in_ch=8, out_ch=8)
        self.myNet13 = myUNet(in_ch=8, out_ch=8)
        self.tau01 = nn.Parameter(torch.Tensor([1]))
        self.tau02 = nn.Parameter(torch.Tensor([1]))
        self.tau03 = nn.Parameter(torch.Tensor([1]))
        self.tau04 = nn.Parameter(torch.Tensor([1]))
        self.tau05 = nn.Parameter(torch.Tensor([1]))
        self.tau06 = nn.Parameter(torch.Tensor([1]))
        self.tau07 = nn.Parameter(torch.Tensor([1]))
        self.tau08 = nn.Parameter(torch.Tensor([1]))
        self.tau09 = nn.Parameter(torch.Tensor([1]))
        self.tau10 = nn.Parameter(torch.Tensor([1]))
        self.tau11 = nn.Parameter(torch.Tensor([1]))
        self.tau12 = nn.Parameter(torch.Tensor([1]))
        self.tau13 = nn.Parameter(torch.Tensor([1]))

    # def forward(self, y, mask, HtH, v, lmbd):
    def forward(self, y, mask, HHt):       


        v = torch.zeros_like(mask)
        lmbd = torch.zeros_like(mask)

        theta01 = self.tau01 * v + lmbd
        temp01 = At(y, mask) + theta01
        denom01 = HHt + self.tau01
        x = (temp01 - At(torch.div(A(temp01, mask) ,denom01), mask)) / self.tau01
        v = self.myNet01(x)
        lmbd = lmbd - (x - v)

        theta02 = self.tau02 * v + lmbd
        temp02 = At(y, mask) + theta02
        denom02 = HHt + self.tau02
        x = (temp02 - At(torch.div(A(temp02, mask) ,denom02), mask)) / self.tau02
        v = self.myNet02(x)
        lmbd = lmbd - (x - v)

        theta03 = self.tau03 * v + lmbd
        temp03 = At(y, mask) + theta03
        denom03 = HHt + self.tau03
        x = (temp03 - At(torch.div(A(temp03, mask) ,denom03), mask)) / self.tau03
        v = self.myNet03(x)
        lmbd = lmbd - (x - v)
        
        theta04 = self.tau04 * v + lmbd
        temp04 = At(y, mask) + theta04
        denom04 = HHt + self.tau04
        x = (temp04 - At(torch.div(A(temp04, mask) ,denom04), mask)) / self.tau04
        v = self.myNet04(x)
        lmbd = lmbd - (x - v)

        theta05 = self.tau05 * v + lmbd
        temp05 = At(y, mask) + theta05
        denom05 = HHt + self.tau05
        x = (temp05 - At(torch.div(A(temp05, mask) ,denom05), mask)) / self.tau05
        v = self.myNet05(x)
        lmbd = lmbd - (x - v)


        # theta01 = self.tau01 * v + lmbd
        # temp01 = At(y, mask) + theta01
        # denom01 = HHt + self.tau01
        # x = (temp01 - At(torch.div(A(temp01, mask) ,denom01), mask)) / self.tau01
        # v = self.myNet01((x - lmbd)/self.tau01)
        # lmbd = lmbd - (x - v)

        # theta02 = self.tau02 * v + lmbd
        # temp02 = At(y, mask) + theta02
        # denom02 = HHt + self.tau02
        # x = (temp02 - At(torch.div(A(temp02, mask) ,denom02), mask)) / self.tau02
        # v = self.myNet02((x - lmbd)/self.tau02)
        # lmbd = lmbd - (x - v)

        # theta03 = self.tau03 * v + lmbd
        # temp03 = At(y, mask) + theta03
        # denom03 = HHt + self.tau03
        # x = (temp03 - At(torch.div(A(temp03, mask) ,denom03), mask)) / self.tau03
        # v = self.myNet03((x - lmbd)/self.tau03)
        # lmbd = lmbd - (x - v)
        
        # theta04 = self.tau04 * v + lmbd
        # temp04 = At(y, mask) + theta04
        # denom04 = HHt + self.tau04
        # x = (temp04 - At(torch.div(A(temp04, mask) ,denom04), mask)) / self.tau04
        # v = self.myNet04((x - lmbd)/self.tau04)
        # lmbd = lmbd - (x - v)

        # theta05 = self.tau05 * v + lmbd
        # temp05 = At(y, mask) + theta05
        # denom05 = HHt + self.tau05
        # x = (temp05 - At(torch.div(A(temp05, mask) ,denom05), mask)) / self.tau05
        # v = self.myNet05((x - lmbd)/self.tau05)
        # lmbd = lmbd - (x - v)

        # yb = A(v+ lmbd, mask)
        # x = v + lmbd + At(torch.div(y-yb, HHt + self.tau01), mask)
        # x1 = x - lmbd
        # v = self.myNet01(x1)
        # lmbd = lmbd - (x - v)

        # yb = A(v+ lmbd, mask)
        # x = v + lmbd + At(torch.div(y-yb, HHt + self.tau02), mask)
        # x1 = x - lmbd
        # v = self.myNet02(x1)
        # lmbd = lmbd - (x - v)

        # yb = A(v+ lmbd, mask)
        # x = v + lmbd + At(torch.div(y-yb, HHt + self.tau03), mask)
        # x1 = x - lmbd
        # v = self.myNet03(x1)
        # lmbd = lmbd - (x - v)

        # yb = A(v+ lmbd, mask)
        # x = v + lmbd + At(torch.div(y-yb, HHt + self.tau04), mask)
        # x1 = x - lmbd
        # v = self.myNet04(x1)
        # lmbd = lmbd - (x - v)

        # yb = A(v+ lmbd, mask)
        # x = v + lmbd + At(torch.div(y-yb, HHt + self.tau05), mask)
        # x1 = x - lmbd
        # v = self.myNet05(x1)
        # lmbd = lmbd - (x - v)
        return x


################################################################################

class SingleStage(nn.Module):
    def __init__(self):
        super(SingleStage, self).__init__()
        # self.myCNN = myCNN(in_ch=8, out_ch=8)
        self.myUNet = myUNet(in_ch=8, out_ch=8)

    def forward(self, y, mask, HtH, v, tau):
        # INPUTS MUST FOLLOW SIZE BELOW!!!!
        # y dim (batch, n_frames, height_px, width_px)
        # mask dim (batch, n_frames, height_px, width_px)

        # ziyi original
        # yb = (v+ lmbd) * mask
        # x = v + lmbd + mask * torch.div(y-yb, HtH + tau)
        # x1 = x - lmbd
        # v = self.myUNet(x1)
        # lmbd = lmbd - (x - v)

        x = v + mask * torch.div(y - v * mask, HtH + tau)
        v = self.myUNet(x)

        return v

class GAPNet2(nn.Module):
    def __init__(self, n_stage):
        super(GAPNet2, self).__init__()
        self.n_stage = n_stage
        onestage = []
        for i in range(0, n_stage):
            onestage.append(SingleStage())
        
        self.allstage = nn.ModuleList(onestage)

    def forward(self, y, mask, HtH, v, tau):
        for i in range(0, self.n_stage):
            v = self.allstage[i](y, mask, HtH, v, tau)

        return v
    





