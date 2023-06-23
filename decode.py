from torch import nn
import torch
from nodes import *
import numpy as np
#from self_attention_cv import AxialAttentionBlock
import torch.nn.functional as F
from copy import deepcopy
from softpool import soft_pool2d


# Graph Norm Constraints
# 1. The graph has to be globally connected.
def topo_sort(G):  # Topological Sorting
    #g=G.copy()
    g=deepcopy(G)
    n = len(g)
    q = []
    for j in range(n):
        flag = True
        for i in range(n):
            if g[i][j] == 1:
                flag = False
                break
        if flag:
            q.insert(0, j)

    res = []  # 记录结果
    while len(q) > 0:
        # p出队，把从p出度的数据置为0
        p = q.pop()
        res.append(p)
        for i in range(n):
            if g[p][i] == 1:
                g[p][i] = 0  # 去掉连通
                # 如果结点i的入度为0则入队结点i
                in_degree_count = 0
                for u in g:
                    if u[i] == 1:
                        in_degree_count += 1
                if in_degree_count == 0:
                    q.insert(0, i)

    return res



# 2. The graph should have no loop in its topology.
def islooped(g):
    # G=g.copy()
    G=deepcopy(g)
    node_set = set()
    r = len(G)
    have_in_zero = True
    while have_in_zero:
        have_in_zero = False
        for i in range(r):
            if i not in node_set and not any([row[i] for row in G]):
                node_set.add(i)
                G[i] = [0] * r
                have_in_zero = True
                break
    return False if len(node_set) == r else True

def topoSort(G): # v:[type,[in_v1,in_v2,...]]
    in_degrees = [0] * len(G)
    for i in range(len(G)):
        for v in G[i][1]:
            if v !=100:
                in_degrees[v] += 1  # 每一个节点的入度
    # Q = [u for u in G if in_degrees[u] == 0]  # 入度为 0 的节点
    Q = []
    S = []
    for i in range(len(G)):
        if in_degrees[i] == 0:
            Q.append(i)
    while Q:
        u = Q.pop()  # 默认从最后一个移除
        S.append(u)
        for v in G[u][1]:
            if v != 100:
                in_degrees[v] -= 1  # 并移除其指向
                if in_degrees[v] == 0:
                    Q.append(v)
    return S


# 3. Each vertex of the graph has no more than three edges that reaches it or comes out of it.
def isinrange(G):
    n=len(G)
    for v in range(n):
        ind = 0
        outd = 0
        for i in range(n):
            ind += G[i][v]
            outd += G[v][i]
            if ind>3 or outd>3:
                return False
    return True

# 4. Functions needed
def get_input(g, v):
    n = len(g)
    inputs = []
    for i in range(0, n):
        if g[i][v] == 1:
            inputs.append(i)
    return inputs

def isEnd(g, v):
    n = len(g)
    outputs = []
    for i in range(0, n):
        if g[v][i] == 1:
            outputs.append(i)
    if len(outputs) == 0:
        return 1
    else:
        return 0

def getEnds(g):
    n = len(g)
    ends = []
    for i in range(0, n):
        if isEnd(g, i):
            ends.append(i)
    return ends

def isIsolated(g,v):
    mark=0
    n=len(g)
    for i in range(n):
        mark+=g[v][i]
        mark+=g[i][v]
    if mark==0:
        return True
    else:
        return False


# Code Format
# TYPE VECTOR [type,param1,param2,param3]
# ('*' represents a param that is not searchable but decided by related structure)
# ID   Type               Param           Param1          Param2        Param3
# 1    Conv               *in_channels    kernel_size
# 2    Sep Conv           *in_channels    kernel_size
# 3    MBv2Block          *in_channels    kernel_size     expansion
# 4    Max_pool           *in_channels    kernel_size
# 5    Avg_pool           *in_channels    kernel_size
# 6    CBAM Block         *in_channels    kernel_size
# ... To be EXPANDED
# g[i][0]:Type Vector
# g[i][1]:Next node Vector

def get_Prev(g,v):
    prev=[]
    for i in range(len(g)):
        if v in g[i][1]:
            prev.append(i)
    return prev
def get_Out(g):
    for i in range(len(g)):
        if len(g[i][1])==0:
            return i


class newcell(nn.Module):
    def __init__(self, in_channels,out_channels, g):
        super(newcell, self).__init__()
        self.in_channels = in_channels
        self.out_channels=out_channels
        self.outnum=0
        for v in g:
            if 100 in v[1]:
                self.outnum += 1
        self.mid_channels = out_channels*self.outnum
        self.g = g
        self.node_num=len(g)
        self.op = nn.ModuleList()
        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2= nn.Conv2d(self.mid_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.bn2 = nn.BatchNorm2d(self.out_channels)

        for i in range(self.node_num):
            if g[i][0][0] == 0:
                self.op.append(self.conv1)
            elif g[i][0][0] == 1:
                self.op.append(Conv(self.out_channels, self.out_channels, self.g[i][0][1]))
            elif g[i][0][0] == 2:
                self.op.append(MNv1Block(self.out_channels, self.g[i][0][1]))
            elif g[i][0][0] == 3:
                self.op.append(
                    MNv2Block(self.out_channels, kernel_size=self.g[i][0][1], expansion=self.g[i][0][2], stride=1))
            elif g[i][0][0] == 4:
                self.op.append(nn.MaxPool2d(self.g[i][0][1], 1, self.g[i][0][1] // 2))
            elif g[i][0][0] == 5:
                self.op.append(nn.AvgPool2d(self.g[i][0][1], 1, self.g[i][0][1] // 2))
            elif g[i][0][0] == 6:
                self.op.append(CBAModule(self.out_channels, self.g[i][0][1]))
            elif g[i][0][0] == 7:
                self.op.append(CoordinateAttention(self.out_channels, self.g[i][0][1]))
            else:
                print("Wrong type in node", i)
                return

    def forward(self,x):
        s = topoSort(self.g)
        outnodes=[]
        for v in s:
            if 100 in self.g[v][1]:
                outnodes.append(v)
        n = len(self.g)
        nodes = [None] * n
        out = None
        for v in s:
            if self.g[v][0][0] == 0:  # input_Node: No other inputs
                nodes[0] = F.relu(self.bn1(self.op[0](x)))
                continue
            prev = get_Prev(self.g, v)
            if len(prev) == 0:
                print("Invalid Encoding in Node", v)
                return
            prev_count = 0
            for p in prev:
                if prev_count == 0:
                    nodes[v] = self.op[v](nodes[p])
                else:
                    nodes[v] += self.op[v](nodes[p])
                prev_count += 1
        o=0
        for v in outnodes:
            if o==0:
                out=nodes[v]
            else:
                out=torch.cat((out,nodes[v]),dim=1)
            o+=1
        out=F.relu(self.bn2(self.conv2(out)))
        return out

class cfnet(nn.Module):
    def __init__(self, in_channels, g,N):
        super(cfnet,self).__init__()
        self.in_channels=in_channels
        # self.out_channel=out_channels
        self.stack_num=N
        self.g=g
        self.image_size=32
        self.out_channels=[64,128,384]
        self.stage1 = nn.ModuleList()
        self.stage2 = nn.ModuleList()
        self.stage3 = nn.ModuleList()
        self.fc1=nn.Linear(32*8*8,128)
        self.fc2=nn.Linear(128,10)
        self.conv2 = nn.Conv2d(384, 32, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(32)
        for i in range(self.stack_num[0]):
            if i == 0:
                self.stage1.append(newcell(self.in_channels,self.out_channels[0],self.g))
            else:
                self.stage1.append(newcell(self.out_channels[0], self.out_channels[0], self.g))

        for j in range(self.stack_num[1]):
            if j==0:
                self.stage2.append(newcell(self.out_channels[0],self.out_channels[1],self.g))
            else:
                self.stage2.append(newcell(self.out_channels[1], self.out_channels[1], self.g))
        for k in range(self.stack_num[2]):
            if k==0:
                self.stage3.append(newcell(self.out_channels[1], self.out_channels[2], self.g))
            else:
                self.stage3.append(newcell(self.out_channels[2], self.out_channels[2], self.g))

    def forward(self,x):
        for i in range(self.stack_num[0]):
            x=self.stage1[i](x)
        x=soft_pool2d(x=x,kernel_size=2,stride=2)
        for i in range(self.stack_num[1]):
            x=self.stage2[i](x)
        x=soft_pool2d(x=x,kernel_size=2,stride=2)
        for i in range(self.stack_num[2]):
            x=self.stage3[i](x)
        x=F.relu(self.bn(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x=nn.Dropout(0.2)(x)
        x=F.relu(self.fc1(x))
        x=nn.Dropout(0.2)(x)
        x =self.fc2(x)
        return x



























class cell(nn.Module):
    def __init__(self, in_channels,out_channels, g):
        super(cell, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.g = g
        self.node_num=len(g)
        # code:[[type1,next_node],[type2,next_node],...]
        # type=0->input
        # type=-1->output
        self.op=nn.ModuleList()
        self.conv=nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1,padding=0,bias=False)

        self.bn=nn.BatchNorm2d(out_channels)
        # self.activate= MetaAconC(out_channels, 16)

        for i in range(self.node_num):
            # print("node",i,"type",g[i][0][0])
            if g[i][0][0]==0:
                self.op.append(self.conv)
            elif g[i][0][0]==1:
                self.op.append(Conv(self.out_channels,self.out_channels, self.g[i][0][1]))
            elif g[i][0][0]==2:
                self.op.append(MNv1Block(self.out_channels, self.g[i][0][1]))
            elif g[i][0][0]==3:
                self.op.append(MNv2Block(self.out_channels, kernel_size=self.g[i][0][1],expansion=self.g[i][0][2], stride=1))
            elif g[i][0][0]==4:
                self.op.append(nn.MaxPool2d(self.g[i][0][1], 1, self.g[i][0][1]//2))
            elif g[i][0][0]==5:
                self.op.append(nn.AvgPool2d(self.g[i][0][1], 1, self.g[i][0][1]//2))
            elif g[i][0][0]==6:
                self.op.append(CBAModule(self.out_channels, self.g[i][0][1]))
            elif g[i][0][0]==7:
                self.op.append(CoordinateAttention(self.out_channels,self.g[i][0][1]))
            else:
                print("Wrong type in node",i)
                return
    def  forward(self,x):
        s=topoSort(self.g)
        n = len(self.g)
        nodes=[None]*n
        out=None
        for v in s:
            if v==0 or self.g[v][0][0]==0: # input_Node: No other inputs
                nodes[0]=F.relu(self.bn(self.conv(x)))
                continue
            prev=get_Prev(self.g,v)
            if len(prev)==0:
                print("Invalid Encoding in Node",v)
                return
            prev_count=0
            for p in prev:
                if prev_count==0:
                    nodes[v] = self.op[v](nodes[p])
                else:
                    nodes[v] += self.op[v](nodes[p])
                prev_count += 1
            if len(self.g[v][1])==0:
                out=nodes[v]
        return out

# new cell--new rules

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

from torch.autograd import Variable



class redCell(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(redCell,self).__init__()
        # [[[0, 0, 0], [1, 100]], [[2, 3, 0], [2, 100]], [[3, 3, 8], [100]]]
        self.fr=FactorizedReduce(out_channels,out_channels)
        self.node1=MNv1Block(out_channels,3,2)
        self.node2=MNv2Block(out_channels,3,4,1)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(3*out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self,x):
        # x=F.relu(self.bn1(self.conv1(x))) # size,out_channels
        y1=self.fr(x) # size/4, out_channels
        y2=self.node1(x) # size/4,out_channels
        y3=self.node2(y2) # size/4,out_channels
        out=torch.cat([y1,y2,y3],dim=1)
        out = F.relu(self.bn2(self.conv2(out)))
        return out


class Netcf(nn.Module):
    def __init__(self, in_channels, g, N, aux, drop_prob):
        super(Netcf, self).__init__()
        self.in_channels = in_channels
        # self.out_channel=out_channels
        self.stack_num = N
        self.g = g
        self.image_size = 32
        self.out_channels = [64, 128, 256]
        self.stage1 = nn.ModuleList()
        self.stage2 = nn.ModuleList()
        self.stage3 = nn.ModuleList()
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)
        self.conv2 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv3 = nn.Conv2d(64, 10, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(10)
        self.aux = aux
        self.dp = drop_prob
        self.red0=redCell(self.out_channels[0],self.out_channels[0])
        self.red1=redCell(self.out_channels[1],self.out_channels[1])
        if aux:
            self.auxhead = AuxiliaryHeadCIFAR(self.out_channels[1], 10)

        for i in range(self.stack_num[0]):
            if i == 0:
                self.stage1.append(cellwithdp(self.in_channels, self.out_channels[0], self.g, self.dp))
            else:
                self.stage1.append(cellwithdp(self.out_channels[0], self.out_channels[0], self.g, self.dp))

        for j in range(self.stack_num[1]):
            if j == 0:
                self.stage2.append(cellwithdp(self.out_channels[0], self.out_channels[1], self.g, self.dp))
            else:
                self.stage2.append(cellwithdp(self.out_channels[1], self.out_channels[1], self.g, self.dp))

        for k in range(self.stack_num[2]):
            if k == 0:
                self.stage3.append(cellwithdp(self.out_channels[1], self.out_channels[2], self.g, self.dp))
            else:
                self.stage3.append(cellwithdp(self.out_channels[2], self.out_channels[2], self.g, self.dp))

    def forward(self, x):
        aux_out = None
        for i in range(self.stack_num[0]):
            x = self.stage1[i](x)
        x=self.red0(x)
        # x = soft_pool2d(x=x, kernel_size=2, stride=2)
        # x = nn.AvgPool2d(2, 2)(x)
        # x=nn.MaxPool2d(2,2)(x)
        for i in range(self.stack_num[1]):
            x = self.stage2[i](x)
        # x = soft_pool2d(x=x, kernel_size=2, stride=2)
        x=self.red1(x)
        # x = nn.AvgPool2d(2, 2)(x)
        # x=self.stage3[0](x)
        if self.aux and self.training:
            aux_out = self.auxhead(x)
        for i in range(0, self.stack_num[2]):
            x = self.stage3[i](x)

        x = F.relu(self.bn1(self.conv2(x)))
        x = F.relu(self.bn2(self.conv3(x)))
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        out = x.view(x.size(0), -1)

        return out, aux_out

# more than one out node: all their outputs are concat together.


class AuxiliaryHeadCIFAR(nn.Module):

  def __init__(self, C, num_classes):
    """assuming input size 8x8"""
    super(AuxiliaryHeadCIFAR, self).__init__()
    self.features = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False), # image size = 2 x 2
      nn.Conv2d(C, 128, 1, bias=False),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True),
      nn.Conv2d(128, 256, 2, bias=False),
      nn.BatchNorm2d(256),
      nn.ReLU(inplace=True)
    )
    self.classifier = nn.Linear(256, num_classes)

  def forward(self, x):
    x = self.features(x)
    x = self.classifier(x.view(x.size(0),-1))
    return x

class NetCF(nn.Module):
    def __init__(self, class_num,in_channels, g,N,aux,drop_prob):
        super(NetCF,self).__init__()
        self.in_channels=in_channels
        # self.out_channel=out_channels
        self.stack_num=N
        self.g=g
        self.image_size=32
        self.out_channels=[64,128,256]
        self.stage1 = nn.ModuleList()
        self.stage2 = nn.ModuleList()
        self.stage3 = nn.ModuleList()
        #self.fc1=nn.Linear(64*8*8,256)
        #self.fc2=nn.Linear(256,10)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv3 = nn.Conv2d(128, class_num, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(class_num)
        self.aux=aux
        self.dp=drop_prob

        if aux:
            self.auxhead=AuxiliaryHeadCIFAR(self.out_channels[1],10)

        for i in range(self.stack_num[0]):
            if i == 0:
                self.stage1.append(cellwithdp(self.in_channels,self.out_channels[0],self.g,self.dp))
            else:
                self.stage1.append(cellwithdp(self.out_channels[0], self.out_channels[0], self.g,self.dp))

        for j in range(self.stack_num[1]):
            if j == 0:
                self.stage2.append(cellwithdp(self.out_channels[0],self.out_channels[1],self.g,self.dp))
            else:
                self.stage2.append(cellwithdp(self.out_channels[1], self.out_channels[1], self.g,self.dp))

        for k in range(self.stack_num[2]):
            if k == 0:
                self.stage3.append(cellwithdp(self.out_channels[1], self.out_channels[2], self.g,self.dp))
            else:
                self.stage3.append(cellwithdp(self.out_channels[2], self.out_channels[2], self.g,self.dp))

    def forward(self,x):
        aux_out=None
        for i in range(self.stack_num[0]):
            x=self.stage1[i](x)

        x=soft_pool2d(x=x,kernel_size=2,stride=2)
        # x = nn.AvgPool2d(2, 2)(x)
        # x=nn.MaxPool2d(2,2)(x)
        for i in range(self.stack_num[1]):
            x=self.stage2[i](x)
        x=soft_pool2d(x=x,kernel_size=2,stride=2)


        # x = nn.AvgPool2d(2, 2)(x)
        # x=self.stage3[0](x)
        if self.aux and self.training:
            aux_out=self.auxhead(x)
        for i in range(0,self.stack_num[2]):
            x=self.stage3[i](x)

        x=F.relu(self.bn1(self.conv2(x)))
        x=F.relu(self.bn2(self.conv3(x)))
        x=nn.functional.adaptive_avg_pool2d(x,(1,1))
        out = x.view(x.size(0), -1)

        return out #,aux_out

class cfnet_c(nn.Module):
    def __init__(self, in_channels, g,N):
        super(cfnet_c,self).__init__()
        self.in_channels=in_channels
        # self.out_channel=out_channels
        self.stack_num=N
        self.g=g
        self.image_size=32
        self.out_channels=[32,128,256]
        self.stage1 = nn.ModuleList()
        self.stage2 = nn.ModuleList()
        self.stage3 = nn.ModuleList()
        self.fc1=nn.Linear(64*8*8,256)
        self.fc2=nn.Linear(256,10)
        self.conv2 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv3 = nn.Conv2d(64, 10, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(10)
        for i in range(self.stack_num[0]):
            if i == 0:
                self.stage1.append(newcell(self.in_channels,self.out_channels[0],self.g))
            else:
                self.stage1.append(newcell(self.out_channels[0], self.out_channels[0], self.g))

        for j in range(self.stack_num[1]):
            if j == 0:
                self.stage2.append(newcell(self.out_channels[0],self.out_channels[1],self.g))
            else:
                self.stage2.append(newcell(self.out_channels[1], self.out_channels[1], self.g))

        for k in range(self.stack_num[2]):
            if k == 0:
                self.stage3.append(newcell(self.out_channels[1], self.out_channels[2], self.g))
            else:
                self.stage3.append(newcell(self.out_channels[2], self.out_channels[2], self.g))

    def forward(self,x):
        for i in range(self.stack_num[0]):
            x=self.stage1[i](x)

        x=soft_pool2d(x=x,kernel_size=2,stride=2)
        # x = nn.AvgPool2d(2, 2)(x)
        # x=nn.MaxPool2d(2,2)(x)
        for i in range(self.stack_num[1]):
            x=self.stage2[i](x)
        x=soft_pool2d(x=x,kernel_size=2,stride=2)
        # x = nn.AvgPool2d(2, 2)(x)
        for i in range(self.stack_num[2]):
            x=self.stage3[i](x)

        x=F.relu(self.bn1(self.conv2(x)))
        x=F.relu(self.bn2(self.conv3(x)))
        x=nn.functional.adaptive_avg_pool2d(x,(1,1))
        x = x.view(x.size(0), -1)

        # x=nn.Dropout(0.2)(x)
        #  in_dim = x.shape[1]
        # print(in_dim)
        # x=F.relu(self.fc1(x))
        # x=nn.Dropout(0.2)(x)
        # x =self.fc2(x)

        return x

class mednet(nn.Module):
    def __init__(self, in_channels, g,N,num_class, isActivated):
        super(mednet,self).__init__()
        self.in_channels=in_channels
        # self.out_channel=out_channels
        self.stack_num=N
        self.g=g
        self.isA=isActivated
        self.image_size=32
        self.num_class=num_class
        self.out_channels=[32,128,256]
        self.stage1 = nn.ModuleList()
        self.stage2 = nn.ModuleList()
        self.stage3 = nn.ModuleList()
        self.conv2 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv3 = nn.Conv2d(64, self.num_class, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(self.num_class)
        for i in range(self.stack_num[0]):
            if i == 0:
                self.stage1.append(newcell(self.in_channels,self.out_channels[0],self.g))
            else:
                self.stage1.append(newcell(self.out_channels[0], self.out_channels[0], self.g))

        for j in range(self.stack_num[1]):
            if j == 0:
                self.stage2.append(newcell(self.out_channels[0],self.out_channels[1],self.g))
            else:
                self.stage2.append(newcell(self.out_channels[1], self.out_channels[1], self.g))

        for k in range(self.stack_num[2]):
            if k == 0:
                self.stage3.append(newcell(self.out_channels[1], self.out_channels[2], self.g))
            else:
                self.stage3.append(newcell(self.out_channels[2], self.out_channels[2], self.g))

    def forward(self,x):
        for i in range(self.stack_num[0]):
            x=self.stage1[i](x)

        x=soft_pool2d(x=x,kernel_size=2,stride=2)
        # x = nn.AvgPool2d(2, 2)(x)
        # x=nn.MaxPool2d(2,2)(x)
        for i in range(self.stack_num[1]):
            x=self.stage2[i](x)
        x=soft_pool2d(x=x,kernel_size=2,stride=2)
        # x = nn.AvgPool2d(2, 2)(x)
        for i in range(self.stack_num[2]):
            x=self.stage3[i](x)
        x=F.relu(self.bn1(self.conv2(x)))
        if self.isA == 0:
            x=F.relu(self.conv3(x))
        elif self.isA == 1:
            x=F.relu(self.bn2(self.conv3(x)))
        x=nn.functional.adaptive_avg_pool2d(x,(1,1))
        x = x.view(x.size(0), -1)
        # x=torch.sigmoid(x)
        return x


class mednet_large(nn.Module):
    def __init__(self, in_channels, g,N,num_class, isActivated):
        super(mednet_large,self).__init__()
        self.in_channels=in_channels
        # self.out_channel=out_channels
        self.stack_num=N
        self.g=g
        self.isA=isActivated
        self.image_size=224
        self.num_class=num_class
        self.out_channels=[32,64,128,256]
        self.stage1 = nn.ModuleList()
        self.stage2 = nn.ModuleList()
        self.stage3 = nn.ModuleList()
        self.stage4 = nn.ModuleList()
        self.conv2 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv3 = nn.Conv2d(256, self.num_class, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(self.num_class)
        for i in range(self.stack_num[0]):
            if i == 0:
                self.stage1.append(newcell(self.in_channels,self.out_channels[0],self.g))
            else:
                self.stage1.append(newcell(self.out_channels[0], self.out_channels[0], self.g))

        for j in range(self.stack_num[1]):
            if j == 0:
                self.stage2.append(newcell(self.out_channels[0],self.out_channels[1],self.g))
            else:
                self.stage2.append(newcell(self.out_channels[1], self.out_channels[1], self.g))

        for k in range(self.stack_num[2]):
            if k == 0:
                self.stage3.append(newcell(self.out_channels[1], self.out_channels[2], self.g))
            else:
                self.stage3.append(newcell(self.out_channels[2], self.out_channels[2], self.g))
        for t in range(self.stack_num[3]):
            if t == 0:
                self.stage4.append(newcell(self.out_channels[2], self.out_channels[3], self.g))
            else:
                self.stage4.append(newcell(self.out_channels[3], self.out_channels[3], self.g))

    def forward(self,x):
        for i in range(self.stack_num[0]):
            x=self.stage1[i](x)

        x=soft_pool2d(x=x,kernel_size=2,stride=2)
        # x = nn.AvgPool2d(2, 2)(x)
        # x=nn.MaxPool2d(2,2)(x)
        for i in range(self.stack_num[1]):
            x=self.stage2[i](x)
        x=soft_pool2d(x=x,kernel_size=2,stride=2)
        # x = nn.AvgPool2d(2, 2)(x)
        for i in range(self.stack_num[2]):
            x=self.stage3[i](x)
        x = soft_pool2d(x=x, kernel_size=2, stride=2)
        # x = nn.AvgPool2d(2, 2)(x)
        for i in range(self.stack_num[3]):
            x = self.stage4[i](x)

        # x=F.relu(self.bn1(self.conv2(x)))
        if self.isA == 0:
            x=F.relu(self.conv3(x))
        elif self.isA == 1:
            x=F.relu(self.bn2(self.conv3(x)))
        x=nn.functional.adaptive_avg_pool2d(x,(1,1))
        x = x.view(x.size(0), -1)

        return x


# ARCH MATRIX [e]n*n
# Arch Matrix uses adjacency matrix's form to describe the topology structure of the individual.

def get_outchannels(in_channels,type,arch):
    g = np.array(arch)
    #type=tp
    t = np.array(arch)
    # print(g)
    s = topo_sort(t)
    # print('Toposort',s)
    n = len(arch)
    out_channels=0
    nodes = np.array([None] * n)  # 节点初始化
    for v in s:
        #print('Processing node', v)
        #print('type v', type[v])
        # print(g)
        # print('input of', v, 'is', get_input(g, v))
        # 是否是输入节点?
        if len(get_input(g, v)) == 0:
            # nodes[v] = main_input
            if type[v][0] == 1:
                type[v][1] = in_channels

            elif type[v][0] == 2:

                type[v][1] = in_channels
                type[v][2] = in_channels

            elif type[v][0] == 3:
                type[v][1] = in_channels
                type[v][2] = in_channels
            elif type[v][0] == 4:
                type[v][1] = in_channels
                type[v][2] = in_channels
            elif type[v][0] == 5:
                type[v][1] = in_channels

                type[v][2] = in_channels
            elif type[v][0] == 6:
                type[v][1] = in_channels
                type[v][2] = in_channels
            else:
                print('Wrong type!')
                return
            # isCaled[v] = 1
        # 查找前驱节点
        if len(get_input(g, v)) == 1:
            if type[v][0] == 1:
                type[v][1] = type[get_input(g, v)[0]][2]  # 修改in_channels

            elif type[v][0] == 2:
                type[v][1] = type[get_input(g, v)[0]][2]  # 修改in_channels
                type[v][2] = type[get_input(g, v)[0]][2]  # 修改out_channels

            elif type[v][0] == 3:
                type[v][1] = type[get_input(g, v)[0]][2]  # 修改in_channels
                type[v][2] = type[get_input(g, v)[0]][2]  # 修改out_channels

            elif type[v][0] == 4:
                type[v][1] = type[get_input(g, v)[0]][2]  # 修改in_channels
                type[v][2] = type[get_input(g, v)[0]][2]  # 修改out_channels
            elif type[v][0] == 5:
                type[v][1] = type[get_input(g, v)[0]][2]  # 修改in_channels
                type[v][2] = type[get_input(g, v)[0]][2]  # 修改out_channels

            elif type[v][0] == 6:
                type[v][1] = type[get_input(g, v)[0]][2]  # 修改in_channels
                type[v][2] = type[get_input(g, v)[0]][2]  # 修改out_channels
            else:
                print('Wrong type!')
                return
            # isCaled[v] = 1
        if len(get_input(g, v)) > 1:
            # print('feature',feature)
            # print(v, feature)
            if type[v][0] == 1:
                type[v][1] = 0  # 修改in_channels
                for i in get_input(g, v):
                    type[v][1] += type[i][2]

            elif type[v][0] == 2:
                type[v][1] = 0
                for i in get_input(g, v):  # 修改in_channels
                    type[v][1] += type[i][2]
                type[v][2] = type[v][1]  # 修改out_channels

            elif type[v][0] == 3:
                type[v][1] = 0
                for i in get_input(g, v):  # 修改in_channels
                    type[v][1] += type[i][2]
                type[v][2] = type[get_input(g, v)[0]][2]  # 修改out_channels

            elif type[v][0] == 4:
                type[v][1] = 0
                for i in get_input(g, v):  # 修改in_channels
                    type[v][1] += type[i][2]
                type[v][2] = type[v][1]  # 修改out_channels

            elif type[v][0] == 5:
                type[v][1] = 0
                for i in get_input(g, v):  # 修改in_channels
                    type[v][1] += type[i][2]
                type[v][2] = type[v][1]  # 修改out_channels

            elif type[v][0] == 6:
                type[v][1] = 0
                for i in get_input(g, v):  # 修改in_channels
                    type[v][1] += type[i][2]
                # print('attention', nodes[v])
                type[v][2] = type[v][1]  # 修改out_channels
            else:
                print('Wrong type!')
                return

        # print(v,nodes[v])
        # nodes[v] = keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(feature)
    # for i in range(0, n):
    # print(nodes[i])
    # print(s)
    # print('end', getEnds(g))
    if len(getEnds(g)) > 1:
        for i in getEnds(g):
            out_channels += type[i][2]
            print(type[i][2])
    else:
        out_channels = type[getEnds(g)[0]][2]
    
    # return type,out_channels
    return out_channels

# 这里输入的Type中inchannels应当已经为处理成功的Type值
# 测试方案，输入错误的inchannel编码以及正确的inchannel比较输出模型结果是否一致且正确。
class Arch(nn.Module):
    def __init__(self,image_size,in_channels,type,arch):
        # node_num = n
        # type [n,4]
        # arch
        super(Arch, self).__init__()
        self.image_size=image_size
        self.in_channels=in_channels
        self.node_num=len(type)
        self.type=type
        self.arch=arch
        self.op=nn.ModuleList()
        self.out_channels = 0
        for v in range(self.node_num):
            # print(type)
            if self.type[v][0] == 1:
                self.op.append(Conv(type[v][1], self.type[v][2], self.type[v][3]))
            elif self.type[v][0] == 2:
                self.op.append(nn.MaxPool2d(self.type[v][3], 1, self.type[v][3]//2))
            elif self.type[v][0] == 3:
                self.op.append(nn.AvgPool2d(self.type[v][3], 1, self.type[v][3]//2))
            elif self.type[v][0] == 4:
                #self.op.append(SpatialAttention(self.type[v][3]))
                self.op.append(CBAModule(self.type[v][3],self.type[v][1]))
            elif self.type[v][0] == 5:
                #self.op.append(ChannelAttention(self.type[v][1]))
                self.op.append(CoordinateAttention(self.type[v][1]))
            elif self.type[v][0] == 6:
                print(' ')
                # self.op.append(AxialAttentionBlock(self.type[v][1], self.image_size, self.type[v][4]))
            else:
                print('Wrong type!')
                return

    def forward(self,x):
        # print('Method: Arch.forward')
        g=np.array(self.arch)
        t=np.array(self.arch)
        # print(g)
        s=topo_sort(t)
        # print('Toposort',s)
        n=self.node_num
        nodes=np.array([None]*n) #节点初始化
        for v in s:
            # print('Processing node',v)
            # print('type v',self.type[v])
            # print(g)
            #print('input of', v, 'is', get_input(g, v))
            # 是否是输入节点?
            if len(get_input(g, v)) == 0:
                # nodes[v] = main_input
                if self.type[v][0] == 1:
                    self.type[v][1] = self.in_channels
                    nodes[v] = self.op[v](x)
                elif self.type[v][0] == 2:
                    nodes[v] = self.op[v](x)
                    self.type[v][1]=self.in_channels
                    self.type[v][2] = self.in_channels

                elif self.type[v][0] == 3:
                    nodes[v] = self.op[v](x)
                    #nodes[v]=soft_pool2d(x,)
                    self.type[v][1] = self.in_channels
                    self.type[v][2] = self.in_channels
                elif self.type[v][0] == 4:
                    nodes[v] = self.op[v](x)
                    self.type[v][1] = self.in_channels
                    self.type[v][2] = self.in_channels
                elif self.type[v][0] == 5:
                    self.type[v][1] = self.in_channels
                    nodes[v] = self.op[v](x)
                    self.type[v][2] = self.in_channels
                elif self.type[v][0] == 6:
                    self.type[v][1] = self.in_channels
                    nodes[v] = self.op[v](x)
                    self.type[v][2] = self.in_channels
                else:
                    print('Wrong type!')
                    return
                # isCaled[v] = 1
            # 查找前驱节点
            if len(get_input(g, v)) == 1:
                if self.type[v][0] == 1:
                    self.type[v][1]=self.type[get_input(g,v)[0]][2]   # 修改in_channels
                    nodes[v] = self.op[v](nodes[get_input(g,v)[0]])

                elif self.type[v][0] == 2:
                    self.type[v][1] = self.type[get_input(g, v)[0]][2]   # 修改in_channels
                    nodes[v] = self.op[v](nodes[get_input(g, v)[0]])
                    self.type[v][2] = self.type[get_input(g, v)[0]][2]   # 修改out_channels

                elif self.type[v][0] == 3:
                    self.type[v][1] = self.type[get_input(g, v)[0]][2]    # 修改in_channels
                    nodes[v] = self.op[v](nodes[get_input(g,v)[0]])
                    self.type[v][2] = self.type[get_input(g, v)[0]][2]    # 修改out_channels

                elif self.type[v][0] == 4:
                    self.type[v][1] = self.type[get_input(g, v)[0]][2]    # 修改in_channels
                    nodes[v] = self.op[v](nodes[get_input(g,v)[0]])
                    self.type[v][2] = self.type[get_input(g, v)[0]][2]    # 修改out_channels
                elif self.type[v][0] == 5:
                    self.type[v][1] = self.type[get_input(g, v)[0]][2]    # 修改in_channels
                    nodes[v] = self.op[v](nodes[get_input(g,v)[0]])
                    self.type[v][2] = self.type[get_input(g, v)[0]][2]  # 修改out_channels

                elif self.type[v][0] == 6:
                    self.type[v][1] = self.type[get_input(g, v)[0]][2]    # 修改in_channels
                    # print(nodes[get_input(g,v)[0]].shape)
                    nodes[v] = self.op[v](nodes[get_input(g,v)[0]])

                    self.type[v][2] = self.type[get_input(g, v)[0]][2]  # 修改out_channels
                else:
                    print('Wrong type!')
                    return
                # isCaled[v] = 1
            if len(get_input(g, v)) > 1:
                prior = []  # 前置节点
                for i in range(0, len(get_input(g, v))):
                    prior.append(nodes[get_input(g, v)[i]])
                feature = torch.cat(prior,1)
                # print(feature.shape)
                # print('feature',feature)
                # print(v, feature)
                if self.type[v][0] == 1:
                    self.type[v][1] = 0  # 修改in_channels
                    for i in get_input(g, v):
                        self.type[v][1] += self.type[i][2]
                    nodes[v] = self.op[v](feature)

                elif self.type[v][0] == 2:
                    self.type[v][1] = 0
                    for i in get_input(g, v):  # 修改in_channels
                        self.type[v][1] += self.type[i][2]
                    nodes[v] = self.op[v](feature)
                    self.type[v][2] = self.type[v][1]  # 修改out_channels

                elif self.type[v][0] == 3:
                    self.type[v][1] = 0
                    for i in get_input(g, v):  # 修改in_channels
                        self.type[v][1] += self.type[i][2]
                    nodes[v] = self.op[v](feature)
                    self.type[v][2] = self.type[v][1]  # 修改out_channels

                elif self.type[v][0] == 4:
                    self.type[v][1] = 0
                    for i in get_input(g, v):  # 修改in_channels
                        self.type[v][1] += self.type[i][2]
                    nodes[v] = self.op[v](feature)
                    self.type[v][2] = self.type[v][1]  # 修改out_channels

                elif self.type[v][0] == 5:
                    self.type[v][1] = 0
                    for i in get_input(g, v):  # 修改in_channels
                        self.type[v][1] += self.type[i][2]
                    nodes[v] = self.op[v](feature)
                    self.type[v][2] = self.type[v][1]  # 修改out_channels

                elif self.type[v][0] == 6:
                    self.type[v][1] = 0
                    for i in get_input(g, v):  # 修改in_channels
                        self.type[v][1] += self.type[i][2]
                    nodes[v] = self.op[v](feature)
                    # print('attention', nodes[v])
                    self.type[v][2] = self.type[v][1]  # 修改out_channels
                else:
                    print('Wrong type!')
                    return

            # print(v,nodes[v])
                # nodes[v] = keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(feature)
        # for i in range(0, n):
        # print(nodes[i])
        # print(s)
        # print('end',getEnds(g))
        if len(getEnds(g))>1:
            post = []
            for i in getEnds(g):
                post.append(nodes[i])
                self.out_channels += self.type[i][2]
            res = torch.cat(post,1)
        else:
            res=nodes[getEnds(g)[0]]
        '''
        res = res.view(res.size(0), -1)
        res = nn.Dropout(0.2)(res)
        in_dim = res.shape[1]
        res = F.relu(nn.Linear(in_dim, 64)(res))
        res = nn.Dropout(0.2)(res)
        res = F.relu(nn.Linear(64, 10)(res))
        '''
        return res





class Net(nn.Module):
    def __init__(self,image_size,in_channels,type,arch):
        super(Net,self).__init__()
        # print('Net Initialization')
        self.image_size=image_size
        self.in_channels=in_channels
        self.num_node=len(type)
        self.type=type
        self.arch = arch
        self.channel=self.get_outchannels(self.in_channels)
        # print(self.type)
        # print(self.channel)
        self.cell1=Arch(self.image_size,self.in_channels,self.type,self.arch)
        # print(self.type)
        out_channel=self.get_outchannels(self.channel)
        self.cell2=Arch(int(self.image_size/2),self.channel,self.type,self.arch)
        self.dense1=nn.Linear(out_channel*int(self.image_size/4)*int(self.image_size/4),64)
        # print()
        self.dense2=nn.Linear(64,10)
    def forward(self,x):
        # x=Arch(self.image_size,self.in_channels,self.num_node,self.type,self.arch)(x)
        # cell1=Arch(self.image_size,self.in_channels,self.num_node,self.type,self.arch)
        #print(x.shape)
        x=self.cell1(x)
        # print(x.shape)
        x=soft_pool2d(x,2,2)
        # print("cell1",x.shape)
        # channel=x.shape[1]
        # self.image_size=int(self.image_size/2)
        # cell2=Arch(self.image_size,channel,self.num_node,self.type,self.arch)
        # print(x.shape)
        # x=Arch(self.image_size,channel,self.num_node,self.type,self.arch)(x)
        x=self.cell2(x)
        #print(x.shape)
        x=nn.AvgPool2d(2)(x)
        # print('cell2',x.shape)
        x = x.view(x.size(0), -1)
        x=nn.Dropout(0.2)(x)
        #  in_dim = x.shape[1]
        # print(in_dim)
        x=F.relu(self.dense1(x))
        x=nn.Dropout(0.2)(x)
        x = F.relu(self.dense2(x))

        return x

    def get_outchannels(self,in_channels):
        # print('Method: get_outchannels')
        g = np.array(self.arch)
        #type=tp
        t = np.array(self.arch)
        # print(g)
        s = topo_sort(t)
        # print('Toposort',s)
        n = self.num_node
        out_channels=0
        nodes = np.array([None] * n)  # 节点初始化
        for v in s:
            # print('Processing node', v)
            # print('type v', self.type[v])
            # print(g)
            # print('input of', v, 'is', get_input(g, v))
            # 是否是输入节点?
            if len(get_input(g, v)) == 0:
                # nodes[v] = main_input
                if self.type[v][0] == 1:
                    self.type[v][1] = in_channels

                elif self.type[v][0] == 2:

                    self.type[v][1] = in_channels
                    self.type[v][2] = in_channels

                elif self.type[v][0] == 3:
                    self.type[v][1] = in_channels
                    self.type[v][2] = in_channels
                elif self.type[v][0] == 4:
                    self.type[v][1] = in_channels
                    self.type[v][2] = in_channels
                elif self.type[v][0] == 5:
                    self.type[v][1] = in_channels

                    self.type[v][2] = in_channels
                elif self.type[v][0] == 6:
                    self.type[v][1] = in_channels
                    self.type[v][2] = in_channels
                else:
                    print('Wrong type!')
                    return
                # isCaled[v] = 1
            # 查找前驱节点
            if len(get_input(g, v)) == 1:
                if self.type[v][0] == 1:
                    self.type[v][1] = self.type[get_input(g, v)[0]][2]  # 修改in_channels

                elif self.type[v][0] == 2:
                    self.type[v][1] = self.type[get_input(g, v)[0]][2]  # 修改in_channels
                    self.type[v][2] = self.type[get_input(g, v)[0]][2]  # 修改out_channels

                elif self.type[v][0] == 3:
                    self.type[v][1] = self.type[get_input(g, v)[0]][2]  # 修改in_channels
                    self.type[v][2] = self.type[get_input(g, v)[0]][2]  # 修改out_channels

                elif self.type[v][0] == 4:
                    self.type[v][1] = self.type[get_input(g, v)[0]][2]  # 修改in_channels
                    self.type[v][2] = self.type[get_input(g, v)[0]][2]  # 修改out_channels
                elif self.type[v][0] == 5:
                    self.type[v][1] = self.type[get_input(g, v)[0]][2]  # 修改in_channels
                    self.type[v][2] = self.type[get_input(g, v)[0]][2]  # 修改out_channels

                elif self.type[v][0] == 6:
                    self.type[v][1] = self.type[get_input(g, v)[0]][2]  # 修改in_channels
                    self.type[v][2] = self.type[get_input(g, v)[0]][2]  # 修改out_channels
                else:
                    print('Wrong type!')
                    return
                # isCaled[v] = 1
            if len(get_input(g, v)) > 1:
                # print('feature',feature)
                # print(v, feature)
                if self.type[v][0] == 1:
                    self.type[v][1] = 0  # 修改in_channels
                    for i in get_input(g, v):
                        self.type[v][1] += self.type[i][2]

                elif self.type[v][0] == 2:
                    self.type[v][1] = 0
                    for i in get_input(g, v):  # 修改in_channels
                        self.type[v][1] += self.type[i][2]
                    self.type[v][2] = self.type[v][1]  # 修改out_channels

                elif self.type[v][0] == 3:
                    self.type[v][1] = 0
                    for i in get_input(g, v):  # 修改in_channels
                        self.type[v][1] += self.type[i][2]
                    self.type[v][2] = self.type[v][1]  # 修改out_channels

                elif self.type[v][0] == 4:
                    self.type[v][1] = 0
                    for i in get_input(g, v):  # 修改in_channels
                        self.type[v][1] += self.type[i][2]
                    self.type[v][2] = self.type[v][1]  # 修改out_channels

                elif self.type[v][0] == 5:
                    self.type[v][1] = 0
                    for i in get_input(g, v):  # 修改in_channels
                        self.type[v][1] += self.type[i][2]
                    self.type[v][2] = self.type[v][1]  # 修改out_channels

                elif self.type[v][0] == 6:
                    self.type[v][1] = 0
                    for i in get_input(g, v):  # 修改in_channels
                        self.type[v][1] += self.type[i][2]
                    # print('attention', nodes[v])
                    self.type[v][2] = self.type[v][1]  # 修改out_channels
                else:
                    print('Wrong type!')
                    return

            # print(v,nodes[v])
            # nodes[v] = keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(feature)
        # for i in range(0, n):
        # print(nodes[i])
        # print(s)
        # print('end', getEnds(g))
        if len(getEnds(g)) > 1:
            for i in getEnds(g):
                out_channels += self.type[i][2]
        else:
            out_channels = self.type[getEnds(g)[0]][2]

        # return type,out_channels
        return out_channels


class StackNet(nn.Module):
    def __init__(self,image_size,in_channels,type,arch,CellNum):

        super(StackNet,self).__init__()
        # print('Net Initialization')
        self.image_size=image_size
        self.in_channels=in_channels
        self.num_node=len(type)
        self.type=type
        #print('chushi ',type)
        self.arch = arch
        self.channel=self.get_outchannels(self.in_channels)

        self.CellNum=CellNum
        # self.list_channels=[0]*self.CellNum
        self.cell_list=nn.ModuleList()
        self.imgsize_list=[image_size]*self.CellNum
        self.dense1 = nn.Linear(self.get_outchannels(self.channel)*int(self.image_size/4)*int(self.image_size/4),10)
        self.get_outchannels(self.in_channels)
        for i in range(self.CellNum):
            if i==0:
                #self.list_channels[i]=in_channels
                # print(self.type)
                #self.get_outchannels(self.list_channels[i])
                self.cell_list.append(Arch(self.imgsize_list[i], self.in_channels, self.type, self.arch))

            else:
                self.channel = self.get_outchannels(self.channel)
                if i >= (self.CellNum / 3):
                    self.imgsize_list[i] = int(self.image_size / 2)
                elif i>=(self.CellNum*2 / 3):
                    self.imgsize_list[i] = int(self.image_size / 4)
                self.cell_list.append(Arch(self.imgsize_list[i], self.channel, self.type, self.arch))

    '''
                else:

                    #print(i, self.list_channels[i-1])
                    self.channel=self.get_outchannels(self.channel)
                    if i >= (self.CellNum/2):
                        self.imgsize_list[i]=int(self.image_size/2)
                    self.cell_list.append(Arch(self.imgsize_list[i], self.channel, self.type, self.arch))
                '''



    #print(self.list_channels)
    # print(self.cell_list)
    # self.dense1=nn.Linear(self.get_outchannels(self.channel)*int(self.image_size/4)*int(self.image_size/4),10)
    # print()
    # self.dense2=nn.Linear(64,10)
    def forward(self,x):
        i=0
        while i<self.CellNum/3:
        # print(i)
            x=self.cell_list[i](x)
            i=i+1
        x = soft_pool2d(x, 2, 2)
        while i<2*self.CellNum/3:
            x=self.cell_list[i](x)
            i=i+1
        # print('cell2',x.shape)
        x = soft_pool2d(x, 2, 2)
        while i < self.CellNum:
            x=self.cell_list[i](x)
            i=i+1

        x = x.view(x.size(0), -1)
        x=nn.Dropout(0.2)(x)
        #  in_dim = x.shape[1]
        # print(in_dim)
        x=F.relu(self.dense1(x))
        #x=nn.Dropout(0.2)(x)
        #x = F.relu(self.dense2(x))

        return x

    def get_outchannels(self,in_channels):
        # print('Method: get_outchannels')
        g = np.array(self.arch)
        # type=tp
        t = np.array(self.arch)
        # print(g)
        s = topo_sort(t)
        # print('Toposort',s)
        n = self.num_node
        out_channels = 0
        nodes = np.array([None] * n)  # 节点初始化
        for v in s:
            # print('Processing node', v)
            # print('type v', self.type[v])
            # print(g)
            # print('input of', v, 'is', get_input(g, v))
            # 是否是输入节点?
            if len(get_input(g, v)) == 0:
                # nodes[v] = main_input
                if self.type[v][0] == 1:  # conv
                    self.type[v][1] = in_channels

                elif self.type[v][0] == 2:  # maxpooling

                    self.type[v][1] = in_channels
                    self.type[v][2] = in_channels

                elif self.type[v][0] == 3:  # avgpooling
                    self.type[v][1] = in_channels
                    self.type[v][2] = in_channels
                elif self.type[v][0] == 4:  # CBAModule
                    self.type[v][1] = in_channels
                    self.type[v][2] = in_channels
                elif self.type[v][0] == 5:  # CoordinAtten
                    self.type[v][1] = in_channels

                    self.type[v][2] = in_channels
                elif self.type[v][0] == 6:
                    self.type[v][1] = in_channels
                    self.type[v][2] = in_channels
                else:
                    print('Wrong type!')
                    return
                # isCaled[v] = 1
            # 查找前驱节点
            if len(get_input(g, v)) == 1:
                if self.type[v][0] == 1:
                    self.type[v][1] = self.type[get_input(g, v)[0]][2]  # 修改in_channels

                elif self.type[v][0] == 2:
                    self.type[v][1] = self.type[get_input(g, v)[0]][2]  # 修改in_channels
                    self.type[v][2] = self.type[get_input(g, v)[0]][2]  # 修改out_channels

                elif self.type[v][0] == 3:
                    self.type[v][1] = self.type[get_input(g, v)[0]][2]  # 修改in_channels
                    self.type[v][2] = self.type[get_input(g, v)[0]][2]  # 修改out_channels

                elif self.type[v][0] == 4:
                    self.type[v][1] = self.type[get_input(g, v)[0]][2]  # 修改in_channels
                    self.type[v][2] = self.type[get_input(g, v)[0]][2]  # 修改out_channels
                elif self.type[v][0] == 5:
                    self.type[v][1] = self.type[get_input(g, v)[0]][2]  # 修改in_channels
                    self.type[v][2] = self.type[get_input(g, v)[0]][2]  # 修改out_channels

                elif self.type[v][0] == 6:
                    self.type[v][1] = self.type[get_input(g, v)[0]][2]  # 修改in_channels
                    self.type[v][2] = self.type[get_input(g, v)[0]][2]  # 修改out_channels
                else:
                    print('Wrong type!')
                    return
                # isCaled[v] = 1
            if len(get_input(g, v)) > 1:
                # print('feature',feature)
                # print(v, feature)
                if self.type[v][0] == 1:
                    self.type[v][1] = 0  # 修改in_channels
                    for i in get_input(g, v):
                        self.type[v][1] += self.type[i][2]

                elif self.type[v][0] == 2:
                    self.type[v][1] = 0
                    for i in get_input(g, v):  # 修改in_channels
                        self.type[v][1] += self.type[i][2]
                    self.type[v][2] = self.type[v][1]  # 修改out_channels

                elif self.type[v][0] == 3:
                    self.type[v][1] = 0
                    for i in get_input(g, v):  # 修改in_channels
                        self.type[v][1] += self.type[i][2]
                    self.type[v][2] = self.type[v][1]  # 修改out_channels

                elif self.type[v][0] == 4:
                    self.type[v][1] = 0
                    for i in get_input(g, v):  # 修改in_channels
                        self.type[v][1] += self.type[i][2]
                    self.type[v][2] = self.type[v][1]  # 修改out_channels

                elif self.type[v][0] == 5:
                    self.type[v][1] = 0
                    for i in get_input(g, v):  # 修改in_channels
                        self.type[v][1] += self.type[i][2]
                    self.type[v][2] = self.type[v][1]  # 修改out_channels

                elif self.type[v][0] == 6:
                    self.type[v][1] = 0
                    for i in get_input(g, v):  # 修改in_channels
                        self.type[v][1] += self.type[i][2]
                    # print('attention', nodes[v])
                    self.type[v][2] = self.type[v][1]  # 修改out_channels
                else:
                    print('Wrong type!')
                    return

            # print(v,nodes[v])
            # nodes[v] = keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(feature)
        # for i in range(0, n):
        # print(nodes[i])
        # print(s)
        # print('end', getEnds(g))
        if len(getEnds(g)) > 1:
            for i in getEnds(g):
                out_channels += self.type[i][2]
        else:
            out_channels = self.type[getEnds(g)[0]][2]

        # return type,out_channels
        return out_channels


