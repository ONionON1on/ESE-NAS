import torch

import torch.nn as nn

from nodes import *

from decode import *

from Train import *

import numpy as np


def isSameSize(dict1, dict2):
    result = 0
    # 判断传入的两个参数是否都是字典，只要有一个不是字典，则停止比较
    if not isinstance(dict1, dict) or not isinstance(dict2, dict):
        #print("传入的参数不全是dict")
        return False

    # 比较长度值大小
    if len(dict1.keys()) != len(dict2.keys()):
        #print( "dict长度不同" )
        return False

    # 字典1的key列表
    dict1_keys_list = list(dict1.keys())
    # 字典2的key列表
    dict2_keys_list = list(dict2.keys())

    # 遍历key列表索引
    for i in range(len(dict1.keys())):

        # 判断两个字典key的hash值是否相等，hash值相同，说明key相同
        if hash(dict1_keys_list[i]) != hash(dict2_keys_list[i]):
            # hash值不一致时，判断不同key的类型是否一致，如果类型不一致，停止比较
            #print("第"+str(i)+"个key不同")
            return False

    # key相同时，比较value大小
    for key in dict1.keys():

        # 判断不同value的类型是否一致，如果类型不一致，停止比较
        if dict1[key].shape != dict2[key].shape:  # 判断value的类型是否一致
            #print(key+" tensor尺寸不同")
            return False
    # print('Size Matched!=============================')
    return True


def getModList(net):
    ModList = []
    for cell in net.stage1:
        for mod in cell.op:
            ModList.append(mod.state_dict())
    for cell in net.stage2:
        for mod in cell.op:
            ModList.append(mod.state_dict())
    for cell in net.stage3:
        for mod in cell.op:
            ModList.append(mod.state_dict())
    return ModList


def WeightSharing(oldcode,oldList,newcode,N):
    # oldList: old model's state dict
    # lo: old cell length
    # newnet: new code
    # ln: new cell length
    # st: Stacking Times

    newnet=cfnet(3,newcode,2)
    newList=getModList(newnet)
    #newcount=len(newcode[0])
    #newlen=len(newList)
    #oldlen=len(oldList)
    #oldcount=oldlen
    n=len(newcode)
    o=len(oldcode)
    flag=0
    count=0
    isloaded=[0]*n
    for k in range(N):
        for i in range(len(newcode)):
            for j in range(len(oldcode)):
                if isloaded[i]==0:
                    if isSameSize(newList[n*k+i],oldList[o*k+j]):
                        newnet.stage1[k].op[i].load_state_dict(oldList[o*k+j])
                        #print('1: node',k,i,j,' weights collected!')
                        flag=1
                        count+=1
                    if isSameSize(newList[N*n+k*n+i],oldList[N*o+k*o+j]):
                        newnet.stage2[k].op[i].load_state_dict(oldList[N*o+k*o+j])
                        #print('2: node',k,i,j,' weights collected!')
                        flag=1
                        count += 1
                    if isSameSize(newList[2*N*n+k*n+i],oldList[2*N*o+k*o+j]):
                        newnet.stage3[k].op[i].load_state_dict(oldList[2*N*o+k*o+j])
                        #print('3: node',k,i,j,' weights collected!')
                        flag=1
                        count += 1
                    if flag==1:
                        isloaded[i]=1
                        flag=0
            isloaded[i]=0
    if count>0:
        print('Prefer node weights collected!')
    return newnet


if __name__ =='__main__':
    print('Model Params SAVING and LOADING!')
    g1=[[[0, 0, 0], [1, 2]], [[2, 3, 0], [2]], [[2, 3, 0], []]]
    g2=[[[0, 0, 0], [1, 2]], [[2, 3, 0], [2]], [[5, 5, 0], []]]

    net1=cfnet(3,g1,2)
    ol=getModList(net1)
    net2=WeightSharing(g1,ol,g2,2)
    torch.save(net2,'/h   /qiuzz/PycharmProjects/pythonProject/cnn/test.pth')
    '''
    # print(paramList)
    for paramstate in modList:
        for param_tensor in paramstate:
            print(param_tensor,'\t',paramstate[param_tensor].size())
    '''


