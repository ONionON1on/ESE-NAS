# import cv2

import numpy as np
from PIL import Image
import torch
from torchvision import utils,transforms
import torch.nn as nn

import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from copy import deepcopy

def soft_pool2d(x, kernel_size=2, stride=None, force_inplace=False):
    #if x.is_cuda and not force_inplace:
        #return CUDA_SOFTPOOL2d.apply(x, kernel_size, stride)
    kernel_size = _pair(kernel_size)
    if stride is None:
        stride = kernel_size
    else:
        stride = _pair(stride)
    # Get input sizes
    _, c, h, w = x.size()
    # Create per-element exponential value sum : Tensor [b x 1 x h x w]
    e_x = torch.sum(torch.exp(x),dim=1,keepdim=True)
    # Apply mask to input and pool and calculate the exponential sum
    # Tensor: [b x c x h x w] -> [b x c x h' x w']
    return F.avg_pool2d(x.mul(e_x), kernel_size, stride=stride).mul_(sum(kernel_size)).div_(F.avg_pool2d(e_x, kernel_size, stride=stride).mul_(sum(kernel_size)))
'''
import softpool_cuda
from SoftPool import soft_pool1d, SoftPool1d
from SoftPool import soft_pool2d, SoftPool2d
from SoftPool import soft_pool3d, SoftPool3d

from einops import rearrange
'''

'''

#img = cv2.imread('paojie.jpg', 1)  # 读入灰度图像
img=Image.open('paojie.jpg')
# img = np.array(img, dtype='float32')
transform2 = transforms.ToTensor()
img = transform2(img)   # torch.Size([3, 313, 500])
img2=deepcopy(img)
# print(img.shape)
#img = torch.from_numpy(img)
#img=img.permute(2,0,1)
#img=transforms.ToPILImage()(img)
#img.save("yuantu.jpg")

img=img.unsqueeze(0)
img2=img2.unsqueeze(0)

maxPool = nn.MaxPool2d(2)
img=soft_pool2d(img,2,2)
img=soft_pool2d(img,2,2)
img=soft_pool2d(img,2,2)
img2=maxPool(img2)
img2=maxPool(img2)
img2=maxPool(img2)

img = torch.squeeze(img) # 去掉1的维度
img2 = torch.squeeze(img2)
img=transforms.ToPILImage()(img)
img2=transforms.ToPILImage()(img2)
img2.save("max.jpg")
img.save("soft.jpg")
'''




'''
img2=deepcopy(img)
#print(img2.shape)

#print(img==img2)
h=img.shape[0]
w=img.shape[1]

img = torch.from_numpy(img)  # 将灰度图像转换为tensor
img2 = torch.from_numpy(img2)
print(img.shape)
img=img.permute(2,0,1)
img2=img2.permute(2,0,1)
#img=img.reshape(1,3,313,500)
#img2=img2.reshape(1,3,313,500)
img=img.unsqueeze(0)
img2=img2.unsqueeze(0)

print(img.shape)
avgPool = nn.AvgPool2d(2)  # 4*4的窗口，步长为4的平均池化
maxPool = nn.MaxPool2d(2)
img2=maxPool(img2)
#img2=avgPool(img2)
# img = avgPool(img)
img=soft_pool2d(img,2,2)
print(img2.shape)
# img = img.numpy()
# img2 = img2.numpy()

# print(img==img2)

# img = img.numpy()
# img2 = img2.numpy()

# print(img==img2)

img = torch.squeeze(img).int() # 去掉1的维度
img2 = torch.squeeze(img2).int()

# img=rearrange(img,'b c h w -> b h w c')
# img2=rearrange(img2,'b c h w -> b h w c')

#img=img.permute(1,2,0)
#img2=img2.permute(1,2,0)
img=transforms.ToPILImage()(img)
img2=transforms.ToPILImage()(img2)
#img = torch.squeeze(img)  # 去掉1的维度
#img2 = torch.squeeze(img2)
img.save("out.jpg")
img2.save("out2.jpg")
'''
'''
img = img.numpy()# .astype('uint8')  # 转换格式，准备输出
img2 = img2.numpy()# .astype('uint8')

print(img==img2)

cv2.imwrite("out.jpg", img)
cv2.imwrite("out2.jpg", img2)
# cv2.imshow("result", img)

cv2.waitKey(0)

cv2.destroyAllWindows()
'''