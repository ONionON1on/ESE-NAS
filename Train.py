from torch import nn
import torch
import numpy as np
from decode import *
from nodes import *
from ParamSaveLoad import *
from os import path
# from nas import *
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
import torchvision.transforms as transforms
#from torchsummary import summary
import numpy as np
import time
from matplotlib import pyplot as plt
from bisect import bisect_right
import joblib
#from Predict_data_process import *
import sys
import os

def str_to_float(str):
    return str.split()[0]

class HiddenPrints:
    def __init__(self, activated=True):
        # activated参数表示当前修饰类是否被激活
        self.activated = activated
        self.original_stdout = None

    def open(self):
        sys.stdout.close()
        sys.stdout = self.original_stdout

    def close(self):
        self.original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        # 这里的os.devnull实际上就是Linux系统中的“/dev/null”
        # /dev/null会使得发送到此目标的所有数据无效化，就像“被删除”一样
        # 这里使用/dev/null对sys.stdout输出流进行重定向

    def __enter__(self):
        if self.activated:
            self.close()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.activated:
            self.open()


# HYPERPARAMS
# global BATCH_SIZE,EPOCH
SEARCH_ROUNDS=100
BATCH_SIZE=4
EPOCH=15
train_batch_size=100
test_batch_size=32


# DATASETS INFORMATION
global IMAGE_SIZE,CHANNELS
IMAGE_SIZE=32
CHANNELS=3

# TRAINING DETAILS

# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# TODO: DATA AUGMENTATION
def data_prepare():
    # BATCH_SIZE=4
    # EPOCH=2
    # ===========================================================================================
    # 准备数据
    # Compose的意思是将多个transform组合在一起用，ToTensor 将像素转化为[0,1]的数字，Normalize则正则化变为 [-1,1]
    #tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    transform_train = transforms.Compose([
          # 对原始32*32图像四周各填充4个0像素（40*40），然后随机裁剪成32*32
                 transforms.RandomCrop(32, padding=4),

          # 按0.5的概率水平翻转图片
                 transforms.RandomHorizontalFlip(),

                 transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])


    transform_test = transforms.Compose([
             transforms.ToTensor(),
             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    # 下载数据集，训练集：需要训练；测试集：不需要训练
    train_set = torchvision.datasets.CIFAR10(root='./cifar10', train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR10(root='./cifar10', train=False, download=True, transform=transform_test)
    # 指定十个类别的标签，有的数据集很大的回加载相应的标签文件(groundtruth)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'truck', 'ship')

    # Training 共50000,取前20000作为训练集
    n_training_sample = 50000
    train_sample = SubsetRandomSampler(np.arange(n_training_sample, dtype=np.int64))
    # Validation 取训练集中的[20000,20000+5000]作为验证集
    # n_validation_sample = 5000
    # validation_sample = SubsetRandomSampler(np.arange(n_training_sample, n_training_sample + n_validation_sample,dtype=np.int64))
    # Testing 共10000,取前5000作为测试集
    n_test_sample = 10000
    test_sample = SubsetRandomSampler(np.arange(n_test_sample, dtype=np.int64))

    # 开启shuffle就等于全集使用SubsetRandomSampler，都是随机采样,num_workers代表多线程加载数据,Windows上不能用(必须0),Linux可用
    train_batch_size = 100
    test_batch_size = 100
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=train_batch_size, sampler=train_sample,
                                               num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_batch_size, sampler=test_sample, num_workers=0)
    return train_loader, test_loader


# TODO: Redefining a warmup scheduler

class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
            self,
            optimizer,
            milestones,
            gamma=0.1,
            warmup_factor=1 / 3,
            warmup_iters=100,
            warmup_method="linear",
            last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]

from ptflops import get_model_complexity_info



def eval_via_predictor(model,train_loader,test_loader):
    predictor = joblib.load('Regression.model')
    # TODO: Deploy the TRAINING PROCESS on the GPU
    #DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    DEVICE=torch.device("cpu")
    print("Training start")
    # TODO: Decode the g-code into a Neural Architecture
    # model = cfnet(3, g,3)
    model.to(DEVICE)
    #model = nn.DataParallel(model, device_ids=[0, 1])
    AS = []
    SP = [0.0, 0.0]
    with HiddenPrints():
        flops_str, params_str = get_model_complexity_info(model, (3, 32, 32), as_strings=True,
                                                          print_per_layer_stat=False)  # 不用写batch_size大小，默认batch_size=1
    SP[0] = str_to_float(flops_str)
    SP[1] = str_to_float(params_str)

    # TODO: Setup the TRAINING CONFIGURATION
    loss_func = torch.nn.CrossEntropyLoss()  # 交叉熵损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.9)
    steps = list(range(5, 150, 5))
    scheduler = WarmupMultiStepLR(optimizer=optimizer,
                                  milestones=steps,
                                  gamma=0.8,
                                  warmup_factor=0.1,
                                  warmup_iters=4,
                                  warmup_method="linear",
                                  last_epoch=-1)

    # TODO: Setup the best-MARK.
    best_loss = 1000000
    best_acc = 0
    counter = 0
    patiency = 5
    losslist = []
    time_per_epoch = 0
    modList = None
    # TODO: Training PROCESS
    for epoch in range(9):
        # print("Epoch:", epoch)
        training_start_time = time.time()  # 开始时间，为了后边统计一个训练花费时间
        start_time = time.time()
        train_loss = 0
        model.train()
        for step, (x_batch, y_batch) in enumerate(train_loader):
            # x_batch = x_batch.cuda()
            # y_batch = y_batch.cuda()
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            outputs = model(x_batch)
            loss = loss_func(outputs, y_batch)
            train_loss += loss.item()
           
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            time_consuming = time.time() - training_start_time
        if epoch==9:
            print("Training loss={}, took {:.2f}s".format(train_loss / (len(train_loader)),
                                                      time.time() - training_start_time))  # 所有的Epoch结束，也就是训练结束，计算花费的时间
        scheduler.step()
        time_per_epoch+=time_consuming
        # TODO: EVALUATE on the TEST SET
        correct = 0
        test_loss = 0
        model.eval()
        with torch.no_grad():
            for data in test_loader:
                # Forward pass
                x_batch, y_batch = data
                
                x_batch = x_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                out = model(x_batch)
                loss = loss_func(out, y_batch)
                predicted = torch.max(out, 1)[1]
                correct += (predicted == y_batch).sum().item()
                test_loss += loss.item()
            current_loss = test_loss / len(test_loader)
            current_acc = correct / len(test_loader) / 100
            if epoch==9:
                print("test loss = {:.2f}, Accuracy={:.6f}".format(test_loss / len(test_loader), correct / len(
                    test_loader) / 100))  # 求验证集的平均损失是多少
            losslist.append(current_loss)
            if epoch>3:
                AS.append(current_acc)
        # TODO: EARLY STOPPING
        '''
        if best_acc < current_acc:
            if best_loss >= current_loss:
                best_loss = current_loss
            best_acc = current_acc
            torch.save(model, '/data/qiuzz/projectResearch/cnn/best_model_so_far.pth')
            modList = getModList(model)
            counter = 0
        else:
            counter += 1
            if counter > patiency:
                return best_acc, modList
        '''
    TC=time_per_epoch/5
    pred_input=[[]]
    for item in AS:
        pred_input[0].append(item)
    pred_input[0].append(SP[0])
    pred_input[0].append(SP[1])
    pred_input[0].append(TC)

    perf_pred=predictor.predict(pred_input)
    # TODO: RETURN The RESULTS
    return perf_pred[0]

def eval(model ,train_loader, test_loader) -> object:
    # TODO: Deploy the TRAINING PROCESS on the GPU
    DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # TODO: Decode the g-code into a Neural Architecture
    #model = cfnet(3, g,3)
    model.to(DEVICE)
    # model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    # TODO: Setup the TRAINING CONFIGURATION
    loss_func = torch.nn.CrossEntropyLoss()  # 交叉熵损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.9)
    steps=list(range(5,300,5))
    scheduler = WarmupMultiStepLR(optimizer=optimizer,
                                  milestones=steps,
                                  gamma=0.8,
                                  warmup_factor=0.1,
                                  warmup_iters=4,
                                  warmup_method="linear",
                                  last_epoch=-1)

    # TODO: Setup the best-MARK.
    best_loss = 1000000
    best_acc = 0
    counter = 0
    patiency = 50
    losslist = []
    modList=None
    # TODO: Training PROCESS
    for epoch in range(300):
        print("Epoch:",epoch)
        training_start_time = time.time()  # 开始时间，为了后边统计一个训练花费时间
        start_time = time.time()
        train_loss = 0
        model.train()
        for step, (x_batch, y_batch) in enumerate(train_loader):
            #x_batch = x_batch.cuda()
            #y_batch = y_batch.cuda()
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            outputs = model(x_batch)
            loss = loss_func(outputs, y_batch)
            train_loss += loss.item()
       
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("Training loss={}, took {:.2f}s".format(train_loss / (len(train_loader)),
                                                      time.time() - training_start_time))  # 所有的Epoch结束，也就是训练结束，计算花费的时间
        scheduler.step()
        # TODO: EVALUATE on the TEST SET
        correct = 0
        test_loss = 0
        model.eval()
        with torch.no_grad():
            for data in test_loader:
                # Forward pass
                x_batch, y_batch = data
            
                x_batch = x_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                out = model(x_batch)
                loss = loss_func(out, y_batch)
                predicted = torch.max(out, 1)[1]
                correct += (predicted == y_batch).sum().item()
                test_loss += loss.item()
            current_loss = test_loss / len(test_loader)
            current_acc = correct / len(test_loader) / 100
            print("test loss = {:.2f}, Accuracy={:.6f}".format(test_loss / len(test_loader), correct / len(
                test_loader) / 100))  # 求验证集的平均损失是多少
            losslist.append(current_loss)

        # TODO: EARLY STOPPING
        if best_acc<current_acc:
            if best_loss>=current_loss:
                best_loss = current_loss
            best_acc = current_acc
            torch.save(model, '../results/best_model_so_far.pth')
            # modList = getModList(model)
            counter = 0
        else:
            counter += 1
            if counter > patiency:
                return best_acc # , modList

    # TODO: RETURN The RESULTS
    return best_acc# , modList

