import argparse, os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import architecture
from data import DIV2K, Set5_val
import utils
import skimage.color as sc
import random
from collections import OrderedDict
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# Training settings
parser = argparse.ArgumentParser(description="IMDN")
parser.add_argument("--batch_size", type=int, default=16,
                    help="training batch size")
parser.add_argument("--testBatchSize", type=int, default=1,
                    help="testing batch size")
parser.add_argument("-nEpochs", type=int, default=1000,
                    help="number of epochs to train")
parser.add_argument("--lr", type=float, default=2e-4,
                    help="Learning Rate. Default=2e-4")
parser.add_argument("--step_size", type=int, default=200,
                    help="learning rate decay per N epochs")
parser.add_argument("--gamma", type=int, default=0.5,
                    help="learning rate decay factor for step decay")
parser.add_argument("--cuda", action="store_true", default=True,
                    help="use cuda")
parser.add_argument("--resume", default="", type=str,
                    help="path to checkpoint")
parser.add_argument("--start-epoch", default=1, type=int,
                    help="manual epoch number")
parser.add_argument("--threads", type=int, default=8,
                    help="number of threads for data loading")
parser.add_argument("--root", type=str, default="training_data/",
                    help='dataset directory')
parser.add_argument("--n_train", type=int, default=800,
                    help="number of training set")
parser.add_argument("--n_val", type=int, default=1,
                    help="number of validation set")
parser.add_argument("--test_every", type=int, default=1000)
parser.add_argument("--scale", type=int, default=2,
                    help="super-resolution scale")
parser.add_argument("--patch_size", type=int, default=192,
                    help="output patch size")
parser.add_argument("--rgb_range", type=int, default=1,
                    help="maxium value of RGB")
parser.add_argument("--n_colors", type=int, default=3,
                    help="number of color channels to use")
parser.add_argument("--pretrained", default="", type=str,
                    help="path to pretrained models")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--isY", action="store_true", default=True)
parser.add_argument("--ext", type=str, default='.npy')
parser.add_argument("--phase", type=str, default='train')

args = parser.parse_args()
print(args)
torch.backends.cudnn.benchmark = True
# random seed
seed = args.seed
if seed is None:
    seed = random.randint(1, 10000)
print("Ramdom Seed: ", seed)
random.seed(seed)
torch.manual_seed(seed)

cuda = args.cuda
device = torch.device('cuda' if cuda else 'cpu')

print("===> Loading datasets")

trainset = DIV2K.div2k(args)
testset = Set5_val.DatasetFromFolderVal("Test_Datasets/Set5/",
                                       "Test_Datasets/Set5_LR/x{}/".format(args.scale),
                                       args.scale)
training_data_loader = DataLoader(dataset=trainset, num_workers=args.threads, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True)
testing_data_loader = DataLoader(dataset=testset, num_workers=args.threads, batch_size=args.testBatchSize,
                                 shuffle=False)

print("===> Building models")
args.is_train = True

model = architecture.IMDN(upscale=args.scale)
l1_criterion = nn.L1Loss()

print("===> Setting GPU")
if cuda:
    model = model.to(device)
    l1_criterion = l1_criterion.to(device)

#加载预训练模型
if args.pretrained:                                                     #1.判断是否提供了 pretrained 参数

    if os.path.isfile(args.pretrained):                                 #2.检查 args.pretrained 是否是一个有效的文件路径
        print("===> loading models '{}'".format(args.pretrained))       #如果文件存在，则打印 eg.:loading models 'checkpoints/IMDN_x2.pth'
        checkpoint = torch.load(args.pretrained)                        #3.加载 PyTorch训练好的模型文件（.pth），返回一个模型的权重字典OrderedDict/state_dict，checkpoint.items() 代表所有的层名称和对应的权重值。
        
       #4.处理 state_dict权重字典，去掉 module. 前缀（如果存在）
        new_state_dcit = OrderedDict()                                    #4.1 创建一个有序字典（用于存储处理后的权重）
        for k, v in checkpoint.items():                                   #4.2 遍历权重文件中的所有键值对 ；.pth 文件中的所有参数名称 (k) 和对应的权重 (v)
            if 'module' in k:                                             #4.3 参数名中是否有modeL
                name = k[7:]                                              #4.4 有，去掉，"module." 长度是 7
            else:
                name = k
            new_state_dcit[name] = v                                      #4.5 存入去掉 module. 前缀后的键值对--经过这一步后，new_state_dcit 里存储的是处理过的预训练模型的参数名称和权重
        #5.预训练模型和当前模型参数匹配
        model_dict = model.state_dict()                                                   #5.1 获取当前模型（正在训练的模型）的参数字典
        pretrained_dict = {k: v for k, v in new_state_dcit.items() if k in model_dict}    #5.2 遍历预训练模型字典，仅保留预训练模型在当前模型结构中存在的参数，过滤不匹配参数---new_state_dcit.items()：遍历预训练模型的参数字典

        for k, v in model_dict.items():                                                   #5.3 遍历当前模型的所有参数，检查已过滤后的预训练模型中是否缺少当前模型的某些参数
            if k not in pretrained_dict:                                                  #5.4 如果预训练中没有，打印
                print(k)
        model.load_state_dict(pretrained_dict, strict=True)                               #5.5 strict=True：严格匹配参数，如果 pretrained_dict 里少了 model_dict 里的任何参数，就会报错。strict=False：允许部分加载，如果有些参数匹配不上，就会跳过加载，但不会报错

    else:
        print("===> no models found at '{}'".format(args.pretrained))  #如不存在，则打印。。。，不会加载模型，直接跳过

print("===> Setting Optimizer")

optimizer = optim.Adam(model.parameters(), lr=args.lr)


def train(epoch):
    model.train()
    utils.adjust_learning_rate(optimizer, epoch, args.step_size, args.lr, args.gamma)
    print('epoch =', epoch, 'lr = ', optimizer.param_groups[0]['lr'])
    for iteration, (lr_tensor, hr_tensor) in enumerate(training_data_loader, 1):

        if args.cuda:
            lr_tensor = lr_tensor.to(device)  # ranges from [0, 1]
            hr_tensor = hr_tensor.to(device)  # ranges from [0, 1]

        optimizer.zero_grad()
        sr_tensor = model(lr_tensor)
        loss_l1 = l1_criterion(sr_tensor, hr_tensor)
        loss_sr = loss_l1

        loss_sr.backward()
        optimizer.step()
        if iteration % 100 == 0:
            print("===> Epoch[{}]({}/{}): Loss_l1: {:.5f}".format(epoch, iteration, len(training_data_loader),
                                                                  loss_l1.item()))

#验证（validation）一个超分辨率SR模型的性能，计算 PSNR（峰值信噪比）和 SSIM（结构相似性）两个指标
def valid():
    model.eval()    
    #遍历测试数据集
    avg_psnr, avg_ssim = 0, 0
    for batch in testing_data_loader:
        lr_tensor, hr_tensor = batch[0], batch[1]
        if args.cuda:
            lr_tensor = lr_tensor.to(device)
            hr_tensor = hr_tensor.to(device)
        #禁用梯度运算
        with torch.no_grad():
            pre = model(lr_tensor)
        # 处理图像数据
        sr_img = utils.tensor2np(pre.detach()[0])           #从 pre 中分离计算图，并取出第一个样本（批量第一个）
        gt_img = utils.tensor2np(hr_tensor.detach()[0])
        #裁剪边缘
        crop_size = args.scale
        cropped_sr_img = utils.shave(sr_img, crop_size)     #utils.shave()：去除图像边缘 crop_size 像素，可能是为了消除边缘效应（如零填充或上采样导致的失真）。
        cropped_gt_img = utils.shave(gt_img, crop_size)
        #选择颜色通道
        if args.isY is True:                                                        # args.isY 表示是否只使用亮度通道（Y 通道），
            im_label = utils.quantize(sc.rgb2ycbcr(cropped_gt_img)[:, :, 0])        # 转换为 YCbCr 颜色空间，取 Y（亮度）通道进行评估
            im_pre = utils.quantize(sc.rgb2ycbcr(cropped_sr_img)[:, :, 0])
        else:
            im_label = cropped_gt_img                                               #若 args.isY 为 False，直接使用完整的 RGB 计算指标
            im_pre = cropped_sr_img
        avg_psnr += utils.compute_psnr(im_pre, im_label)                            #计算PSNR  
        avg_ssim += utils.compute_ssim(im_pre, im_label)                            #计算SSIM
    print("===> Valid. psnr: {:.4f}, ssim: {:.4f}".format(avg_psnr / len(testing_data_loader), avg_ssim / len(testing_data_loader)))

# 保存训练参数，作为下一次的预训练
def save_checkpoint(epoch):
    model_folder = "checkpoint_x{}/".format(args.scale)
    model_out_path = model_folder + "epoch_{}.pth".format(epoch)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    torch.save(model.state_dict(), model_out_path)
    print("===> Checkpoint saved to {}".format(model_out_path))

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


print("===> Training")
print_network(model)
for epoch in range(args.start_epoch, args.nEpochs + 1):
    valid()                                                  #调用模型性能计算模块--得到PSNR和SSIM值
    train(epoch)                                             #调用训练模块
    save_checkpoint(epoch)                                   #保存模型的参数，下一次train时的作为预训练模型参数
