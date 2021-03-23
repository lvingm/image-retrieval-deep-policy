import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import itertools
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import torchvision.transforms as transforms
import torch.utils.data as torchdata
import os
from PIL import Image


def get_triplets(labels):
    labels = labels.cpu().data.numpy()
    triplets = []
    for i in range(labels.shape[0]):
        label = labels[i]
        label_mask = np.matmul(labels, np.transpose(label)) > 0
        label_indices = np.where(label_mask)[0]
        if len(label_indices) < 2:
            continue
        negative_indices = np.where(np.matmul(labels, np.transpose(label)) == 0)[0]
        anchor_positives = list(combinations(label_indices, 2))  # All anchor-positive pairs

        # Add all negatives for all positive pairs
        temp_triplets = [[anchor_positive[0], anchor_positive[1], neg_ind] for anchor_positive in anchor_positives
                         for neg_ind in negative_indices]
        triplets += temp_triplets

    return torch.LongTensor(np.array(triplets))


def toBinaryString(binary_like_values):
    numOfImage, bit_length = binary_like_values.shape
    list_string_binary = []
    for i in range(numOfImage):
        str = ''
        for j in range(bit_length):
            str += '0' if binary_like_values[i][j] <= 0.5 else '1'
        list_string_binary.append(str)
    return list_string_binary


def resize_img(img, opt):
    if opt.model_name == 'alexnet':
        img = F.upsample(img, size=(227, 227), mode='bilinear')
    elif opt.model_name == 'CNNH':
        img = F.upsample(img, size=(64, 64), mode='bilinear')
    else:
        img = F.upsample(img, size=(224, 224), mode='bilinear')
    return img


def combinations(iterable, r):
    pool = list(iterable)
    n = len(pool)
    for indices in itertools.permutations(range(n), r):
        if sorted(indices) == list(indices):
            yield list(pool[i] for i in indices)


def choose_gpu(i_gpu):
    """choose current CUDA device"""
    torch.cuda.device(i_gpu).__enter__()
    cudnn.benchmark = True


def adjust_learning_rate(optimizer, epoch, opt):
    lr_ = opt.lr * (0.1 ** (epoch // opt.lr_decay_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_
        if epoch % opt.lr_decay_epoch == 0:
            print('setting lr to %.2E' % lr_)


def getNowTime():
    '''print current local time'''
    return '[' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + ']'


def feed_random_seed(seed=np.random.randint(1, 10000)):
    """feed random seed"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def compute_result(dataloader, net, opt):
    '''
       generate binary codes.
       :param dataloader: the dataloader including images and labels
       :param net: the network
       :param opt: parser arguments
       :return: binary codes and labels
    '''
    net.eval()
    binary_code = torch.cuda.FloatTensor()
    labels = torch.cuda.LongTensor()

    for batch_idx, (img, cls) in enumerate(dataloader):
        img, cls = [Variable(x.cuda()) for x in (img, cls)]
        img = resize_img(img, opt)

        output = net(img)
        sigmoid = nn.Sigmoid()
        output = sigmoid(output)
        binary_code = torch.cat((binary_code, output.data), 0)
        labels = torch.cat((labels, cls.data), 0)

    return torch.round(binary_code), labels


def compute_AP(trn_binary, tst_binary, trn_label, tst_label):# nuswide
    """
    compute mAP by searching testset from trainset
    https://github.com/flyingpot/pytorch_deephash
    """
    for x in trn_binary, tst_binary, trn_label, tst_label: x.long()

    AP = []

    # # nus_wide
    top_num = 5000

    # # mirFlickr
    # top_num = trn_binary.size(0)

    Ns = torch.arange(1, top_num + 1)  # top5000
    Ns = Ns.type(torch.FloatTensor)

    for i in range(tst_binary.size(0)):
        query_label, query_binary = tst_label[i], tst_binary[i]
        query_label[query_label == 0] = -1
        _, query_result = torch.sum((query_binary != trn_binary).long(), dim=1).sort()
        idx = query_result[0:top_num]
        correct = (torch.sum(query_label == trn_label[idx, :], dim=1) > 0).type(torch.FloatTensor)
        P = (torch.cumsum(correct, dim=0) / Ns).type(torch.FloatTensor)

        AP.append(torch.sum(P * correct) / torch.sum(correct))

    return AP


def default_loader(path):
    return Image.open(path).convert('RGB')


class MyDataset(torchdata.Dataset):
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            if line == '':
                break
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split('.jpg ')
            imgs.append(words)
        self.imgs = imgs

    def __getitem__(self, index):
        words = self.imgs[index]
        #
        # nus_wide
        img = self.loader('/home/disk2/zenghaien/nus-wide/images_256' + words[0] + '.jpg')  # 1,2,3

        # # mirFlickr
        # img = self.loader('/home/disk2/zenghaien/mirflickr/mirflickr_256/' + words[0] + '.jpg') # 1,2,3

        if self.transform is not None:
            img = self.transform(img)
        label = ''.join(words[1].split())
        label = torch.tensor(list(map(int, label)))
        return img, label

    def __len__(self):
        return len(self.imgs)


def init_cifar_dataloader(batchSize):
    root_dir = os.getcwd() + '/nuswide_txt/'
    # root_dir = os.getcwd() + '/mirFlickr_txt/'

    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    transform_test = transforms.Compose([
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    # # nus_wide
    test_dir = root_dir + 'test_2100.txt'
    train_dir = root_dir + 'train_10500.txt'
    DB_dir = root_dir + 'DB.txt'

    # mirFlickr
    # test_dir = root_dir + 'test_1000.txt'
    # train_dir = root_dir + 'train_5000.txt'
    # DB_dir = root_dir + 'DB.txt'

    testset = MyDataset(txt=test_dir, transform=transform_test)
    test_loader = torchdata.DataLoader(testset, batch_size=batchSize, shuffle=False,
                                       num_workers=4, pin_memory=True)

    train_set = MyDataset(txt=train_dir, transform=transform_train)
    train_loader = torchdata.DataLoader(train_set, batch_size=batchSize, shuffle=False,
                                        num_workers=4, pin_memory=True)

    DB_set = MyDataset(txt=DB_dir, transform=transform_train)
    DB_loader = torchdata.DataLoader(DB_set, batch_size=batchSize, shuffle=False,
                                     num_workers=4, pin_memory=True)

    return DB_loader, train_loader, test_loader