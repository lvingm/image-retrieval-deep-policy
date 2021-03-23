import torch.nn as nn
import torchvision.models as models
import torch.nn.init as init
import torch
import os

current_dir = os.getcwd()

def setup_net(opt, net = None):
    if opt.model_name == 'resnet18':
        print('the model used is resnet18.')
        net = models.resnet18(pretrained=True)
        fc_features = net.fc.in_features
        net.fc = nn.Linear(fc_features, opt.binary_bits)

    elif opt.model_name == 'CNNH':
        net = CNNH(opt.binary_bits)
        print('the model used is CNNH.')

    elif opt.model_name == 'alexnet':
        print('the model used is AlexNet.')
        alexnet = AlexNet()
        alexnet.load_state_dict(torch.load(current_dir + '/result_alex/alexnet.pth'), strict=True)   # pretrained model
        net = FixAlexNet(alexnet, opt.binary_bits)

    elif opt.model_name == 'vgg19_bn':
        print('the model used is vgg19_bn.')
        net = models.vgg19_bn(pretrained=False)
        net.load_state_dict(torch.load(current_dir + '/result_vgg19_bn/vgg19_bn.pth'), strict=True)
        in_features = net.classifier[-1].in_features
        net.classifier[-1] = nn.Linear(in_features=in_features, out_features=opt.binary_bits)

    elif opt.model_name == 'vgg19':
        print('the model used is vgg19.')
        net = models.vgg19(pretrained=False)
        net.load_state_dict(torch.load(current_dir + '/result_nus/vgg19.pth'), strict=True)
        in_features = net.classifier[-1].in_features
        net.classifier[-1] = nn.Linear(in_features=in_features, out_features=opt.binary_bits)

    return net


class CNNH(nn.Module):
    def __init__(self, num_binary):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),  # same padding
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.fc = nn.Sequential(nn.Linear(128 * 7 * 7, num_binary))

        for m in self.modules():
            if m.__class__ == nn.Conv2d or m.__class__ == nn.Linear:
                init.xavier_normal(m.weight.data)
                m.bias.data.fill_(0)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class DSH(nn.Module):
    def __init__(self, num_binary):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),  # same padding
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3, stride=2),

            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3, stride=2),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 3 * 3, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, num_binary)
        )

        for m in self.modules():
            if m.__class__ == nn.Conv2d or m.__class__ == nn.Linear:
                init.xavier_normal(m.weight.data)
                m.bias.data.fill_(0)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


class FixAlexNet(nn.Module):
    def __init__(self, model, num_binary):
        super(FixAlexNet,  self).__init__()
        #remove last two layer of model
        self.features = nn.Sequential(*list(model.features.children()))
        self.classifier = nn.Sequential(*list(model.classifier.children())[:-1])
        self.linear = nn.Linear(4096, num_binary)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        x = self.linear(x)

        return x



