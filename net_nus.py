import argparse
from model import *
from utils_nus import *

current_dir = os.getcwd()

def triplet_hashing_loss_regu(embeddings, cls, opt):
    triplets = get_triplets(cls)

    if embeddings.is_cuda:
        triplets = triplets.cuda()

    ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
    an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)  # .pow(.5)

    losses = F.relu(ap_distances - an_distances + opt.margin)

    return losses.mean()


def train(train_loader, agent, optimizer, opt):
    accum_tloss = 0.0
    agent.train()

    for batch_idx, (imgs,labels) in enumerate(train_loader):
        imgs, labels = [Variable(x.cuda()) for x in (imgs, labels)]
        imgs = resize_img(imgs, opt)

        embedding = agent(imgs)
        sigmoid = nn.Sigmoid()
        embedding = sigmoid(embedding)        # ∈[0, 1]

        triplet_loss = triplet_hashing_loss_regu(embedding, labels, opt)
        loss = triplet_loss

        agent.zero_grad()
        loss.backward()
        optimizer.step()

        accum_tloss += triplet_loss

    tloss = accum_tloss / len(train_loader)

    return tloss

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train vgg19')

    parser.add_argument('--dataset', type=str, default='cifar10', help='the name of dataset')
    parser.add_argument('--data_dir', type=str, default=current_dir + '/data/', help='the dirextory to save mini data set')
    parser.add_argument('--ckpt_dir', type=str, default=current_dir + '/result_mir/',
                        help='the directory to save pretrained model parameters')
    parser.add_argument('--load', type=bool, default=True, help='if load trained model before')
    parser.add_argument('--model_name', type=str, default='vgg19', help='choose which model to train')

    parser.add_argument('--batch_size', type=int, default=50, help='input batch size')
    parser.add_argument('--ngpu', type=int, default=0, help='which GPU to use')

    parser.add_argument('--binary_bits', type=int, default=12, help='length of hashing binary')
    parser.add_argument('--alpha', type=float, default=0.001, help='weighting of regularizer')

    parser.add_argument('--niter', type=int, default=200, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')

    parser.add_argument('--margin', type=int, default=1, help='loss_type')

    opt = parser.parse_args()
    print(opt)

    os.makedirs(opt.ckpt_dir, exist_ok=True)
    choose_gpu(opt.ngpu)
    feed_random_seed()

    DB_loader, train_loader, test_loader = init_cifar_dataloader(opt.batch_size)

    pth_file = os.path.join(opt.ckpt_dir, f'{opt.model_name}_{opt.binary_bits}bits_m{opt.margin}.pth')
    print(pth_file)
    if opt.load and os.path.exists(pth_file):
        agent = torch.load(pth_file)
        print('finish loaded parameters.')
    else:
        agent = setup_net(opt)
    agent.cuda()

    optimizer = torch.optim.SGD(agent.parameters(), lr=opt.lr, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    # start to train
    max_map = 0.0
    for epoch in range(0, opt.niter):
        tloss = train(train_loader, agent, optimizer, opt)

        time_now = getNowTime()
        print(f'{time_now} [{epoch}] tloss: {tloss:.4f}')
        scheduler.step()

        # calculate mAP and save trained parameters
        # if epoch % 10 == 0 and epoch >= 20:
        if (epoch+1) % 10 == 0:
            trn_binary, trn_label = compute_result(DB_loader, agent, opt)
            tst_binary, tst_label = compute_result(test_loader, agent, opt)

            time_now = getNowTime()
            AP = compute_AP(trn_binary, tst_binary, trn_label, tst_label)
            mAP = torch.Tensor(AP).mean()
            print(f'{time_now} [{epoch}] retrieval mAP: {mAP:.4f}')

            if max_map < mAP:
                max_map = mAP
                torch.save(agent, os.path.join(opt.ckpt_dir, f'{opt.model_name}_{opt.binary_bits}bits'
                                                             f'_m{opt.margin}.pth'))

    print(f'binary bits: {opt.binary_bits} max mAP: {max_map:.4f}')
    print(opt)





