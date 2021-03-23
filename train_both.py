import argparse
from torch.distributions import Bernoulli

from model import *
from utils import *

current_dir = os.getcwd()

def triplet_hashing_loss_regu(embeddings, cls, margin):
    triplets = get_triplets(cls)

    if embeddings.is_cuda:
        triplets = triplets.cuda()

    ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
    an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)  # .pow(.5)

    losses = F.relu(ap_distances - an_distances + margin)

    return losses.mean()


def get_reward(binary, labels, database_code, database_labels, opt):
    AP = compute_AP(database_code, binary, database_labels, labels)
    AP = np.array(AP)[:, np.newaxis]
    reward =np.zeros(AP.shape, dtype=float)

    # Reward Scheme 1: regard AP as reward directly, and every reward is positive
    if opt.reward_scheme == 1:
        reward = AP / 1.0

    # Reward Scheme 2: regard AP as reward directly when AP > 0.8.
    # Else, when AP <= 0.8, take AP - 1 as reward.
    # Then reward may be positive or negative.
    if opt.reward_scheme == 2:
        reward = AP / 1.0
        reward[reward <= 0.5] -= 1.0

    reward = torch.Tensor(reward)

    return reward, AP


def train(train_loader, agent, database_code, database_labels, optimizer, opt):
    accum_tloss = 0.0
    accum_ploss = 0.0
    accum_AP = 0.0
    agent.train()

    for batch_idx, (imgs,labels) in enumerate(train_loader):
        imgs, labels = [Variable(x.cuda()) for x in (imgs, labels)]
        imgs = resize_img(imgs, opt)

        probs = agent(imgs)
        sigmoid = nn.Sigmoid()
        probs = sigmoid(probs)        # ∈[0, 1]
        probs = probs * opt.bound_factor + (1 - probs) * (1 - opt.bound_factor)

        # generate binary code according to threshold
        binary_thd = probs.data.clone()
        binary_thd[binary_thd <= 0.5] = 0
        binary_thd[binary_thd > 0.5] = 1

        # generate binary codes according to Bernoulli distribution
        distr = Bernoulli(probs)
        binary_ber = distr.sample()

        reward_thd, _ = get_reward(binary_thd, labels, database_code, database_labels, opt)
        reward_ber, AP = get_reward(binary_ber, labels, database_code, database_labels, opt)

        advantage = reward_ber - reward_thd

        ad_loss = -distr.log_prob(binary_ber) * Variable(advantage.cuda().float()).expand_as(binary_ber)
        ad_loss = ad_loss.sum()

        probs = probs.clamp(1e-15, 1 - 1e-15)
        entropy_loss = -probs * torch.log(probs)
        entropy_loss = opt.beta * entropy_loss.sum()

        policy_loss = (ad_loss - entropy_loss) / imgs.size(0)
        triplet_loss = triplet_hashing_loss_regu(probs, labels, opt.margin)

        loss = policy_loss + triplet_loss

        agent.zero_grad()
        loss.backward()
        optimizer.step()

        accum_tloss += triplet_loss
        accum_ploss += policy_loss
        accum_AP += AP.mean()
    #
    mAP = accum_AP / len(train_loader)
    tloss = accum_tloss / len(train_loader)
    ploss = accum_ploss / len(train_loader)

    return tloss, ploss, mAP

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train resNet')

    parser.add_argument('--data_dir', type=str, default=current_dir + '/data/', help='the dirextory to save mini data set')
    parser.add_argument('--ckpt_dir', type=str, default=current_dir + '/logs_alex16pt0/',
                        help='the directory to save pretrained model parameters')
    parser.add_argument('--binary_file', type=str, default='binary16_alesp.txt', help='file to save binary code')
    parser.add_argument('--load', type=bool, default=True, help='if load trained model before')
    parser.add_argument('--model_name', type=str, default='alexnet', help='choose which model to train')

    parser.add_argument('--batch_size', type=int, default=50, help='input batch size')
    parser.add_argument('--ngpu', type=int, default=1, help='which GPU to use')

    parser.add_argument('--binary_bits', type=int, default=16, help='length of hashing binary')
    parser.add_argument('--alpha', type=float, default=0.001, help='weighting of regularizer')

    parser.add_argument('--niter', type=int, default=200, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')

    parser.add_argument('--margin', type=float, default=1, help='loss_type')
    parser.add_argument('--bound_factor', type=float, default=0.8, help='probability bounding factor')
    parser.add_argument('--reward_scheme', type=int, default=1, help='use which reward scheme to train')
    parser.add_argument('--beta', type=float, default=1e-1, help='entropy multiplier')

    opt = parser.parse_args()
    print(opt)

    os.makedirs(opt.ckpt_dir, exist_ok=True)
    choose_gpu(opt.ngpu)
    feed_random_seed()

    DB_loader, train_loader, test_loader = init_cifar_dataloader(opt.batch_size)

    if opt.load and os.listdir(opt.ckpt_dir):
        agent = torch.load(os.path.join(opt.ckpt_dir, f'pt_alex_{opt.binary_bits}bits.pth'))
        print('finish loaded parameters.')
    else:
        agent = setup_net(opt)
    agent.cuda()

    optimizer = torch.optim.SGD(agent.parameters(), lr=opt.lr, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)

    database_code, database_labels = compute_result(train_loader, agent, opt)

    max_map = 0.0
    for epoch in range(0, opt.niter):
        if epoch % 50 == 0 and epoch != 0:
            database_code, database_labels = compute_result(train_loader, agent, opt)
            print('finish updated database binary code.')
        # database_code, database_labels = compute_result(train_loader, agent, opt)

        tloss, ploss, mAP = train(train_loader, agent, database_code, database_labels, optimizer, opt)
        time_now = getNowTime()
        print(f'{time_now} [{epoch}] tloss: {tloss:.4f} ploss: {ploss:.4f} mAP: {mAP:.4F}')

        scheduler.step()

        # calculate mAP and save trained parameters
        if epoch % 10 == 0 and epoch >= 20:
            trn_binary, trn_label = compute_result(DB_loader, agent, opt)
            tst_binary, tst_label = compute_result(test_loader, agent, opt)

            time_now = getNowTime()
            AP = compute_AP(trn_binary, tst_binary, trn_label, tst_label)
            mAP = torch.Tensor(AP).mean()
            print(f'{time_now} [{epoch}] retrieval mAP: {mAP:.4f}')

            if max_map < mAP:
                max_map = mAP
                torch.save(agent, os.path.join(opt.ckpt_dir, f'pt_alex_{opt.binary_bits}bits.pth'))

    print(f'binary bits: {opt.binary_bits} max mAP: {max_map:.4f}')


