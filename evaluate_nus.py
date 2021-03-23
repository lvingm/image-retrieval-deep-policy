import argparse
from utils_nus import *
from model import *
from PIL import Image

current_dir = os.getcwd()


def compute_topK_precision(db_binary, db_labels, tst_binary, tst_labels):
    for x in db_binary, tst_binary, db_labels, tst_labels: x.long()

    precision_k = []

    for i in range(10):
        R = (i + 1) * 100  # top k returned
        precision = []

        for i in range(tst_binary.size(0)):
            query_label, query_binary = tst_labels[i], tst_binary[i]
            query_label[query_label == 0] = -1
            _, query_result = torch.sum((query_binary != db_binary).long(), dim=1).sort()
            idx = query_result[0:R]
            correct = (torch.sum(query_label == db_labels[idx, :], dim=1) > 0).type(torch.FloatTensor)

            relevant_num = correct.sum().type(torch.FloatTensor)
            precision.append(relevant_num / R)

        precision_k.append(torch.Tensor(precision).mean())
    return precision_k


def hamming_precision(db_binary, db_labels, tst_binary, tst_labels, radius):
    for x in db_binary, tst_binary, db_labels, tst_labels: x.long()

    precision = []

    for i in range(tst_binary.size(0)):
        query_label, query_binary = tst_labels[i], tst_binary[i]
        query_label[query_label == 0] = -1

        distance = torch.abs(query_binary - db_binary).sum(1)
        similarity = (torch.sum(query_label == db_labels, dim=1) > 0).type(torch.FloatTensor)

        total_rel_num = torch.sum(distance <= radius).type(torch.FloatTensor)
        true_positive = torch.sum((distance <= radius).type(torch.FloatTensor) * similarity)

        if total_rel_num != 0:
            precision.append(true_positive / total_rel_num)
        else:
            precision.append(0.0)

    return torch.Tensor(precision).mean()


# def precision_recall(db_binary, db_labels, tst_binary, tst_labels):
#     for x in db_binary, tst_binary, db_labels, tst_labels: x.long()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train AlexNet')

    parser.add_argument('--data_dir', type=str, default=current_dir + '/data/', help='the directory to save data set')
    parser.add_argument('--ckpt_dir', type=str, default=current_dir + '/result_nus/',
                        help='the directory to save pretrained model parameters')
    parser.add_argument('--batch_size', type=int, default=50, help='input batch size')
    parser.add_argument('--ngpu', type=int, default=3, help='which GPU to use')
    parser.add_argument('--model_name', type=str, default='vgg19', help='model name')
    parser.add_argument('--binary_bits', type=int, default=48, help='bits of binary codes')
    parser.add_argument('--test_times', type=int, default=10, help='times for test')
    parser.add_argument('--margin', type=int, default=4, help='loss_type')
    parser.add_argument('--factor', type=int, default=2, help='return factor')

    opt = parser.parse_args()
    print(opt)

    os.makedirs(opt.ckpt_dir, exist_ok=True)
    choose_gpu(opt.ngpu)
    feed_random_seed()

    # # data set
    DB_loader, train_loader, test_loader = init_cifar_dataloader(opt.batch_size)
    pth_file = os.path.join(opt.ckpt_dir, f't_policy_{opt.binary_bits}bits_m{opt.margin}r2p4.pth')
    # pth_file = os.path.join(opt.ckpt_dir, f'vgg19_{opt.binary_bits}bits_m{opt.margin}.pth')
    print('pth_file:', pth_file)

    net = torch.load(pth_file)
    print('finish loaded parameters.')

    net.cuda()
    net.eval()
    #
    # for i in range(10):
    #     db_binary, db_label = compute_result(DB_loader, net, opt)
    #     tst_binary, tst_label = compute_result(test_loader, net, opt)
    #
    #     # # calculate mAP for test_times, and calculate mean value
    #     # accum_map = 0.0
    #     # for i in range(opt.test_times):
    #     #     AP = compute_AP(db_binary, tst_binary, db_label, tst_label)
    #     #     mAP = torch.Tensor(AP).mean()
    #     #     time_now = getNowTime()
    #     #     print(f'{time_now} mAP: {mAP:.4f}')
    #     #     accum_map += mAP
    #     #
    #     # total_mAP = accum_map / opt.test_times
    #     # print(f'{time_now} bits:{opt.binary_bits} mAP: {total_mAP:.4f}')
    #
    #     # calculate precision within hamming radius 2
    #     print('start to calculate precision within hamming radius 2...')
    #     ham2_precision = hamming_precision(db_binary, db_label, tst_binary, tst_label, 2)
    #     print(f'precision within hamming radius 2 is: {ham2_precision:.3f}')
    #
    #     # calculate top k precision
    #     if opt.binary_bits == 48:
    #         print('start to calculate top k precision')
    #         topk_precision = compute_topK_precision(db_binary, db_label, tst_binary, tst_label)
    #         print(f'top 100 to top 1000 precision: ')
    #         print(topk_precision)
    db_binary, db_label = compute_result(DB_loader, net, opt)
    tst_binary, tst_label = compute_result(test_loader, net, opt)
    AP = compute_AP(db_binary, tst_binary, db_label, tst_label)
    mAP = torch.Tensor(AP).mean()
    print('map:', mAP)

    for x in db_binary, tst_binary, db_label, tst_label: x.long()
    top_num = 10
    for i in range(5):
        q_idx = i * opt.factor
        query_label, query_binary = tst_label[q_idx], tst_binary[q_idx]
        query_label[query_label == 0] = -1
        _, query_result = torch.sum((query_binary != db_binary).long(), dim=1).sort()
        idxi = query_result[0:top_num]

        rlabel = db_label[idxi]
        print('the query index:', q_idx)
        print('the query label:', query_label)
        print('the top 10 returned index:', idxi)
        print('the top 10 returned label:', rlabel)


        # qimg = Image.fromarray(tst_imgs[q_idx].numpy())
        # qimg.save(path + 'qimg' + str(i) + '.jpg')
        # for j in range(10):
        #     rimg = Image.fromarray(db_imgs[idxi[j]].numpy())
        #     rimg.save(path + str(i) + 'rimg' + str(j) + '.jpg')




