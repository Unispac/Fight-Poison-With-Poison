'''codes used to call key functional module of confusion training and use it to cleanse poisoned Ember dataset
'''
import os, sys
import argparse
import numpy as np
from utils import default_args

parser = argparse.ArgumentParser()
parser.add_argument('-ember_options', type=str, required=False,
                    choices=['constrained', 'unconstrained', 'none'],
                    default='unconstrained')
parser.add_argument('-devices', type=str, default='0')
parser.add_argument('-log', default=False, action='store_true')
parser.add_argument('-seed', type=int, required=False, default=default_args.seed)
parser.add_argument('-debug_info', default=False, action='store_true')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "%s" % args.devices

args.poison_type = 'ember_backdoor'

import torch
import torch.optim as optim
from torch import nn
from utils import supervisor, tools, ember_nn
import config
import confusion_training

# tools.setup_seed(args.seed)

if args.log:

    out_path = 'logs'
    if not os.path.exists(out_path): os.mkdir(out_path)
    out_path = os.path.join(out_path, '%s_seed=%s' % ('ember', args.seed))
    if not os.path.exists(out_path): os.mkdir(out_path)
    out_path = os.path.join(out_path, 'cleanse')
    if not os.path.exists(out_path): os.mkdir(out_path)
    out_path = os.path.join(out_path, 'CT_%s.out' % (supervisor.get_dir_core(args, include_poison_seed=config.record_poison_seed)))
    fout = open(out_path, 'w')
    ferr = open('/dev/null', 'a')
    sys.stdout = fout
    sys.stderr = ferr



num_classes = 2
weight_decay = 1e-6
batch_size =  64
kwargs = {'num_workers': 2, 'pin_memory': True}





### Dataset to be inspected
inspection_set_dir = os.path.join('poisoned_train_set', 'ember', args.ember_options)
x = np.load(os.path.join(inspection_set_dir, 'watermarked_X.npy'))
sts_mean = x.mean(axis=0)
sts_std = x.std(axis=0) + 0.1
sts = [sts_mean, sts_std]
inspection_set = tools.EMBER_Dataset_norm( x_path=os.path.join(inspection_set_dir, 'watermarked_X.npy'),
                                        y_path=os.path.join(inspection_set_dir, 'watermarked_y.npy'), sts=sts)
inspection_set_inverse = tools.EMBER_Dataset_norm( x_path=os.path.join(inspection_set_dir, 'watermarked_X.npy'),
                                        y_path=os.path.join(inspection_set_dir, 'watermarked_y.npy'), sts=sts, inverse=True)
final_budget = int(len(inspection_set)//25)

#print('final_budget:', final_budget)


### Small clean set for confusion training
clean_set_dir = os.path.join('clean_set', 'ember', 'clean_split')
clean_set = tools.EMBER_Dataset_norm(x_path=os.path.join(clean_set_dir, 'X.npy'),
                                   y_path=os.path.join(clean_set_dir, 'Y.npy'),
                                   sts=sts, inverse=True)

if args.debug_info:

    poison_indices_path = os.path.join(inspection_set_dir, 'poison_indices')
    poison_indices = torch.tensor(torch.load(poison_indices_path)).cuda()

    test_set_dir = os.path.join('clean_set', 'ember', 'test_split')
    test_set = tools.EMBER_Dataset_norm(x_path=os.path.join(test_set_dir, 'X.npy'),
                                   y_path=os.path.join(test_set_dir, 'Y.npy'),
                                   sts=sts)
    test_set_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size, shuffle=False, worker_init_fn=tools.worker_init, **kwargs)

    backdoor_test_set_dir = os.path.join('poisoned_train_set', 'ember', args.ember_options)
    backdoor_test_set = tools.EMBER_Dataset_norm(x_path=os.path.join(inspection_set_dir, 'watermarked_X_test.npy'),
                                            y_path=None, sts=sts)
    backdoor_test_set_loader = torch.utils.data.DataLoader(
        backdoor_test_set,
        batch_size=batch_size, shuffle=False, worker_init_fn=tools.worker_init, **kwargs)


    debug_packet = {
        'test_set_loader' : test_set_loader,
        'backdoor_test_set_loader' :  backdoor_test_set_loader
    }




def iterative_poison_distillation(inspection_set, inspection_set_inverse, clean_set, args, start_iter=0):

    arch = ember_nn.EmberNN_narrow
    distillation_ratio = [1/5, 1/20, 1/40, 1/80]
    momentums = [0.5, 0.5, 0.5, 0.5, 0.5]
    lrs = [0.1, 0.01, 0.01, 0.01, 0.01]
    lambs = [20, 20, 20, 20, 20]
    batch_factors = [1, 16, 64, 128, 128]


    params = {
        'kwargs': kwargs,
        'inspection_set_dir': inspection_set_dir,
        'num_classes': num_classes,
        'arch': config.arch['ember'],
        'distillation_ratio': distillation_ratio,
        'batch_size': batch_size,
        'median_sample_rate': 0.1
    }

    clean_set_loader = torch.utils.data.DataLoader(
        clean_set, batch_size=batch_size,
        shuffle=True, worker_init_fn=tools.worker_init, **kwargs)

    print('>>> Iterative Data Distillation with Confusion Training')

    distilled_samples_indices, median_sample_indices = None, None
    num_confusion_iter = len(distillation_ratio) + 1

    criterion = nn.BCELoss().cuda()
    criterion_no_reduction = nn.BCELoss(reduction='none').cuda()


    if start_iter != 0:
        distilled_samples_indices, median_sample_indices = confusion_training.distill(args, params, inspection_set,
                                   start_iter-1, criterion_no_reduction, dataset_name='ember', custom_arch=arch)
        distilled_set = torch.utils.data.Subset(inspection_set, distilled_samples_indices)

    else:
        distilled_set = inspection_set

    for confusion_iter in range(start_iter, num_confusion_iter):

        lr = lrs[confusion_iter]
        momentum = momentums[confusion_iter]

        size_of_distilled_set = len(distilled_set)
        print('<Round-%d> Size_of_distillation_set = ' % confusion_iter, size_of_distilled_set)

        # different weights for each class based on their frequencies in the distilled set

        nums_of_each_class = np.zeros(num_classes)
        for i in range(size_of_distilled_set):
            _, gt = distilled_set[i]
            gt = int(gt.item())
            nums_of_each_class[gt] += 1
        print(nums_of_each_class)
        freq_of_each_class = nums_of_each_class / size_of_distilled_set
        freq_of_each_class = freq_of_each_class + 0.1
        freq_of_each_class[:] = 1

        if confusion_iter < 1: # lr=0.01 for round 0,1,2
            pretrain_epochs = 10
            pretrain_lr = 0.1
            distillation_iters = 6000
        elif confusion_iter < 3:
            pretrain_epochs = 5
            pretrain_lr = 0.1
            distillation_iters = 4000
        elif confusion_iter < 4:
            pretrain_epochs = 5
            pretrain_lr = 0.1
            distillation_iters = 4000
        else:
            pretrain_epochs = 5
            pretrain_lr = 0.1 # lr=0.001 for round 3,4
            distillation_iters = 4000


        print('freq:', freq_of_each_class)

        if confusion_iter < 2:
            distilled_set_loader = torch.utils.data.DataLoader(
                distilled_set,
                batch_size=512, shuffle=True,
                worker_init_fn=tools.worker_init, **kwargs)
        else:
            distilled_set_loader = torch.utils.data.DataLoader(
                torch.utils.data.ConcatDataset([distilled_set, clean_set]),
                batch_size=512, shuffle=True,
                worker_init_fn=tools.worker_init, **kwargs)


        # pretrain base model
        confusion_training.pretrain(args, debug_packet, arch, num_classes, weight_decay, pretrain_epochs,
                                    distilled_set_loader, criterion, inspection_set_dir, confusion_iter, pretrain_lr,
                                    dataset_name='ember', load=False)


        distilled_set_loader = torch.utils.data.DataLoader(
            distilled_set,
            batch_size=batch_size, shuffle=True,
            worker_init_fn=tools.worker_init, **kwargs)

        # confusion_training
        model = confusion_training.confusion_train(args, params, inspection_set, debug_packet, distilled_set_loader, clean_set_loader, confusion_iter, arch,
                                   num_classes, inspection_set_dir, weight_decay, criterion_no_reduction,
                                   momentum, lambs[confusion_iter],
                                   freq_of_each_class, lr, batch_factors[confusion_iter], distillation_iters, dataset_name='ember')

        # distill the inspected set according to the loss values
        distilled_samples_indices, median_sample_indices = confusion_training.distill(args, params, inspection_set,
                                                                                      confusion_iter, criterion_no_reduction,
                                                                                      dataset_name='ember', final_budget=final_budget, custom_arch=arch)

        distilled_set = torch.utils.data.Subset(inspection_set, distilled_samples_indices)

    return distilled_samples_indices, median_sample_indices, model



distilled_samples_indices, median_sample_indices, model = iterative_poison_distillation(inspection_set, inspection_set_inverse,
                                                clean_set, args, start_iter=0)


suspicious_indices = distilled_samples_indices


# save indicies
suspicious_indices.sort()
remain_indices = list( set(range(0,len(inspection_set))) - set(suspicious_indices) )
remain_indices.sort()
save_path = os.path.join(inspection_set_dir, 'cleansed_set_indices_seed=%d' % args.seed)
torch.save(remain_indices, save_path)
print('[Save] %s' % save_path)


if args.debug_info:
    suspicious_indices.sort()
    poison_indices = torch.load(os.path.join(inspection_set_dir, 'poison_indices'))
    num_samples = len(inspection_set)
    num_poison = len(poison_indices)
    num_collected = len(suspicious_indices)
    pt = 0
    recall = 0
    for idx in suspicious_indices:
        if pt >= num_poison:
            break
        while(idx > poison_indices[pt] and pt+1 < num_poison) : pt+=1
        if pt < num_poison and poison_indices[pt] == idx:
            recall += 1
    fpr = num_collected - recall

    print('recall = %d/%d = %f, fpr = %d/%d = %f' % (recall, num_poison, recall / num_poison if num_poison != 0 else 0,
                                                     fpr, num_samples - num_poison,
                                                     fpr / (num_samples - num_poison) if (num_samples - num_poison) != 0 else 0))