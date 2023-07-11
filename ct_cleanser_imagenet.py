'''codes used to call key functional module of confusion training and use it to cleanse poisoned ImageNet
'''
import os, sys
import argparse
import numpy as np
from utils import default_args

parser = argparse.ArgumentParser()
parser.add_argument('-poison_type', type=str,  required=True,
        choices=default_args.parser_choices['poison_type'])
parser.add_argument('-poison_rate', type=float,  required=False,
                    choices=default_args.parser_choices['poison_rate'],
                    default=default_args.parser_default['poison_rate'])
parser.add_argument('-cover_rate', type=float,  required=False,
                    choices=default_args.parser_choices['cover_rate'],
                    default=default_args.parser_default['cover_rate'])
parser.add_argument('-alpha', type=float,  required=False, default=default_args.parser_default['alpha'])
parser.add_argument('-test_alpha', type=float,  required=False, default=None)
parser.add_argument('-trigger', type=str,  required=False,
                    default=None)
parser.add_argument('-devices', type=str, default='0')
parser.add_argument('-debug_info', default=False, action='store_true')
parser.add_argument('-log', default=False, action='store_true')
parser.add_argument('-seed', type=int, required=False, default=default_args.seed)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.devices

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch import nn
from utils import supervisor, tools, resnet, imagenet
import config
import confusion_training

# tools.setup_seed(args.seed)

args.dataset = 'imagenet'
args.trigger = imagenet.triggers[args.poison_type]


if args.log:

    out_path = 'logs'
    if not os.path.exists(out_path): os.mkdir(out_path)
    out_path = os.path.join(out_path, '%s_seed=%s' % (args.dataset, args.seed))
    if not os.path.exists(out_path): os.mkdir(out_path)
    out_path = os.path.join(out_path, 'cleanse')
    if not os.path.exists(out_path): os.mkdir(out_path)
    out_path = os.path.join(out_path, 'CT_%s.out' % (supervisor.get_dir_core(args, include_poison_seed=config.record_poison_seed)))
    fout = open(out_path, 'w')
    ferr = open('/dev/null', 'a')
    sys.stdout = fout
    sys.stderr = ferr


num_classes = 1000
weight_decay = 5e-5
batch_size =  128
kwargs = {'num_workers': 8, 'pin_memory': True}

inspection_set_dir = supervisor.get_poison_set_dir(args)
poison_indices_path = os.path.join(inspection_set_dir, 'poison_indices')
inspection_set_img_dir = os.path.join(inspection_set_dir, 'data')
print('dataset : %s' % inspection_set_dir)
poison_indices = torch.load(poison_indices_path)

train_set_dir = '/path_to_imagenet/train'
test_set_dir = '/path_to_imagenet/val'

inspection_set = imagenet.imagenet_dataset(directory=train_set_dir, shift=False, aug=False,
                 poison_directory=inspection_set_img_dir, poison_indices=poison_indices, target_class=imagenet.target_class,
                 label_file=None, num_classes=1000, scale_for_ct=True)


clean_set = imagenet.imagenet_dataset(directory=test_set_dir, shift=True, aug=False,
                 label_file=imagenet.test_set_labels, num_classes=1000, scale_for_ct=True)
clean_split_meta_dir = os.path.join('clean_set', args.dataset, 'clean_split')
clean_split_indices = torch.load(os.path.join(clean_split_meta_dir, 'clean_split_indices'))
clean_set = torch.utils.data.Subset(clean_set, clean_split_indices)


if args.debug_info:

    poison_transform = imagenet.get_poison_transform_for_imagenet(args.poison_type)

    test_set = imagenet.imagenet_dataset(directory=test_set_dir, shift=False, aug=False,
                                         label_file=imagenet.test_set_labels, num_classes=1000, scale_for_ct=True)
    test_set_backdoor = imagenet.imagenet_dataset(directory=test_set_dir, shift=False, aug=False,
                                         label_file=imagenet.test_set_labels, num_classes=1000, scale_for_ct=True,
                                                  poison_transform=poison_transform)

    test_split_meta_dir = os.path.join('clean_set', args.dataset, 'test_split')
    test_split_indices = torch.load(os.path.join(test_split_meta_dir, 'test_indices'))

    test_set = torch.utils.data.Subset(test_set, test_split_indices)
    test_set_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size, shuffle=False, worker_init_fn=tools.worker_init, **kwargs)

    test_set_backdoor = torch.utils.data.Subset(test_set_backdoor, test_split_indices)
    test_set_backdoor_loader = torch.utils.data.DataLoader(
        test_set_backdoor,
        batch_size=batch_size, shuffle=False, worker_init_fn=tools.worker_init, **kwargs)


    debug_packet = {
        'test_set_loader' : test_set_loader,
        'test_set_backdoor_loader': test_set_backdoor_loader
        #'poison_transform' : poison_transform
    }


def iterative_poison_distillation(inspection_set, clean_set, args, start_iter=0):

    arch = resnet.ResNet18_narrow
    distillation_ratio = [1/5, 1/20, 1/40, 1/80]
    momentums = [0.5, 0.5, 0.5, 0.5, 0.5]
    lrs = [0.01, 0.01, 0.01, 0.01, 0.01]
    lambs = [20, 20, 20, 20, 20]
    batch_factors = [1, 2, 4, 4, 4]

    params = {
        'kwargs': kwargs,
        'inspection_set_dir': inspection_set_dir,
        'num_classes': num_classes,
        'arch': config.arch['imagenet'],
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

    criterion = nn.CrossEntropyLoss().cuda()
    criterion_no_reduction = nn.CrossEntropyLoss(reduction='none').cuda()


    if start_iter != 0:
        distilled_samples_indices, median_sample_indices = confusion_training.distill(args, params, inspection_set,
                                   start_iter-1, criterion_no_reduction, dataset_name='imagenet', custom_arch=arch)
        distilled_set = torch.utils.data.Subset(inspection_set, distilled_samples_indices)

    else:
        distilled_set = inspection_set

    for confusion_iter in range(start_iter, num_confusion_iter):

        lr = lrs[confusion_iter]
        momentum = momentums[confusion_iter]

        size_of_distilled_set = len(distilled_set)
        print('<Round-%d> Size_of_distillation_set = ' % confusion_iter, size_of_distilled_set)

        # different weights for each class based on their frequencies in the distilled set


        freq_of_each_class = np.ones((num_classes,))

        if confusion_iter < 1:
            pretrain_epochs = 20
            pretrain_lr = 0.5
            distillation_iters = 10000
        elif confusion_iter < 3:
            pretrain_epochs = 20
            pretrain_lr = 0.5
            distillation_iters = 5000
        elif confusion_iter < 4:
            pretrain_epochs = 20
            pretrain_lr = 0.5
            distillation_iters = 5000
        else:
            pretrain_epochs = 20
            pretrain_lr = 0.5
            distillation_iters = 5000

        if confusion_iter < 2:
            distilled_set_loader = torch.utils.data.DataLoader(
                distilled_set,
                batch_size=1024, shuffle=True,
                worker_init_fn=tools.worker_init, **kwargs)
        else:
            distilled_set_loader = torch.utils.data.DataLoader(
                torch.utils.data.ConcatDataset([distilled_set, clean_set]),
                batch_size=1024, shuffle=True,
                worker_init_fn=tools.worker_init, **kwargs)


        # pretrain base model
        confusion_training.pretrain(args, debug_packet, arch, num_classes, weight_decay, pretrain_epochs,
                                    distilled_set_loader, criterion, inspection_set_dir, confusion_iter, pretrain_lr, load=False,
                                    dataset_name='imagenet')


        distilled_set_loader = torch.utils.data.DataLoader(
            distilled_set,
            batch_size=batch_size, shuffle=True,
            worker_init_fn=tools.worker_init, **kwargs)

        # confusion_training
        model = confusion_training.confusion_train(args, params, inspection_set, debug_packet, distilled_set_loader, clean_set_loader, confusion_iter, arch,
                                   num_classes, inspection_set_dir, weight_decay, criterion_no_reduction,
                                   momentum, lambs[confusion_iter],
                                   freq_of_each_class, lr, batch_factors[confusion_iter], distillation_iters, dataset_name='imagenet')



        # distill the inspected set according to the loss values
        distilled_samples_indices, median_sample_indices = confusion_training.distill(args, params, inspection_set,
                                                                                      confusion_iter, criterion_no_reduction,
                                                                                      dataset_name='imagenet', custom_arch=arch)

        distilled_set = torch.utils.data.Subset(inspection_set, distilled_samples_indices)

    return distilled_samples_indices, median_sample_indices, model



distilled_samples_indices, median_sample_indices, model = iterative_poison_distillation(inspection_set,
                                                clean_set, args, start_iter=0)




print('to identify poison samples')
# detect backdoor poison samples with the confused model
suspicious_indices = confusion_training.identify_poison_samples_simplified(inspection_set, median_sample_indices,
                                                                model, num_classes=num_classes)


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

