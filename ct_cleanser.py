import os, sys
import argparse
import numpy as np
from utils import default_args

parser = argparse.ArgumentParser()
parser.add_argument('-dataset', type=str, required=False, default=default_args.parser_default['dataset'],
                    choices=default_args.parser_choices['dataset'])
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
from utils import supervisor, tools, resnet
import config
import confusion_training

tools.setup_seed(args.seed)

if args.trigger is None:

    if args.dataset != 'imagenette':
        args.trigger = config.trigger_default[args.poison_type]

    else:
        if args.poison_type == 'badnet':
            args.trigger = 'badnet_high_res.png'
        else:
            raise NotImplementedError('%s not implemented for imagenette' % args.poison_type)

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

params = config.get_params(args)
inspection_set, clean_set = config.get_dataset(params['inspection_set_dir'], params['data_transform'],
                                               args, num_classes=params['num_classes'])

debug_packet = None
if args.debug_info:
    debug_packet = config.get_packet_for_debug(params['inspection_set_dir'], params['data_transform'],
                                               params['batch_size'], args)


def iterative_poison_distillation(inspection_set, clean_set, params, args, debug_packet=None, start_iter=0):

    if args.debug_info and (debug_packet is None):
        raise Exception('debug_packet is needed to compute debug info')

    kwargs = params['kwargs']
    inspection_set_dir = params['inspection_set_dir']
    num_classes = params['num_classes']
    pretrain_epochs = params['pretrain_epochs']
    weight_decay = params['weight_decay']
    arch = params['arch']
    distillation_ratio = params['distillation_ratio']
    momentums = params['momentums']
    lambs = params['lambs']
    lrs = params['lrs']
    batch_factor = params['batch_factors']

    clean_set_loader = torch.utils.data.DataLoader(
        clean_set, batch_size=params['batch_size'],
        shuffle=True, worker_init_fn=tools.worker_init, **kwargs)

    print('>>> Iterative Data Distillation with Confusion Training')

    distilled_samples_indices, median_sample_indices = None, None
    num_confusion_iter = len(distillation_ratio) + 1
    criterion_no_reduction = nn.CrossEntropyLoss(reduction='none')
    criterion = nn.CrossEntropyLoss()


    if start_iter != 0:
        distilled_samples_indices, _ = confusion_training.distill(args, params, inspection_set,
                                   start_iter-1, criterion_no_reduction)
        distilled_set = torch.utils.data.Subset(inspection_set, distilled_samples_indices)

    else:
        distilled_set = inspection_set

    for confusion_iter in range(start_iter, num_confusion_iter):

        size_of_distilled_set = len(distilled_set)
        print('<Round-%d> Size_of_distillation_set = ' % confusion_iter, size_of_distilled_set)

        # different weights for each class based on their frequencies in the distilled set
        nums_of_each_class = np.zeros(num_classes)
        for i in range(size_of_distilled_set):
            _, gt = distilled_set[i]
            gt = gt.item()
            nums_of_each_class[gt] += 1
        print(nums_of_each_class)
        freq_of_each_class = nums_of_each_class / size_of_distilled_set
        freq_of_each_class = np.sqrt(freq_of_each_class + 0.001)

        if confusion_iter >= 4 :

            if confusion_iter == num_confusion_iter-1:
                freq_of_each_class[:] = 1

            lr = lrs[confusion_iter]
            distilled_set_loader = torch.utils.data.DataLoader(
                torch.utils.data.ConcatDataset([distilled_set, clean_set]),
                batch_size=params['batch_size'], shuffle=True,
                worker_init_fn=tools.worker_init, **kwargs)
        else:
            lr = lrs[confusion_iter]
            distilled_set_loader = torch.utils.data.DataLoader(
                distilled_set,
                batch_size=params['batch_size'], shuffle=True,
                worker_init_fn=tools.worker_init, **kwargs)

        # pretrain base model
        confusion_training.pretrain(args, debug_packet, arch, num_classes, weight_decay, pretrain_epochs,
                                    distilled_set_loader, criterion, inspection_set_dir, confusion_iter)


        distilled_set_loader = torch.utils.data.DataLoader(
            distilled_set,
            batch_size=params['batch_size'], shuffle=True,
            worker_init_fn=tools.worker_init, **kwargs)

        # confusion_training
        model = confusion_training.confusion_train(args, debug_packet, distilled_set_loader, clean_set_loader, confusion_iter, arch,
                                   num_classes, inspection_set_dir, weight_decay, criterion_no_reduction,
                                   momentums[confusion_iter], lambs[confusion_iter],
                                   freq_of_each_class, lr, batch_factor[confusion_iter])

        # distill the inspected set according to the loss values
        distilled_samples_indices, median_sample_indices = confusion_training.distill(args, params, inspection_set,
                                                                                      confusion_iter, criterion_no_reduction)

        distilled_set = torch.utils.data.Subset(inspection_set, distilled_samples_indices)

    return distilled_samples_indices, median_sample_indices, model



# iterative confusion training
distilled_samples_indices, median_sample_indices, model = iterative_poison_distillation(inspection_set,
                                                clean_set, params, args, debug_packet, start_iter=0)

"""
arch = params['arch']
num_classes = params['num_classes']
inspection_set_dir = params['inspection_set_dir']
model = arch(num_classes=num_classes)
model.load_state_dict(torch.load(os.path.join(inspection_set_dir, 'confused_%d_seed=%d.pt' % (7, args.seed))))
model = nn.DataParallel(model)
model = model.cuda()
criterion_no_reduction = nn.CrossEntropyLoss(reduction='none')
distilled_samples_indices, median_sample_indices = confusion_training.distill(args, params, inspection_set,
                                                                7, criterion_no_reduction)"""

print('to identify poison samples')
# detect backdoor poison samples with the confused model
suspicious_indices = confusion_training.identify_poison_samples_simplified(inspection_set, median_sample_indices,
                                                                model, num_classes=params['num_classes'])


# save indicies
suspicious_indices.sort()
remain_indices = list( set(range(0,len(inspection_set))) - set(suspicious_indices) )
remain_indices.sort()
save_path = os.path.join(params['inspection_set_dir'], 'cleansed_set_indices_seed=%d' % args.seed)
torch.save(remain_indices, save_path)
print('[Save] %s' % save_path)


if args.debug_info:
    suspicious_indices.sort()
    poison_indices = torch.load(os.path.join(params['inspection_set_dir'], 'poison_indices'))
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