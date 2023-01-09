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
import confusion_training_classwise

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
inspection_set, clean_set, clean_set_no_shift = config.get_dataset(params['inspection_set_dir'], params['data_transform'],
                                               args, num_classes=params['num_classes'])


debug_packet = None
if args.debug_info:
    debug_packet = config.get_packet_for_debug(params['inspection_set_dir'], params['data_transform'],
                                               params['batch_size'], args)


def iterative_poison_distillation(inspection_set, clean_set, clean_set_no_shift, params, args, debug_packet=None, start_iter=0):


    if args.debug_info and (debug_packet is None):
        raise Exception('debug_packet is needed to compute debug info')


    kwargs = params['kwargs']
    inspection_set_dir = params['inspection_set_dir']
    num_classes = params['num_classes']
    weight_decay = params['weight_decay']
    arch = resnet.ResNet18

    momentums =  [0.7, 0.7, 0.7]
    lambs =  [5, 5, 5]  # 30, 15
    lrs = [0.005, 0.005, 0.005]
    batch_factor = [1, 1, 1]

    print('num_classes : ', num_classes)

    # lamb = 20, bf = 20

    clean_set_loader = torch.utils.data.DataLoader(
        clean_set, batch_size=32,
        shuffle=True, worker_init_fn=tools.worker_init, **kwargs)

    distilled_set_loader = torch.utils.data.DataLoader(
        inspection_set,
        batch_size=128, shuffle=True,
        worker_init_fn=tools.worker_init, **kwargs)

    criterion_no_reduction = nn.CrossEntropyLoss(reduction='none')
    criterion = nn.CrossEntropyLoss()

    print('>>> pretrain')
    #confusion_training_classwise.pretrain(args, debug_packet, arch, num_classes, weight_decay, 40,
    #                                    distilled_set_loader, criterion, inspection_set_dir, 0.01, load=False)


    print('>>> Iterative Data Distillation with Confusion Training')
    distilled_samples_indices, median_sample_indices = None, None


    num_confusion_iter = 1
    distillation_iters = 400


    suspicious_indices = []

    for current_class in range(2, num_classes):

        """
        top_indices_each_class = [[] for _ in range(num_classes)]
        num = len(inspection_set)
        for i in range(num):
            _, gt = inspection_set[i]
            gt = gt.item()
            top_indices_each_class[gt].append(i)"""

        top_indices_each_class = confusion_training_classwise.distill(current_class, arch, args, params, inspection_set,
                                       2, criterion_no_reduction, class_wise=True, debug=True)


        budget = len(top_indices_each_class[current_class]) // 4
        distilled_set = torch.utils.data.Subset(inspection_set, top_indices_each_class[current_class][:budget])
        class_size = len(distilled_set)

        for confusion_iter in range(num_confusion_iter):

            size_of_distilled_set = len(distilled_set)
            print('<Class-%d, Round-%d> Size_of_distillation_set = ' % (current_class, confusion_iter), size_of_distilled_set)

            lr = lrs[confusion_iter]
            freq_of_each_class = np.ones( (num_classes,) )

            """
            distilled_set_loader = torch.utils.data.DataLoader(
                torch.utils.data.ConcatDataset([distilled_set, clean_set]),
                batch_size=64, shuffle=True,
                worker_init_fn=tools.worker_init, **kwargs)"""


            distilled_set_loader = torch.utils.data.DataLoader(
                distilled_set,
                batch_size=32, shuffle=True,
                worker_init_fn=tools.worker_init, **kwargs)

            model = confusion_training_classwise.confusion_train(args, debug_packet, distilled_set_loader, clean_set_loader,
                                                       confusion_iter, arch,
                                                       num_classes, inspection_set_dir, weight_decay,
                                                       criterion_no_reduction,
                                                       momentums[confusion_iter], lambs[confusion_iter],
                                                       freq_of_each_class, lr, batch_factor[confusion_iter],
                                                       distillation_iters)

            ####################################################################################################

            distilled_set_loader = torch.utils.data.DataLoader(
                distilled_set,
                batch_size=params['batch_size'], shuffle=False,
                worker_init_fn=tools.worker_init, **kwargs)

            """
            model.train()
            train_loss = 0
            tot = 0
            for data, target in distilled_set_loader:
                data, target = data.cuda(), target.cuda()
                output = model(data)
                loss_vals = criterion_no_reduction(output, target)
                train_loss += loss_vals.sum()
                tot += len(target)
            train_loss /= tot
            print('training_loss = %f' % train_loss.item())

            ####################################################################################################"""

            distilled_set_loader = torch.utils.data.DataLoader(
                distilled_set,
                batch_size=params['batch_size'], shuffle=False,
                worker_init_fn=tools.worker_init, **kwargs)

            model.eval()
            test_loss = 0
            tot = 0
            for data, target in distilled_set_loader:
                data, target = data.cuda(), target.cuda()
                output = model(data)
                loss_vals = criterion_no_reduction(output, target)
                test_loss += loss_vals.sum()
                tot += len(target)
            test_loss /= tot
            print('test_loss = %f' % test_loss.item())

            ####################################################################################################



            top_indices_each_class = confusion_training_classwise.distill(current_class, arch, args, params, inspection_set,
                                       confusion_iter, criterion_no_reduction, class_wise=True)


            if confusion_iter == num_confusion_iter - 1:

                distilled_set = torch.utils.data.Subset(inspection_set, top_indices_each_class[current_class])
                likelihood_ratio, isolated_indices_local = \
                    confusion_training_classwise.identify_poison_samples_simplified(
                        distilled_set, model)
                print('class-%d : likelihood_ratio = %f' % (current_class, likelihood_ratio) )

                isolated_indices = []
                for i in isolated_indices_local:
                    isolated_indices.append(top_indices_each_class[current_class][i])

                poison_indices = torch.load(os.path.join(inspection_set_dir, 'poison_indices'))

                num_poison_within_class = 0
                for i in poison_indices:
                    _, gt = inspection_set[i]
                    gt = int(gt.item())
                    if gt == current_class:
                        num_poison_within_class += 1

                num_poison = len(poison_indices)
                detected = 0
                for pid in isolated_indices:
                    if pid in poison_indices: detected+=1
                num_fp = len(isolated_indices) - detected


                recall = detected/num_poison_within_class if num_poison_within_class!=0 else 0

                print('Recall = %d/%d = %f, FPR = %d/%d = %f' % (detected, num_poison_within_class, recall,
                                                                 num_fp, class_size, num_fp/class_size) )

                suspicious_indices += list(isolated_indices)

            else:
                num_to_extract = size_of_distilled_set // 2
                distilled_set = torch.utils.data.Subset(inspection_set, top_indices_each_class[current_class][:num_to_extract])


    return suspicious_indices




suspicious_indices = iterative_poison_distillation(inspection_set,
                                                clean_set, clean_set_no_shift, params,
                                                args, debug_packet, start_iter=0)



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