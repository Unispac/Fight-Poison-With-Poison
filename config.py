from utils import resnet, vgg, mobilenetv2, ember_nn
from utils import supervisor
from utils import tools
import torch
from torchvision import transforms
import os


data_dir = './data' # defaul clean dataset directory
triggers_dir = './triggers' # default triggers directory
target_class = {
    'cifar10' : 0,
    'gtsrb' : 2,
    'imagenette': 0,
    'imagenet' : 0
}

# default target class (without loss of generality)
source_class = 1 # default source class for TaCT
cover_classes = [5,7] # default cover classes for TaCT
poison_seed = 0
record_poison_seed = True
record_model_arch = False

trigger_default = {
    'adaptive': 'hellokitty_32.png',
    'adaptive_blend': 'hellokitty_32.png',
    'adaptive_mask': 'hellokitty_32.png',
    'adaptive_patch': 'none',
    'clean_label' : 'badnet_patch4_dup_32.png',
    'basic' : 'badnet_patch_32.png',
    'badnet' : 'badnet_patch.png',
    'blend' : 'hellokitty_32.png',
    'refool': 'none',
    'TaCT' : 'trojan_square_32.png',
    'SIG' : 'none',
    'WaNet': 'none',
    'dynamic' : 'none',
    'ISSBA': 'none',
    'SleeperAgent': 'none',
    'none' : 'none',
    'badnet_all_to_all' : 'badnet_patch.png',
}

arch = {
    ### for base model & poison distillation
    'cifar10': resnet.ResNet18,
    'gtsrb' : resnet.ResNet18,
    'imagenette': resnet.ResNet18,
    'ember': ember_nn.EmberNN,
    'imagenet' : resnet.ResNet18
    #'abl':  wresnet.WideResNet
}


# adapitve-patch triggers for different datasets
adaptive_patch_train_trigger_names = {
    'cifar10': [
        'phoenix_corner_32.png',
        'firefox_corner_32.png',
        'badnet_patch4_32.png',
        'trojan_square_32.png',
    ],
    'gtsrb': [
        'phoenix_corner_32.png',
        'firefox_corner_32.png',
        'badnet_patch4_32.png',
        'trojan_square_32.png',
    ],
}

adaptive_patch_train_trigger_alphas = {
    'cifar10': [
        0.5,
        0.2,
        0.5,
        0.3,
    ],
    'gtsrb': [
        0.5,
        0.2,
        0.5,
        0.3,
    ],
}

adaptive_patch_test_trigger_names = {
    'cifar10': [
        'phoenix_corner2_32.png',
        'badnet_patch4_32.png',
    ],
    'gtsrb': [
        'firefox_corner_32.png',
        'trojan_square_32.png',
    ],
}

adaptive_patch_test_trigger_alphas = {
    'cifar10': [
        1,
        1,
    ],
    'gtsrb': [
        1,
        1,
    ],
}


def get_params(args):

    if args.dataset == 'cifar10':

        num_classes = 10

        data_transform_normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]),
        ])

        data_transform_aug = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]),
        ])

    elif args.dataset == 'gtsrb':

        num_classes = 43

        data_transform_normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
        ])

        data_transform_aug = transforms.Compose([
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
        ])

    elif args.dataset == 'imagenette':

        num_classes = 10

        data_transform_normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        data_transform_aug = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    else:
        raise NotImplementedError('<Unimplemented Dataset> %s' % args.dataset)


    params = {
        'data_transform' : data_transform_normalize,
        'data_transform_aug' : data_transform_aug,
        ####################################################################
        #'distillation_ratio': [1/2, 1/5, 1/25, 1/50, 1/100, 1/100, 1/100],
        #'momentums': [0.5, 0.5, 0.5, 0.7, 0.7, 0.7, 0.7, 0.7],
        #'lambs': [20, 40, 60, 60, 60, 60, 40, 10],
        #'lrs': [0.001, 0.001, 0.003, 0.003, 0.003, 0.003, 0.003, 0.005],
        #'batch_factors': [2, 2, 2, 2, 2, 2, 2, 2],
        ####################################################################


        #'distillation_ratio': [1/2, 1/5, 1/25, 1/50],
        #'momentums': [0.5, 0.5, 0.5, 0.7, 0.7],
        #'lambs': [20, 40, 60, 30, 30],
        #'lrs': [0.001, 0.001, 0.003, 0.005, 0.005],
        #'batch_factors': [2, 2, 2, 8, 4],

        'distillation_ratio': [1/2, 1/5, 1/25, 1/50],
        'momentums': [0.5, 0.5, 0.5, 0.7, 0.7], # 0.5, 0.7
        'lambs': [20, 40, 60, 30, 20], # 30, 15
        'lrs': [0.001, 0.001, 0.003, 0.005, 0.005],
        'batch_factors': [2, 2, 2, 2, 2], # 4,4

        'weight_decay' : 1e-4,
        'num_classes' : num_classes,
        'batch_size' : 32,
        'pretrain_epochs' : 100,
        'median_sample_rate': 0.1,
        'base_arch' :  arch[args.dataset],
        'arch' :  arch[args.dataset],
        'kwargs' : {'num_workers': 2, 'pin_memory': True},
        'inspection_set_dir': supervisor.get_poison_set_dir(args)
    }


    return params


def get_dataset(inspection_set_dir, data_transform, args, num_classes = 10):

    print('|num_classes = %d|' % num_classes)

    # Set Up Inspection Set (dataset that is to be inspected
    inspection_set_img_dir = os.path.join(inspection_set_dir, 'data')
    inspection_set_label_path = os.path.join(inspection_set_dir, 'labels')
    inspection_set = tools.IMG_Dataset(data_dir=inspection_set_img_dir,
                                     label_path=inspection_set_label_path, transforms=data_transform)

    # Set Up Clean Set (the small clean split at hand for defense
    clean_set_dir = os.path.join('clean_set', args.dataset, 'clean_split')
    clean_set_img_dir = os.path.join(clean_set_dir, 'data')
    clean_label_path = os.path.join(clean_set_dir, 'clean_labels')
    clean_set = tools.IMG_Dataset(data_dir=clean_set_img_dir,
                                  label_path=clean_label_path, transforms=data_transform,
                                  num_classes=num_classes, shift=True)

    clean_set_random = tools.IMG_Dataset(data_dir=clean_set_img_dir,
                                  label_path=clean_label_path, transforms=data_transform,
                                  num_classes=num_classes, random_labels=True)

    return inspection_set, clean_set, clean_set_random


def get_packet_for_debug(poison_set_dir, data_transform, batch_size, args):

    # Set Up Test Set for Debug & Evaluation
    test_set_dir = os.path.join('clean_set', args.dataset, 'test_split')
    test_set_img_dir = os.path.join(test_set_dir, 'data')
    test_set_label_path = os.path.join(test_set_dir, 'labels')
    test_set = tools.IMG_Dataset(data_dir=test_set_img_dir,
                                 label_path=test_set_label_path, transforms=data_transform)


    kwargs = {'num_workers': 2, 'pin_memory': True}
    test_set_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size, shuffle=True, worker_init_fn=tools.worker_init, **kwargs)

    trigger_transform = data_transform
    poison_transform = supervisor.get_poison_transform(poison_type=args.poison_type, dataset_name=args.dataset,
                                                       target_class=target_class[args.dataset],
                                                       trigger_transform=trigger_transform,
                                                       is_normalized_input=True,
                                                       alpha=args.alpha if args.test_alpha is None else args.test_alpha,
                                                       trigger_name=args.trigger, args=args)

    poison_indices = torch.load(os.path.join(poison_set_dir, 'poison_indices'))

    if args.poison_type == 'TaCT':
        source_classes = [source_class]
    else:
        source_classes = None

    debug_packet = {
        'test_set_loader' : test_set_loader,
        'poison_transform' : poison_transform,
        'poison_indices' : poison_indices,
        'source_classes' : source_classes
    }

    return debug_packet