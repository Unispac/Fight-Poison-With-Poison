import argparse
import os, sys
import time
from tqdm import tqdm
from utils import default_args, imagenet
from torch.cuda.amp import autocast, GradScaler

parser = argparse.ArgumentParser()
parser.add_argument('-dataset', type=str, required=False,
                    default=default_args.parser_default['dataset'],
                    choices=default_args.parser_choices['dataset'])
parser.add_argument('-poison_type', type=str, required=False,
                    default='none',
                    choices=default_args.parser_choices['poison_type'])
parser.add_argument('-poison_rate', type=float,  required=False,
                    choices=default_args.parser_choices['poison_rate'],
                    default=default_args.parser_default['poison_rate'])
parser.add_argument('-cover_rate', type=float, required=False,
                    choices=default_args.parser_choices['cover_rate'],
                    default=default_args.parser_default['cover_rate'])
parser.add_argument('-ember_options', type=str, required=False,
                    choices=['constrained', 'unconstrained', 'none'],
                    default='unconstrained')
parser.add_argument('-alpha', type=float, required=False,
                    default=default_args.parser_default['alpha'])
parser.add_argument('-test_alpha', type=float, required=False, default=None)
parser.add_argument('-trigger', type=str, required=False,
                    default=None)
parser.add_argument('-no_aug', default=False, action='store_true')
parser.add_argument('-no_normalize', default=False, action='store_true')
parser.add_argument('-devices', type=str, default='0')
parser.add_argument('-log', default=False, action='store_true')
parser.add_argument('-seed', type=int, required=False, default=default_args.seed)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "%s" % args.devices

import config
from torchvision import datasets, transforms
from torch import nn
import torch
from utils import supervisor, tools


if args.trigger is None:
    if args.dataset != 'imagenette' and args.dataset != 'imagenet':
        args.trigger = config.trigger_default[args.poison_type]
    elif args.dataset == 'imagenet':
        args.trigger = imagenet.triggers[args.poison_type]
    else:
        if args.poison_type == 'badnet':
            args.trigger = 'badnet_high_res.png'
        else:
            raise NotImplementedError('%s not implemented for imagenette' % args.poison_type)


all_to_all = False
if args.poison_type == 'badnet_all_to_all':
    all_to_all = True


tools.setup_seed(args.seed)

if args.log:
    out_path = 'logs'
    if not os.path.exists(out_path): os.mkdir(out_path)
    out_path = os.path.join(out_path, '%s_seed=%s' % (args.dataset, args.seed))
    if not os.path.exists(out_path): os.mkdir(out_path)
    out_path = os.path.join(out_path, 'base')
    if not os.path.exists(out_path): os.mkdir(out_path)
    out_path = os.path.join(out_path, '%s_%s.out' % (supervisor.get_dir_core(args, include_poison_seed=config.record_poison_seed), 'no_aug' if args.no_aug else 'aug'))
    fout = open(out_path, 'w')
    ferr = open('/dev/null', 'a')
    sys.stdout = fout
    sys.stderr = ferr

if args.dataset == 'cifar10':

    data_transform_aug = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]),
    ])

    data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
    ])

elif args.dataset == 'gtsrb':

    data_transform_aug = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
    ])

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
    ])

elif args.dataset == 'imagenette':

    data_transform_aug = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

elif args.dataset == 'imagenet':
    print('[ImageNet]')

elif args.dataset == 'ember':
    print('[Non-image Dataset] Ember')
else:
    raise NotImplementedError('dataset %s not supported' % args.dataset)



if args.dataset == 'cifar10':

    num_classes = 10
    arch = config.arch[args.dataset]
    momentum = 0.9
    weight_decay = 1e-4
    epochs = 100
    milestones = torch.tensor([50, 75])
    learning_rate = 0.1
    batch_size = 128

elif args.dataset == 'gtsrb':

    num_classes = 43
    arch = config.arch[args.dataset]
    momentum = 0.9
    weight_decay = 1e-4
    epochs = 100
    milestones = torch.tensor([30, 60])
    learning_rate = 0.01
    batch_size = 128

elif args.dataset == 'imagenette':

    num_classes = 10
    arch = config.arch[args.dataset]
    momentum = 0.9
    weight_decay = 1e-4
    epochs = 100
    milestones = torch.tensor([40, 80])
    learning_rate = 0.1
    batch_size = 128

elif args.dataset == 'imagenet':

    num_classes = 1000
    arch = config.arch[args.dataset]
    momentum = 0.9
    weight_decay = 1e-4
    epochs = 90
    milestones = torch.tensor([30, 60])
    learning_rate = 0.1
    batch_size = 256

elif args.dataset == 'ember':

    num_classes = 2
    arch = config.arch[args.dataset]
    momentum = 0.9
    weight_decay = 1e-6
    epochs = 10
    learning_rate = 0.1
    milestones = torch.tensor([])
    batch_size = 512

else:

    print('<Undefined Dataset> Dataset = %s' % args.dataset)
    raise NotImplementedError('<To Be Implemented> Dataset = %s' % args.dataset)


if args.dataset == 'imagenet':
    kwargs = {'num_workers': 32, 'pin_memory': True}
else:
    kwargs = {'num_workers': 4, 'pin_memory': True}




# Set Up Poisoned Set

if args.dataset != 'ember' and args.dataset != 'imagenet':
    poison_set_dir = supervisor.get_poison_set_dir(args)
    poisoned_set_img_dir = os.path.join(poison_set_dir, 'data')
    poisoned_set_label_path = os.path.join(poison_set_dir, 'labels')
    poison_indices_path = os.path.join(poison_set_dir, 'poison_indices')


    print('dataset : %s' % poisoned_set_img_dir)

    poisoned_set = tools.IMG_Dataset(data_dir=poisoned_set_img_dir,
                                     label_path=poisoned_set_label_path, transforms=data_transform if args.no_aug else data_transform_aug)

    poisoned_set_loader = torch.utils.data.DataLoader(
        poisoned_set,
        batch_size=batch_size, shuffle=True, worker_init_fn=tools.worker_init, **kwargs)

elif args.dataset == 'imagenet':

    poison_set_dir = supervisor.get_poison_set_dir(args)
    poison_indices_path = os.path.join(poison_set_dir, 'poison_indices')
    poisoned_set_img_dir = os.path.join(poison_set_dir, 'data')
    print('dataset : %s' % poison_set_dir)

    poison_indices = torch.load(poison_indices_path)

    root_dir = '/shadowdata/xiangyu/imagenet_256/'
    #'./data/imagenet/'
    #'/shadowdata/xiangyu/imagenet_256/'
    #train_set_dir = './data/imagenet/train'
    #test_set_dir = './data/imagenet/val'
    train_set_dir = os.path.join(root_dir, 'train')
    test_set_dir = os.path.join(root_dir, 'val')

    from utils import imagenet
    #imagenet_ffcv
    poisoned_set = imagenet.imagenet_dataset(directory=train_set_dir, poison_directory=poisoned_set_img_dir,
                                             poison_indices = poison_indices, target_class=imagenet.target_class,
                                             num_classes=1000)

    poisoned_set_loader = torch.utils.data.DataLoader(
        poisoned_set,
        batch_size=batch_size, shuffle=True, worker_init_fn=tools.worker_init, **kwargs)

    """
    (self, directory, shift=False, aug=True,
                 poison_directory=None, poison_indices=None,
                 label_file=None, target_class = None, num_classes=1000, scale_for_ct=False)
    """


    """
    poisoned_set = imagenet.imagenet_dataset(directory=train_set_dir, shift=False,
                 poison_directory=poisoned_set_img_dir, poison_indices=poison_indices, target_class=imagenet.target_class,
                 label_file=None, num_classes=1000)

    poisoned_set_loader = imagenet_ffcv.get_ffcv_loader(dataset=poisoned_set, nick_name='poison_%s' % args.poison_type,
                                                        batch_size=batch_size, aug=True)"""

else:
    poison_set_dir = os.path.join('poisoned_train_set', 'ember', args.ember_options)
    poison_indices_path = os.path.join(poison_set_dir, 'poison_indices')

    #stats_path = os.path.join('data', 'ember', 'stats')
    poisoned_set = tools.EMBER_Dataset( x_path=os.path.join(poison_set_dir, 'watermarked_X.npy'),
                                        y_path=os.path.join(poison_set_dir, 'watermarked_y.npy'))
    print('dataset : %s' % poison_set_dir)

    poisoned_set_loader = torch.utils.data.DataLoader(
        poisoned_set,
        batch_size=batch_size, shuffle=True, worker_init_fn=tools.worker_init, **kwargs)


if args.dataset != 'ember' and args.dataset != 'imagenet':

    # Set Up Test Set for Debug & Evaluation
    test_set_dir = os.path.join('clean_set', args.dataset, 'test_split')
    test_set_img_dir = os.path.join(test_set_dir, 'data')
    test_set_label_path = os.path.join(test_set_dir, 'labels')
    test_set = tools.IMG_Dataset(data_dir=test_set_img_dir,
                                 label_path=test_set_label_path, transforms=data_transform)
    test_set_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size, shuffle=False, worker_init_fn=tools.worker_init, **kwargs)

    # Poison Transform for Testing
    poison_transform = supervisor.get_poison_transform(poison_type=args.poison_type, dataset_name=args.dataset,
                                                       target_class=config.target_class[args.dataset], trigger_transform=data_transform,
                                                       is_normalized_input=True,
                                                       alpha=args.alpha if args.test_alpha is None else args.test_alpha,
                                                       trigger_name=args.trigger, args=args)


elif args.dataset == 'imagenet':

    poison_transform = imagenet.get_poison_transform_for_imagenet(args.poison_type)

    test_set = imagenet.imagenet_dataset(directory=test_set_dir, shift=False, aug=False,
                 label_file=imagenet.test_set_labels, num_classes=1000)
    test_set_backdoor = imagenet.imagenet_dataset(directory=test_set_dir, shift=False, aug=False,
                 label_file=imagenet.test_set_labels, num_classes=1000, poison_transform=poison_transform)

    test_split_meta_dir = os.path.join('clean_set', args.dataset, 'test_split')
    test_indices = torch.load(os.path.join(test_split_meta_dir, 'test_indices'))

    test_set = torch.utils.data.Subset(test_set, test_indices)
    test_set_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size, shuffle=False, worker_init_fn=tools.worker_init, **kwargs)

    test_set_backdoor = torch.utils.data.Subset(test_set_backdoor, test_indices)
    test_set_backdoor_loader = torch.utils.data.DataLoader(
        test_set_backdoor,
        batch_size=batch_size, shuffle=False, worker_init_fn=tools.worker_init, **kwargs)



else:
    normalizer = poisoned_set.normal

    test_set_dir = os.path.join('clean_set', args.dataset, 'test_split')

    test_set = tools.EMBER_Dataset(x_path=os.path.join(test_set_dir, 'X.npy'),
                                   y_path=os.path.join(test_set_dir, 'Y.npy'),
                                   normalizer = normalizer)

    test_set_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size, shuffle=False, worker_init_fn=tools.worker_init, **kwargs)


    backdoor_test_set_dir = os.path.join('poisoned_train_set', 'ember', args.ember_options)
    backdoor_test_set = tools.EMBER_Dataset(x_path=os.path.join(poison_set_dir, 'watermarked_X_test.npy'),
                                       y_path=None, normalizer = normalizer)
    backdoor_test_set_loader = torch.utils.data.DataLoader(
        backdoor_test_set,
        batch_size=batch_size, shuffle=False, worker_init_fn=tools.worker_init, **kwargs)


"""
from torchvision.models import resnet18, ResNet18_Weights
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model = nn.DataParallel(model).cuda()
model.eval()


with torch.no_grad():

    correct = 0
    tot = 0
    for imgs, labels in tqdm(test_set_loader):
        imgs = imgs.cuda()
        output = model(imgs)
        preds = torch.argmax(output, dim=1).detach().cpu()
        tot += len(labels)
        correct += (preds == labels).sum()
    print('test set accuracy = %d/%d = %f' % (correct, tot, correct / tot))

    correct = 0
    tot = 0
    for imgs, labels in tqdm(poisoned_set_loader):
        imgs = imgs.cuda()
        output = model(imgs)
        preds = torch.argmax(output, dim=1).detach().cpu()
        tot += len(labels)
        correct += (preds == labels).sum()
    print('training set accuracy = %d/%d = %f' % (correct, tot, correct/tot))


exit(0)"""






# Train Code

if args.dataset != 'ember':
    model = arch(num_classes=num_classes)
else:
    model = arch()


milestones = milestones.tolist()
model = nn.DataParallel(model)
model = model.cuda()


if args.dataset != 'ember':
    print(f"Will save to '{supervisor.get_model_dir(args)}'.")
    if os.path.exists(supervisor.get_model_dir(args)):
        print(f"Model '{supervisor.get_model_dir(args)}' already exists!")

    if args.dataset == 'imagenet':
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss().cuda()
else:
    model_path = os.path.join('poisoned_train_set', 'ember', args.ember_options, 'backdoored_model.pt')
    print(f"Will save to '{model_path}'.")
    if os.path.exists(model_path):
        print(f"Model '{model_path}' already exists!")
    criterion = nn.BCELoss().cuda()

optimizer = torch.optim.SGD(model.parameters(), learning_rate, momentum=momentum, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones)

if args.poison_type == 'TaCT' or args.poison_type == 'SleeperAgent':
    source_classes = [config.source_class]
else:
    source_classes = None

import time
st = time.time()

"""
if args.dataset == 'imagenet':
    tools.test_imagenet(model=model, test_loader=test_set_loader,
                                     poison_transform=poison_transform)
    print('<time : %f minutes>' % ( (time.time() - st) / 60 ))
"""

scaler = GradScaler()
for epoch in range(1, epochs+1):  # train backdoored base model
    start_time = time.perf_counter()
    
    # Train
    model.train()
    preds = []
    labels = []
    for data, target in tqdm(poisoned_set_loader):

        data = data.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad(set_to_none=True)

        with autocast():
            output = model(data)
            loss = criterion(output, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print('<Backdoor Training> Train Epoch: {} \tLoss: {:.6f}, lr: {:.6f}, Time: {:.2f}s'.format(epoch, loss.item(), optimizer.param_groups[0]['lr'], elapsed_time))
    scheduler.step()

    # Test

    if args.dataset != 'ember':
        if True:
        # if epoch % 5 == 0:
            if args.dataset == 'imagenet':
                tools.test_imagenet(model=model, test_loader=test_set_loader,
                                    test_backdoor_loader=test_set_backdoor_loader)
                torch.save(model.module.state_dict(), supervisor.get_model_dir(args))
            else:
                tools.test(model=model, test_loader=test_set_loader, poison_test=True,
                           poison_transform=poison_transform, num_classes=num_classes, source_classes=source_classes, all_to_all=all_to_all)
                torch.save(model.module.state_dict(), supervisor.get_model_dir(args))
    else:

        tools.test_ember(model=model, test_loader=test_set_loader,
                             backdoor_test_loader=backdoor_test_set_loader)
        torch.save(model.module.state_dict(), model_path)
    print("")

if args.dataset != 'ember':
    torch.save(model.module.state_dict(), supervisor.get_model_dir(args))
else:
    torch.save(model.module.state_dict(), model_path)

if args.poison_type == 'none':
    if args.no_aug:
        torch.save(model.module.state_dict(), f'models/{args.dataset}_vanilla_no_aug.pt')
        torch.save(model.module.state_dict(), f'models/{args.dataset}_vanilla_no_aug_seed={args.seed}.pt')
    else:
        torch.save(model.module.state_dict(), f'models/{args.dataset}_vanilla_aug.pt')
        torch.save(model.module.state_dict(), f'models/{args.dataset}_vanilla_aug_seed={args.seed}.pt')