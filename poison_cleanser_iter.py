import os, sys
import argparse
import numpy as np
from utils import default_args
import random

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
from utils import supervisor, tools
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
inspection_set, clean_set = config.get_dataset(params['inspection_set_dir'], params['data_transform'], args)

debug_packet = None
if args.debug_info:
    debug_packet = config.get_packet_for_debug(params['inspection_set_dir'], params['data_transform'],
                                               params['batch_size'], args)


distilled_samples_indices, median_sample_indices = confusion_training.iterative_poison_distillation(
    inspection_set, clean_set, params, args, debug_packet)
distilled_set = torch.utils.data.Subset(inspection_set, distilled_samples_indices)



inference_model = confusion_training.generate_inference_model(
    distilled_set, clean_set, params, args, debug_packet)


print('<start inference>')

model = inference_model

kwargs = {'num_workers': 2, 'pin_memory': True}
inspection_set_loader = torch.utils.data.DataLoader(inspection_set, batch_size=params['batch_size'],
                                                            shuffle=False, **kwargs)



criterion_no_reduction = nn.CrossEntropyLoss(reduction='none')
loss_array = []
model.eval()

with torch.no_grad():

    for data, target in inspection_set_loader:
        data, target = data.cuda(), target.cuda()
        output = model(data)
        batch_loss = criterion_no_reduction(output, target)
        for loss_val in batch_loss:
            loss_array.append(loss_val.item())

loss_array = np.array(loss_array)
sorted_indices = np.argsort(loss_array)

print('<loss values collected>')

num_classes = 10
condensation_num = 2000
num_samples = 50000
median_sample_rate = params['median_sample_rate']
distilled_samples_indices = list(sorted_indices[:condensation_num])

median_sample_indices = []
sorted_indices_each_class = [[] for _ in range(num_classes)]
for temp_id in sorted_indices:
    _, gt = inspection_set[temp_id]
    sorted_indices_each_class[gt.item()].append(temp_id)

for i in range(num_classes):
    num_class_i = len(sorted_indices_each_class[i])
    st = int(num_class_i / 2 - num_class_i * median_sample_rate / 2)
    ed = int(num_class_i / 2 + num_class_i * median_sample_rate / 2)
    for temp_id in range(st, ed):
        median_sample_indices.append(sorted_indices_each_class[i][temp_id])


class_dist = np.zeros(num_classes, dtype=int)
for t in distilled_samples_indices:
    _, gt = inspection_set[t]
    class_dist[gt.item()] += 1

median_indices_each_class = [[] for _ in range(num_classes)]
for t in median_sample_indices:
    _, gt = inspection_set[t]
    median_indices_each_class[gt.item()].append(t)

# slightly rebalance the distilled set
for i in range(num_classes):
    minimal_sample_num = len(sorted_indices_each_class[i]) // 20  # 5% of each class
    if class_dist[i] < minimal_sample_num:
        for _ in range(class_dist[i], minimal_sample_num):
            s = random.randint(0, len(median_indices_each_class[i]) - 1)
            distilled_samples_indices.append(median_indices_each_class[i][s])


distilled_samples_indices.sort()
median_sample_indices.sort()
head = distilled_samples_indices

inspection_set_dir = params['inspection_set_dir']


print('get median ... ')

if True:

    if args.poison_type == 'TaCT' or args.poison_type == 'adaptive_blend':
        cover_indices = torch.load(os.path.join(inspection_set_dir, 'cover_indices'))

    poison_indices = torch.load(os.path.join(inspection_set_dir, 'poison_indices'))

    cnt = 0
    for s, cid in enumerate(head):  # enumerate the head part
        original_id = cid
        if original_id in poison_indices:
            cnt += 1
    print('How Many Poison Samples are Concentrated in the Head? --- %d/%d' % (cnt, len(poison_indices)))

    cover_dist = []
    poison_dist = []
    for temp_id in range(num_samples):
        if sorted_indices[temp_id] in poison_indices:
            poison_dist.append(temp_id)

        if args.poison_type == 'TaCT' or args.poison_type == 'adaptive_blend':
            if sorted_indices[temp_id] in cover_indices:
                cover_dist.append(temp_id)
    print('poison distribution : ', poison_dist)
    if args.poison_type == 'TaCT' or args.poison_type == 'adaptive_blend':
        print('cover distribution : ', cover_dist)
    print('collected : %d' % len(head))


exit(0)






print('>>> Dataset Cleanse ...')
num_classes = params['num_classes']

suspicious_indices = confusion_training.cleanser(args = args, inspection_set=inspection_set, clean_set_indices = median_sample_indices,
                               model=inference_model, num_classes=num_classes)

suspicious_indices.sort()
remain_indices = list( set(range(0,len(inspection_set))) - set(suspicious_indices) )
remain_indices.sort()

remian_dist = np.zeros(num_classes)
for temp_id in remain_indices:
    _, gt = inspection_set[temp_id]
    gt = gt.item()
    remian_dist[gt]+=1
print('remain dist : ', remian_dist)


save_path = os.path.join(params['inspection_set_dir'], 'cleansed_set_indices_seed=%d' % args.seed)
torch.save(remain_indices, save_path)
print('[Save] %s' % save_path)

if args.debug_info: # evaluate : how many poison samples are eliminated ?

    poison_indices = debug_packet['poison_indices']
    poison_indices.sort()

    true_positive = 0
    num_positive = 0
    num_negative = 0
    false_positive = 0

    tot_poison = len(poison_indices)
    num_samples = len(inspection_set)

    pt = 0
    for pid in range(num_samples):
        while pt+1 < tot_poison and poison_indices[pt] < pid: pt+=1
        if pt < tot_poison and poison_indices[pt] == pid : num_positive+=1
        else: num_negative+=1

    pt = 0
    for pid in suspicious_indices:
        while pt+1 < tot_poison and poison_indices[pt] < pid: pt+=1
        if pt < tot_poison and poison_indices[pt] == pid: true_positive+=1
        else: false_positive+=1

    tpr = 0 if num_positive == 0 else true_positive/num_positive
    print('Elimination Rate = %d/%d = %f' % (true_positive, num_positive, tpr) )
    fpr = 0 if num_negative == 0 else false_positive/num_negative
    print('Sacrifice Rate = %d/%d = %f' % (false_positive, num_negative, fpr))