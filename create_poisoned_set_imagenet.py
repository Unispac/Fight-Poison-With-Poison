import os
import torch
import argparse
from utils import default_args, tools, supervisor, imagenet
import torchvision.transforms as transforms
import random
from PIL import Image
from torchvision.utils import save_image

parser = argparse.ArgumentParser()
parser.add_argument('-poison_type', type=str,  required=False,
                    choices=default_args.parser_choices['poison_type'],
                    default=default_args.parser_default['poison_type'])
parser.add_argument('-poison_rate', type=float,  required=False,
                    choices=default_args.parser_choices['poison_rate'],
                    default=default_args.parser_default['poison_rate'])
args = parser.parse_args()
args.dataset = 'imagenet'
args.alpha= 0.2
tools.setup_seed(0)

if args.poison_type not in ['none', 'badnet', 'trojan', 'blend']:
    raise NotImplementedError('%s is not implemented on ImageNet' % args.poison_type)

if args.poison_type == 'none':
    args.poison_rate = 0

if not os.path.exists(os.path.join('poisoned_train_set', 'imagenet')):
    os.mkdir(os.path.join('poisoned_train_set', 'imagenet'))


poison_set_dir = supervisor.get_poison_set_dir(args)
if not os.path.exists(poison_set_dir):
    os.mkdir(poison_set_dir)

poison_imgs_dir = os.path.join(poison_set_dir, 'data')
if not os.path.exists(poison_imgs_dir):
    os.mkdir(poison_imgs_dir)

num_imgs = 1281167 # size of imagenet training set


# random sampling
id_set = list(range(0,num_imgs))
random.shuffle(id_set)
num_poison = int(num_imgs * args.poison_rate)
poison_indices = id_set[:num_poison]
poison_indices.sort() # increasing order


train_set_dir = '/shadowdata/xiangyu/imagenet_256/train'

classes, class_to_idx, idx_to_class = imagenet.find_classes(train_set_dir)
num_imgs, img_id_to_path, img_labels = imagenet.assign_img_identifier(train_set_dir, classes)

transform_to_tensor = transforms.Compose([
    transforms.ToTensor(),
])

poison_transform = imagenet.get_poison_transform_for_imagenet(args.poison_type)


cnt = 0
tot = len(poison_indices)
print('# poison samples = %d' % tot)
for pid in poison_indices:
    cnt+=1
    ori_img = transform_to_tensor(Image.open( os.path.join(train_set_dir, img_id_to_path[pid]) ).convert("RGB"))
    poison_img, _ = poison_transform.transform(ori_img, None)

    cls_path = os.path.join(poison_imgs_dir, idx_to_class[img_labels[pid]])
    if not os.path.exists(cls_path):
        os.mkdir(cls_path)

    dst_path = os.path.join(poison_imgs_dir, img_id_to_path[pid])
    save_image(poison_img, dst_path)
    print('save [%d/%d]: %s' % (cnt,tot, dst_path))



poison_indices_path = os.path.join(poison_set_dir, 'poison_indices')
torch.save(poison_indices, poison_indices_path)
print('[Generate Poisoned Set] Save %s' % poison_indices_path)