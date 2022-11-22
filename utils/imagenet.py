import torch
import numpy as np

import os
import os.path
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.utils import save_image



target_class = 0

triggers= {
    'badnet': 'badnet_high_res.png',
    'blend' : 'random_224.png',
    'trojan' : 'xxx.png',
}

#test_set_labels = 'data/imagenet/ILSVRC2012_validation_ground_truth.txt'
test_set_labels = '/shadowdata/xiangyu/imagenet_256/ILSVRC2012_validation_ground_truth.txt'
#'data/imagenet/ILSVRC2012_validation_ground_truth.txt'

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

resized_normalizer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=[256, 256]),
            normalize,
])

normalizer = transforms.Compose([
            transforms.ToTensor(),
            normalize,
])

transform_no_aug = transforms.Compose([
            transforms.Resize(size=[224, 224]),
        ])

transform_aug = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip()
        ])


# confusion training will scale image to smaller sizes for efficienct
# since the goal of confusion training is just to identify poison samples, this scaling will not impact the effectivness.
scale_for_confusion_training = transforms.Compose([
    transforms.Resize(size=[64, 64]),
])



def find_classes(directory: str) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    """Finds the class folders in a dataset.
    See :class:`DatasetFolder` for details.
    """
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

    idx_to_class = {i: cls_name for i, cls_name in enumerate(classes)}

    return classes, class_to_idx, idx_to_class



def assign_img_identifier(directory, classes):

    num_imgs = 0
    img_id_to_path = []
    img_labels = []

    for i, cls_name in enumerate(classes):
        cls_dir = os.path.join(directory, cls_name)
        img_entries = sorted(entry.name for entry in os.scandir(cls_dir))

        for img_entry in img_entries:
            entry_path = os.path.join(cls_name, img_entry)
            img_id_to_path.append(entry_path)
            img_labels.append(i)
            num_imgs += 1

    return num_imgs, img_id_to_path, img_labels




class imagenet_dataset(Dataset):
    def __init__(self, directory, shift=False, aug=True,
                 poison_directory=None, poison_indices=None,
                 label_file=None, target_class = None, num_classes=1000, scale_for_ct=False):

        self.num_classes = num_classes
        self.shift = shift

        if label_file is None: # divide classes by directory
            self.classes, self.class_to_idx, self.idx_to_class = find_classes(directory)
            self.num_imgs, self.img_id_to_path, self.img_labels = assign_img_identifier(directory, self.classes)
        else: # samples from all classes are in the same directory
            entries = sorted(entry.name for entry in os.scandir(directory))
            self.num_imgs = len(entries)
            self.img_id_to_path = []
            for i, img_name in enumerate(entries):
                self.img_id_to_path.append(img_name)
            self.img_labels = []
            label_file = open(label_file)
            line = label_file.readline()
            while line:
                self.img_labels.append(int(line))
                line = label_file.readline()

        self.img_labels = torch.LongTensor(self.img_labels)

        self.is_poison = [False for _ in range(self.num_imgs)]


        if poison_indices is not None:
            for i in poison_indices:
                self.is_poison[i] = True

        self.poison_directory = poison_directory
        self.aug = aug
        self.directory = directory
        self.target_class = target_class
        if self.target_class is not None:
            self.target_class = torch.tensor(self.target_class).long()

        self.scale_for_ct = scale_for_ct


        for i in range(self.num_imgs):
            if self.is_poison[i]:
                self.img_id_to_path[i] = os.path.join(self.poison_directory, self.img_id_to_path[i])
                self.img_labels[i] = self.target_class
            else:
                self.img_id_to_path[i] = os.path.join(self.directory, self.img_id_to_path[i])
                if self.shift:
                    self.img_labels[i] = (self.img_labels[i] + 1) % self.num_classes





    def __len__(self):
        return self.num_imgs

    def __getitem__(self, idx):

        idx = int(idx)

        img_path = self.img_id_to_path[idx]
        label = self.img_labels[idx]

        img = normalizer(Image.open(img_path).convert("RGB")) # 256 x 256

        if self.aug:
            img = transform_aug(img)
        else:
            img = transform_no_aug(img)

        if self.scale_for_ct:
            img = scale_for_confusion_training(img)


        return img, label


def get_poison_transform_for_imagenet(poison_type):

    trigger_path = 'triggers/%s' % triggers[poison_type]

    if poison_type == 'badnet':
        trigger = normalizer(Image.open(trigger_path).convert("RGB"))
        return badnet_transform(trigger, target_class=target_class)
    elif poison_type == 'trojan':
        raise NotImplementedError('%s is not implemented on ImageNet' % poison_type)
    elif poison_type == 'blend':
        trigger = resized_normalizer(Image.open(trigger_path).convert("RGB"))
        return blend_transform(trigger, target_class=target_class)
    else:
        raise NotImplementedError('%s is not implemented on ImageNet' % poison_type)


def get_poison_transform_for_imagenet_no_normalize(poison_type):

    transform_to_tensor = transforms.Compose([
            transforms.ToTensor()
    ])

    transform_resize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=[256, 256]),
    ])

    trigger_path = 'triggers/%s' % triggers[poison_type]

    if poison_type == 'badnet':
        trigger = transform_to_tensor(Image.open(trigger_path).convert("RGB"))
        return badnet_transform(trigger, target_class=target_class)
    elif poison_type == 'trojan':
        raise NotImplementedError('%s is not implemented on ImageNet' % poison_type)
    elif poison_type == 'blend':
        trigger = transform_resize(Image.open(trigger_path).convert("RGB"))
        return blend_transform(trigger, target_class=target_class)
    else:
        raise NotImplementedError('%s is not implemented on ImageNet' % poison_type)




def get_batch_poison_transform_for_imagenet(poison_type, scale_for_ct=False):

    resized_224_normalizer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=[224, 224]),
        normalize,
    ])

    resized_64_normalizer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=[64, 64]),
        normalize,
    ])

    img_size = 64 if scale_for_ct else 224

    trigger_path = 'triggers/%s' % triggers[poison_type]

    if poison_type == 'badnet':
        trigger = normalizer(Image.open(trigger_path).convert("RGB")).cuda()
        return badnet_transform_batch(trigger, target_class=target_class, img_size=img_size)
    elif poison_type == 'trojan':
        raise NotImplementedError('%s is not implemented on ImageNet' % poison_type)
    elif poison_type == 'blend':

        if scale_for_ct:
            trigger = resized_64_normalizer(Image.open(trigger_path).convert("RGB")).cuda()
        else:
            trigger = resized_224_normalizer(Image.open(trigger_path).convert("RGB")).cuda()

        return blend_transform_batch(trigger, target_class=target_class, img_size=img_size)
    else:
        raise NotImplementedError('%s is not implemented on ImageNet' % poison_type)


class badnet_transform():

    def __init__(self, trigger, target_class = 0, img_size = 256):
        self.img_size = img_size
        self.trigger = trigger
        self.target_class = target_class # by default : target_class = 0
        # shape of the patch trigger
        _, self.dx, self.dy = trigger.shape

    def transform(self, data, label):
        # transform clean samples to poison samples
        posx = self.img_size - self.dx
        posy = self.img_size - self.dy
        data[:,posx:,posy:] = self.trigger
        return data, self.target_class


class blend_transform():
    def __init__(self, trigger, target_class=0, alpha=0.2, img_size = 256):
        self.img_size = img_size
        self.trigger = trigger
        self.target_class = target_class  # by default : target_class = 0
        self.alpha = alpha

    def transform(self, data, label):
        data = (1 - self.alpha) * data + self.alpha * self.trigger
        return data, self.target_class




class badnet_transform_batch():

    def __init__(self, trigger, target_class = 0, img_size = 224):
        self.img_size = img_size
        self.trigger = trigger
        self.target_class = target_class # by default : target_class = 0
        # shape of the patch trigger
        _, self.dx, self.dy = trigger.shape

    def transform(self, data, labels):
        # transform clean samples to poison samples
        data, labels = data.clone(), labels.clone()
        posx = self.img_size - self.dx
        posy = self.img_size - self.dy
        data[:, :, posx:,posy:] = self.trigger
        labels[:] = self.target_class
        return data, labels


class blend_transform_batch():
    def __init__(self, trigger, target_class = 0, alpha=0.2, img_size = 224):
        self.img_size = img_size
        self.trigger = trigger
        self.target_class = target_class  # by default : target_class = 0
        self.alpha = alpha

    def transform(self, data, labels):
        data, labels = data.clone(), labels.clone()
        data = (1 - self.alpha) * data + self.alpha * self.trigger
        labels[:] = self.target_class
        return data, labels


transform_resize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=[256, 256]),
    ])


"""
def sub_process(pars):

    src_cls_dir_path, dst_cls_dir_path, img_entry = pars
    src_img_path = os.path.join(src_cls_dir_path, img_entry)
    dst_img_path = os.path.join(dst_cls_dir_path, img_entry)
    scaled_img = transform_resize(Image.open(src_img_path).convert("RGB"))
    save_image(scaled_img, dst_img_path)
    print('save : ', dst_img_path)"""


def create_256_scaled_version(src_directory, dst_directory, is_train_set=True):

    import time

    st = time.time()

    if is_train_set:
        classes = sorted(entry.name for entry in os.scandir(src_directory) if entry.is_dir())
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {src_directory}.")


        cnt = 41
        classes = classes[42:]
        tot = len(classes)

        for cls_name in classes:

            print('start :', cls_name)

            cnt += 1

            dst_cls_dir_path = os.path.join(dst_directory, cls_name)
            if not os.path.exists(dst_cls_dir_path):
                os.mkdir(dst_cls_dir_path)
            src_cls_dir_path = os.path.join(src_directory, cls_name)
            img_entries = sorted(entry.name for entry in os.scandir(src_cls_dir_path))

            #with Pool(8) as p:
            #    p.map(sub_process, pars_set)


            for img_entry in img_entries:
                src_img_path = os.path.join(src_cls_dir_path, img_entry)
                dst_img_path = os.path.join(dst_cls_dir_path, img_entry)
                scaled_img = transform_resize(Image.open(src_img_path).convert("RGB"))
                save_image(scaled_img, dst_img_path)

            print('[time: %f minutes] progress by classes [%d/%d], done : %s' % ( (time.time() - st)/60, cnt, tot, cls_name) )


    else:

        img_entries = sorted(entry.name for entry in os.scandir(src_directory))
        tot = len(img_entries)
        for i, img_entry in enumerate(img_entries):
            src_img_path = os.path.join(src_directory, img_entry)
            dst_img_path = os.path.join(dst_directory, img_entry)
            scaled_img = transform_resize(Image.open(src_img_path).convert("RGB"))
            save_image(scaled_img, dst_img_path)
            print('[time: %f minutes] progress : [%d/%d]' % ((time.time() - st)/60, i+1, tot))





if __name__ == "__main__":

    #create_256_scaled_version('/shadowdata/xiangyu/imagenet/train', '/shadowdata/xiangyu/imagenet_256/train', is_train_set=True)

    create_256_scaled_version('/shadowdata/xiangyu/imagenet/val', '/shadowdata/xiangyu/imagenet_256/val',
                              is_train_set=False)


    #train_set = imagenet_dataset(directory='data/imagenet/train', transforms=None)
    #test_set = imagenet_dataset(directory='data/imagenet/val', transforms=None, label_file='data/imagenet/ILSVRC2012_validation_ground_truth.txt')


    #print('train_set_size:', len(train_set))
    #print('test_set_size:', len(test_set))


