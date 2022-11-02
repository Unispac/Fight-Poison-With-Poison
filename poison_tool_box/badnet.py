import os
import torch
import random
from torchvision.utils import save_image

class poison_generator():

    def __init__(self, img_size, dataset, poison_rate, trigger, path, target_class = 0):

        self.img_size = img_size
        self.dataset = dataset
        self.poison_rate = poison_rate
        self.trigger = trigger
        self.path = path  # path to save the dataset
        self.target_class = target_class # by default : target_class = 0

        # shape of the patch trigger
        _, self.dx, self.dy = trigger.shape

        print('trigger_size : %d x %d' % (self.dx, self.dy))

        # number of images
        self.num_img = len(dataset)

    def generate_poisoned_training_set(self):

        # poison for placing trigger pattern
        posx = self.img_size - self.dx
        posy = self.img_size - self.dy

        # random sampling
        id_set = list(range(0,self.num_img))
        random.shuffle(id_set)
        num_poison = int(self.num_img * self.poison_rate)
        poison_indices = id_set[:num_poison]
        poison_indices.sort() # increasing order

        print('poison_indicies : ', poison_indices)

        label_set = []
        pt = 0
        for i in range(self.num_img):
            img, gt = self.dataset[i]

            if pt < num_poison and poison_indices[pt] == i:
                gt = self.target_class
                img[:,posx:,posy:] = self.trigger
                pt+=1

            img_file_name = '%d.png' % i
            img_file_path = os.path.join(self.path, img_file_name)
            save_image(img, img_file_path)
            print('[Generate Poisoned Set] Save %s' % img_file_path)
            label_set.append(gt)

        label_set = torch.LongTensor(label_set)

        return poison_indices, label_set



class poison_transform():
    def __init__(self, img_size, trigger, target_class = 0):
        self.img_size = img_size
        self.trigger = trigger
        self.target_class = target_class # by default : target_class = 0
        # shape of the patch trigger
        _, self.dx, self.dy = trigger.shape

        print(self.img_size, self.target_class, self.dx, self.dy, self.trigger.shape)

    def transform(self, data, labels):

        data = data.clone()
        labels = labels.clone()

        # transform clean samples to poison samples
        posx = self.img_size - self.dx
        posy = self.img_size - self.dy
        labels[:] = self.target_class
        data[:,:,posx:,posy:] = self.trigger
        return data, labels