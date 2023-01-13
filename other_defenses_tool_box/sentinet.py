#!/usr/bin/env python3

# from ..backdoor_defense import BackdoorDefense
# from trojanvision.environ import env
# from trojanzoo.utils import to_numpy

from turtle import pos
import torch, torchvision
import numpy as np
from sklearn import metrics
from tqdm import tqdm
from .tools import AverageMeter, generate_dataloader, tanh_func, to_numpy, jaccard_idx, normalize_mad, unpack_poisoned_train_set
from . import BackdoorDefense
import config, os
from utils import supervisor
from matplotlib import pyplot as plt


class SentiNet(BackdoorDefense):
    """
    Assuming oracle knowledge of the used trigger.
    """
    
    name: str = 'sentinet'

    def __init__(self, args, defense_fpr: float = 0.05, N: int = 100):
        super().__init__(args)
        self.args = args
        
        # Only support localized attacks
        support_list = ['adaptive_patch', 'badnet', 'badnet_all_to_all', 'dynamic', 'TaCT']
        assert args.poison_type in support_list
        assert args.dataset in ['cifar10', 'gtsrb']

        self.defense_fpr = defense_fpr
        self.N = N
        self.folder_path = 'other_defenses_tool_box/results/Sentinet'
        if not os.path.exists(self.folder_path):
            os.mkdir(self.folder_path)
        self.random_img = self.normalizer(torch.rand((3, self.img_size, self.img_size))).cuda()

    def detect(self):
        args = self.args
        loader = generate_dataloader(dataset=self.dataset,
                                    dataset_path=config.data_dir,
                                    batch_size=1,
                                    split='valid',
                                    shuffle=True,
                                    drop_last=False)
        loader = tqdm(loader)
        
        clean_loader = generate_dataloader(dataset=self.dataset,
                                            dataset_path=config.data_dir,
                                            batch_size=100,
                                            split='test',
                                            shuffle=True,
                                            drop_last=False)
        clean_subset, _ = torch.utils.data.random_split(clean_loader.dataset, [self.N, len(clean_loader.dataset) - self.N])
        clean_loader = torch.utils.data.DataLoader(clean_subset, batch_size=100, shuffle=False, drop_last=False, num_workers=4, pin_memory=True)
        
        clean_fooled = []
        clean_avgconf = []
        poison_fooled = []
        poison_avgconf = []
        
        for i, (_input, _label) in enumerate(loader):
            if i > 30: break
            # For the clean input
            _input, _label = _input.cuda(), _label.cuda()
            fooled_num = 0
            avgconf = 0
            # simulate GradCAM map with a randomized central square area for clean inputs
            from random import random
            from numpy.random import normal
            if args.dataset == 'gtsrb':
                scale = random() * 4 + 2
            elif args.dataset == 'cifar10':
                scale = random() * 4 + 2
            else: raise NotImplementedError()
            # scale = torch.normal(mean=4.0, std=2.0, size=(1,)).clamp(2, 6).item()
            # scale = 6
            for c_input, c_label in clean_loader:
                adv_input = c_input.clone().cuda()
                inert_input = c_input.clone().cuda()
                
                # dx = dy = 26
                # posx = self.img_size // 2 - dx // 2
                # posy = self.img_size // 2 - dy // 2
                
                # adv_input[:, :, posx:posx+dx, posy:posy+dy] = _input[:, :, posx:posx+dx, posy:posy+dy]
                # inert_input[:, :, posx:posx+dx, posy:posy+dy] = self.normalizer(torch.rand((3, dx, dy))).cuda()
                
                st_cd = int(self.img_size / scale)
                ed_cd = self.img_size - st_cd
                
                adv_input[:, :, st_cd:ed_cd, st_cd:ed_cd] = _input[0, :, st_cd:ed_cd, st_cd:ed_cd]
                # inert_input[:, :, st_cd:ed_cd, st_cd:ed_cd] = self.normalizer(torch.rand((3, ed_cd - st_cd, ed_cd - st_cd))).cuda()
                inert_input[:, :, st_cd:ed_cd, st_cd:ed_cd] = self.normalizer(torch.rand((inert_input.shape[0], 3, ed_cd - st_cd, ed_cd - st_cd))).cuda()
                # inert_input[:, :, st_cd:ed_cd, st_cd:ed_cd] = torch.normal(mean=0.5, std=1.0, size=(3, ed_cd - st_cd, ed_cd - st_cd)).clamp(0, 1).cuda()
                # inert_input[:, :, st_cd:ed_cd, st_cd:ed_cd] = self.random_img[:, st_cd:ed_cd, st_cd:ed_cd]
                
                adv_output = self.model(adv_input)
                adv_pred = torch.argmax(adv_output, dim=1)
                fooled_num += torch.eq(adv_pred, _label).sum()
                
                inert_output = self.model(inert_input)
                inert_conf = torch.softmax(inert_output, dim=1)
                # avgconf += torch.cat([inert_conf[x, y].unsqueeze(0) for x, y in list(zip(range(len(adv_pred)), adv_pred.tolist()))]).sum()
                avgconf += inert_conf.max(dim=1)[0].sum()
            
            fooled = fooled_num / len(clean_loader.dataset)
            avgconf /= len(clean_loader.dataset)
            # print(avgconf)
            clean_fooled.append(fooled.item())
            clean_avgconf.append(avgconf.item())
            
            # For the poison input
            poison_input, poison_label = self.poison_transform.transform(_input, _label)
            fooled_num = 0
            avgconf = 0
            for c_input, c_label in clean_loader:
                adv_input = c_input.clone().cuda()
                inert_input = c_input.clone().cuda()
                
                # Oracle (approximate) knowledge to the trigger position
                if args.poison_type == 'badnet' or args.poison_type == 'badnet_all_to_all':
                    dx = dy = 5
                    posx = self.img_size - dx
                    posy = self.img_size - dy
                    
                    adv_input[:, :, posx:posx+dx, posy:posy+dy] = poison_input[0, :, posx:posx+dx, posy:posy+dy]
                    inert_input[:, :, posx:posx+dx, posy:posy+dy] = self.normalizer(torch.rand((inert_input.shape[0], 3, dx, dy))).cuda()
                    # inert_input[:, :, posx:posx+dx, posy:posy+dy] = self.random_img[:, posx:posx+dx, posy:posy+dy]
                elif args.poison_type == 'TaCT':
                    dx = dy = 16
                    posx = self.img_size - dx
                    posy = self.img_size - dy
                    
                    adv_input[:, :, posx:posx+dx, posy:posy+dy] = poison_input[0, :, posx:posx+dx, posy:posy+dy]
                    inert_input[:, :, posx:posx+dx, posy:posy+dy] = self.normalizer(torch.rand((inert_input.shape[0], 3, dx, dy))).cuda()
                    # inert_input[:, :, posx:posx+dx, posy:posy+dy] = self.random_img[:, posx:posx+dx, posy:posy+dy]
                elif args.poison_type == 'dynamic' or args.poison_type == 'adaptive_patch':
                    trigger_mask = ((poison_input - _input).abs() > 1e-4)[0].cuda()
                    # print(trigger_mask.sum())
                    # print(poison_input.reshape(-1)[:10], _input.reshape(-1)[:10], trigger_mask.reshape(-1)[:10])
                    # exit()
                    adv_input[:, trigger_mask] = poison_input[0, trigger_mask]
                    # self.debug_save_img(adv_input[1])
                    # exit()
                    inert_input[:, trigger_mask] = self.normalizer(torch.rand(inert_input.shape))[:, trigger_mask].cuda()
                    # self.debug_save_img(inert_input[1])
                    # exit()
                else: raise NotImplementedError()
                
                adv_output = self.model(adv_input)
                adv_pred = torch.argmax(adv_output, dim=1)
                fooled_num += torch.eq(adv_pred, poison_label).sum()
                
                inert_output = self.model(inert_input)
                inert_conf = torch.softmax(inert_output, dim=1)
                # avgconf += torch.cat([inert_conf[x, y].unsqueeze(0) for x, y in list(zip(range(len(adv_pred)), adv_pred.tolist()))]).sum()
                avgconf += inert_conf.max(dim=1)[0].sum()

            fooled = fooled_num / len(clean_loader.dataset)
            avgconf /= len(clean_loader.dataset)
            poison_fooled.append(fooled.item())
            poison_avgconf.append(avgconf.item())

        plt.scatter(clean_avgconf, clean_fooled, marker='o', color='blue', s=5, alpha=1.0)
        plt.scatter(poison_avgconf, poison_fooled, marker='^', s=8, color='red', alpha=0.7)
        save_path = 'assets/SentiNet_%s.png' % (supervisor.get_dir_core(args))
        plt.xlabel("AvgConf")
        plt.ylabel("#Fooled")
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.savefig(save_path)
        
        
        print("Saved figure at {}".format(save_path))
        plt.clf()
        raise NotImplementedError("Further implementation not finished yet!")
        exit()
        
        y_true
        y_pred
        
        print(f'Inputs with entropy among thresholds ({threshold_low:5.3f}, {threshold_high:5.3f}) are considered benign.')
        print('Filtered input num:', torch.eq(y_pred, 1).sum().item())
        print('fpr:', (((clean_entropy < threshold_low).int().sum() + (clean_entropy > threshold_high).int().sum()) / len(clean_entropy)).item())
        print("f1_score:", metrics.f1_score(y_true, y_pred))
        print("precision_score:", metrics.precision_score(y_true, y_pred))
        print("recall_score:", metrics.recall_score(y_true, y_pred))
        print("accuracy_score:", metrics.accuracy_score(y_true, y_pred))
    
    
    def debug_save_img(self, t, path='a.png'):
        torchvision.utils.save_image(self.denormalizer(t.reshape(3, self.img_size, self.img_size)), path)