import os
import torch
from torch import nn
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR
from utils import tools

# extract features
def get_features(data_loader, model):
    label_list = []
    preds_list = []
    feats = []
    gt_confidence = []
    loss_vals = []
    criterion_no_reduction = nn.CrossEntropyLoss(reduction='none')
    model.eval()
    with torch.no_grad():
        for i, (ins_data, ins_target) in enumerate(tqdm(data_loader)):
            ins_data, ins_target = ins_data.cuda(), ins_target.cuda()
            output, x_features = model(ins_data, return_hidden=True)

            loss = criterion_no_reduction(output, ins_target).cpu().numpy()

            preds = torch.argmax(output, dim=1).cpu().numpy()
            prob = torch.softmax(output, dim=1).cpu().numpy()
            this_batch_size = len(ins_target)
            for bid in range(this_batch_size):
                gt = ins_target[bid].cpu().item()
                feats.append(x_features[bid].cpu().numpy())
                label_list.append(gt)
                preds_list.append(preds[bid])
                gt_confidence.append(prob[bid][gt])
                loss_vals.append(loss[bid])
    return feats, label_list, preds_list, gt_confidence, loss_vals


def identify_poison_samples_simplified(inspection_set, model):

    from scipy.stats import multivariate_normal

    kwargs = {'num_workers': 4, 'pin_memory': True}
    num_samples = len(inspection_set)

    # main dataset we aim to cleanse
    inspection_split_loader = torch.utils.data.DataLoader(
        inspection_set,
        batch_size=128, shuffle=False, worker_init_fn=tools.worker_init, **kwargs)

    model.eval()
    feats_inspection, class_labels_inspection, \
    preds_inspection, gt_confidence_inspection, loss_vals = get_features(inspection_split_loader, model)
    feats_inspection = np.array(feats_inspection)
    class_labels_inspection = np.array(class_labels_inspection)
    temp_feats = torch.FloatTensor(feats_inspection).cuda()

    # reduce dimensionality
    U, S, V = torch.pca_lowrank(temp_feats, q=2)
    projected_feats = torch.matmul(temp_feats, V[:, :2]).cpu()

    # isolate samples via the confused inference model

    isolated_indices_local = []
    other_indices_local = []
    labels = []

    for i in range(num_samples):

        print(gt_confidence_inspection[i])

        if preds_inspection[i] == class_labels_inspection[i]:
            isolated_indices_local.append(i)
            labels.append(1)
        else:
            other_indices_local.append(i)
            labels.append(0)

    projected_feats_isolated = projected_feats[isolated_indices_local]
    projected_feats_other = projected_feats[other_indices_local]

    num_isolated = projected_feats_isolated.shape[0]

    print('num_isolated : ', num_isolated)

    if num_isolated >= 2 and num_isolated < num_samples - 2:

        mu = np.zeros((2,2))
        covariance = np.zeros((2,2,2))

        mu[0] = projected_feats_other.mean(axis=0)
        covariance[0] = np.cov(projected_feats_other.T)

        mu[1] = projected_feats_isolated.mean(axis=0)
        covariance[1] = np.cov(projected_feats_isolated.T)

        covariance += 0.001

        single_cluster_likelihood = 0
        two_clusters_likelihood = 0

        for i in range(num_samples):

            single_cluster_likelihood += multivariate_normal.logpdf(x=projected_feats[i:i + 1], mean=mu[0],
                                                                        cov=covariance[0], allow_singular=True).sum()

            two_clusters_likelihood += multivariate_normal.logpdf(x=projected_feats[i:i + 1], mean=mu[labels[i]],
                                                                      cov=covariance[labels[i]], allow_singular=True).sum()

        likelihood_ratio = np.exp((two_clusters_likelihood - single_cluster_likelihood) /num_samples)

    else:

        likelihood_ratio = 1


    return likelihood_ratio, isolated_indices_local



# pretraining on the poisoned datast to learn a prior of the backdoor
def pretrain(args, debug_packet, arch, num_classes, weight_decay, pretrain_epochs, distilled_set_loader, criterion,
             inspection_set_dir, lr, load=True, dataset_name=None):

    all_to_all = False
    if args.poison_type == 'badnet_all_to_all':
        all_to_all = True

    ######### Pretrain Base Model ##############
    model = arch(num_classes=num_classes)
    if load:
        print('load:', os.path.join(inspection_set_dir, 'pretrain_classwise_base_seed=%d.pt' % (args.seed)) )
        ckpt = torch.load(os.path.join(inspection_set_dir, 'pretrain_classwise_base_seed=%d.pt' % (args.seed)))
        model.load_state_dict(ckpt)
    model = nn.DataParallel(model)
    model = model.cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr,  momentum=0.9, weight_decay=weight_decay)
    for epoch in range(1, pretrain_epochs + 1):  # pretrain backdoored base model with the distilled set
        model.train()

        for batch_idx, (data, target) in enumerate( tqdm(distilled_set_loader) ):
            optimizer.zero_grad()
            data, target = data.cuda(), target.cuda()  # train set batch
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        if epoch % 5 == 0:
            print('<Pretrain> Train Epoch: {}/{} \tLoss: {:.6f}'.format(epoch, pretrain_epochs, loss.item()))
            if args.debug_info:
                model.eval()

                if dataset_name != 'ember' and dataset_name != 'imagenet':
                    tools.test(model=model, test_loader=debug_packet['test_set_loader'], poison_test=True,
                           poison_transform=debug_packet['poison_transform'], num_classes=num_classes,
                           source_classes=debug_packet['source_classes'], all_to_all = all_to_all)
                elif dataset_name == 'imagenet':
                    tools.test_imagenet(model=model, test_loader=debug_packet['test_set_loader'],
                                        poison_transform=debug_packet['poison_transform'])
                else:
                    tools.test_ember(model=model, test_loader=debug_packet['test_set_loader'],
                                     backdoor_test_loader=debug_packet['backdoor_test_set_loader'])


    base_ckpt = model.module.state_dict()

    if not load:
        torch.save(base_ckpt, os.path.join(inspection_set_dir, 'pretrain_classwise_base_seed=%d.pt' % (args.seed)))
        print('save : ', os.path.join(inspection_set_dir, 'pretrain_classwise_base_seed=%d.pt' % (args.seed)))
    else:
        torch.save(base_ckpt, os.path.join(inspection_set_dir, 'classwise_base_seed=%d.pt' % (args.seed)))
        print('save : ', os.path.join(inspection_set_dir, 'classwise_base_seed=%d.pt' % (args.seed)))

    return model


# confusion training : joint training on the poisoned dataset and a randomly labeled small clean set (i.e. confusion set)
def confusion_train(args, debug_packet, distilled_set_loader, clean_set_loader, confusion_iter, arch,
                    num_classes, inspection_set_dir, weight_decay, criterion_no_reduction,
                    momentum, lamb, freq, lr, batch_factor, distillation_iters, dataset_name = None):

    all_to_all = False
    if args.poison_type == 'badnet_all_to_all':
        all_to_all = True

    ######### Distillation Step ################
    model = arch(num_classes=num_classes)
    #model.load_state_dict(
    #            torch.load(os.path.join(inspection_set_dir, 'pretrain_classwise_base_seed=%d.pt' % (args.seed))))
    #print('load : ', os.path.join(inspection_set_dir, 'pretrain_classwise_base_seed=%d.pt' % (args.seed)) )

    model.load_state_dict(
                torch.load(os.path.join(inspection_set_dir, 'confused_2_seed=%d.pt' % (args.seed))))
    print('load : ', os.path.join(inspection_set_dir, 'confused_2_seed=%d.pt' % (args.seed)) )

    print('<----- freeze_non_bn ------>')
    model.freeze_none_bn()

    model = nn.DataParallel(model)
    model = model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay,
                                momentum=momentum)

    distilled_set_iters = iter(distilled_set_loader)
    clean_set_iters = iter(clean_set_loader)

    rounder = 0

    for batch_idx in tqdm(range(distillation_iters)):
        model.train()

        try:
            data_shift, target_shift = next(clean_set_iters)
        except Exception as e:
            clean_set_iters = iter(clean_set_loader)
            data_shift, target_shift = next(clean_set_iters)
        data_shift, target_shift = data_shift.cuda(), target_shift.cuda()


        #rid = batch_idx // 100
        #if (rid + rounder + 1) % num_classes == 0:
        #    rounder += 1

        if dataset_name != 'ember':

            #target_confusion = target_shift
            #model.module.freeze_none_bn()

            """
            target_clean = (target_shift + num_classes - 2) % num_classes
            s = len(target_clean)
            target_confusion = torch.randint(high=num_classes, size=(s,)).cuda()
            for i in range(s):
                if target_confusion[i] == target_clean[i]:
                    # make sure the confusion set is never correctly labeled
                    target_confusion[i] = (target_confusion[i] + 1) % num_classes"""

            target_clean = (target_shift + num_classes - 2) % num_classes
            term = batch_idx // 20
            if (term + rounder) % num_classes == 0:
                rounder += 1
            target_confusion = (target_clean + term + rounder) % num_classes


        else:
           target_confusion = target_shift


        if (batch_idx+1) % batch_factor == 0:

            try:
                data, target = next(distilled_set_iters)
            except Exception as e:
                distilled_set_iters = iter(distilled_set_loader)
                data, target = next(distilled_set_iters)

            data, target = data.cuda(), target.cuda()
            data_mix = torch.cat([data_shift, data], dim=0)
            target_mix = torch.cat([target_confusion, target], dim=0)
            boundary = data_shift.shape[0]

            output_mix = model(data_mix)
            loss_mix = criterion_no_reduction(output_mix, target_mix)


            loss_inspection_batch_all = loss_mix[boundary:]
            loss_confusion_batch_all = loss_mix[:boundary]
            loss_confusion_batch = loss_confusion_batch_all.mean()

            target_inspection_batch_all = target_mix[boundary:]
            inspection_batch_size = len(loss_inspection_batch_all)
            loss_inspection_batch = 0
            normalizer = 0
            for i in range(inspection_batch_size):
                gt = int(target_inspection_batch_all[i].item())
                loss_inspection_batch += (loss_inspection_batch_all[i] / freq[gt])
                normalizer += (1 / freq[gt])
            loss_inspection_batch = loss_inspection_batch / normalizer


            #weighted_loss = loss_mix.mean()
            #weighted_loss = loss_inspection_batch_all.mean()
            weighted_loss = (loss_confusion_batch * (lamb - 1) + loss_inspection_batch) / lamb
            loss_confusion_batch = loss_confusion_batch.item()
            loss_inspection_batch = loss_inspection_batch.item()

        else:
            output = model(data_shift)
            weighted_loss = loss_confusion_batch = criterion_no_reduction(output, target_confusion).mean()
            loss_confusion_batch = loss_confusion_batch.item()

        optimizer.zero_grad()
        weighted_loss.backward()
        optimizer.step()

        if (batch_idx + 1) % 400 == 0:

            print('<Round-{} : Distillation Step> Batch_idx: {}, lr: {}, lamb : {}, moment : {}, Loss: {:.6f}'.format(
                confusion_iter, batch_idx + 1, optimizer.param_groups[0]['lr'], lamb, momentum,
                weighted_loss.item()))
            print('inspection_batch_loss = %f, confusion_batch_loss = %f' %
                  (loss_inspection_batch, loss_confusion_batch))

            if args.debug_info:
                model.eval()

                if dataset_name != 'ember' and dataset_name != 'imagenet':
                    tools.test(model=model, test_loader=debug_packet['test_set_loader'], poison_test=True,
                           poison_transform=debug_packet['poison_transform'], num_classes=num_classes,
                           source_classes=debug_packet['source_classes'], all_to_all = all_to_all)
                elif dataset_name == 'imagenet':
                    tools.test_imagenet(model=model, test_loader=debug_packet['test_set_loader'],
                                        poison_transform=debug_packet['poison_transform'])
                else:
                    tools.test_ember(model=model, test_loader=debug_packet['test_set_loader'],
                                     backdoor_test_loader=debug_packet['backdoor_test_set_loader'])

    model.module.unfreeze_none_bn()

    torch.save( model.module.state_dict(),
               os.path.join(inspection_set_dir, 'classwise_confused_%d_seed=%d.pt' % (confusion_iter, args.seed)) )
    print('save : ', os.path.join(inspection_set_dir, 'classwise_confused_%d_seed=%d.pt' % (confusion_iter, args.seed)))

    return model



# restore from a certain iteration step
def distill(target_class, arch, args, params, inspection_set, n_iter, criterion_no_reduction,
            dataset_name = None, final_budget = None, class_wise = False, debug=False):

    kwargs = params['kwargs']
    inspection_set_dir = params['inspection_set_dir']
    num_classes = params['num_classes']
    num_samples = len(inspection_set)
    distillation_ratio = params['distillation_ratio']
    num_confusion_iter = len(distillation_ratio) + 1

    model = arch(num_classes=num_classes)
    if debug:
        ckpt = torch.load(os.path.join(inspection_set_dir, 'confused_%d_seed=%d.pt' % (2, args.seed)))
    else:
        ckpt = torch.load(os.path.join(inspection_set_dir, 'classwise_confused_%d_seed=%d.pt' % (n_iter, args.seed)))
    model.load_state_dict(ckpt)
    model = nn.DataParallel(model)
    model = model.cuda()
    inspection_set_loader = torch.utils.data.DataLoader(inspection_set, batch_size=params['batch_size'],
                                                            shuffle=False, worker_init_fn=tools.worker_init, **kwargs)

    """
        Collect loss values for inspected samples.
    """
    loss_array = []
    correct_instances = []
    gts = []
    model.eval()
    st = 0
    with torch.no_grad():

        for data, target in tqdm(inspection_set_loader):
            data, target = data.cuda(), target.cuda()
            output = model(data)

            if dataset_name != 'ember':
                preds = torch.argmax(output, dim=1)
            else:
                preds = (output >= 0.5).float()

            batch_loss = criterion_no_reduction(output, target)

            this_batch_size = len(target)

            for i in range(this_batch_size):
                loss_array.append(batch_loss[i].item())
                gts.append(int(target[i].item()))
                if dataset_name != 'ember':
                    if preds[i] == target[i]:
                        correct_instances.append(st + i)
                else:
                    if (target[i] == 0 and output[i] < 0.5) or (target[i] == 1 and output[i] >= 0.5):
                        correct_instances.append(st + i)

            st += this_batch_size

    loss_array = np.array(loss_array)
    sorted_indices = np.argsort(loss_array)

    top_indices_each_class = [[] for _ in range(num_classes)]
    for t in sorted_indices:
        gt = gts[t]
        top_indices_each_class[gt].append(t)




    if args.debug_info:

        print('num_correct : ', len(correct_instances))

        if args.poison_type == 'TaCT' or args.poison_type == 'adaptive_blend':
            cover_indices = torch.load(os.path.join(inspection_set_dir, 'cover_indices'))

        poison_indices = torch.load(os.path.join(inspection_set_dir, 'poison_indices'))



        cover_dist = []
        poison_dist = []

        tot = len(top_indices_each_class[target_class])

        for rk, temp_id in enumerate(top_indices_each_class[target_class]):
            if temp_id in poison_indices:
                poison_dist.append(rk)
            if args.poison_type == 'TaCT' or args.poison_type == 'adaptive_blend':
                if temp_id in cover_indices:
                    cover_dist.append(rk)
        print('poison distribution [within %d samples]: ' % tot , poison_dist)

        if args.poison_type == 'TaCT' or args.poison_type == 'adaptive_blend':
            print('cover distribution [within %d samples] : ' % tot , cover_dist)

    return top_indices_each_class
