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

    model.eval()
    with torch.no_grad():
        for i, (ins_data, ins_target) in enumerate(tqdm(data_loader)):
            ins_data = ins_data.cuda()
            preds, x_features = model(ins_data, return_hidden=True)
            preds = torch.argmax(preds, dim=1).cpu().numpy()
            this_batch_size = len(ins_target)
            for bid in range(this_batch_size):
                feats.append(x_features[bid].cpu().numpy())
                label_list.append(ins_target[bid].cpu().numpy())
                preds_list.append(preds[bid])

    return feats, label_list, preds_list


# cluster_analysis_in_the_feature_space
def constrained_GMM(init, chunklets, X, C):

    from scipy.stats import multivariate_normal
    from scipy.special import softmax

    # init_gmm = [init_pi, init_mu, init_sigma]
    p = init[0].numpy()
    mu = init[1].numpy()
    covariance = init[2].numpy()

    L = len(chunklets)  # number of chunklets
    last_posterior = np.random.rand(L, C)
    EPS = 1e-2
    clean_chunklet_label = -1

    labels = np.zeros(L)

    for t in range(100):  # maximum iterations : 100

        posterior = np.zeros((L, C))
        mu_next = np.zeros_like(mu)
        covariance_next = np.zeros_like(covariance)

        p_next = np.zeros_like(p)
        N = 0

        for i in range(L):  # compute posterior of each chunklet
            r = np.zeros(C)
            for j in range(C):
                r[j] = np.log(p[j]) + max(multivariate_normal.logpdf(x=chunklets[i], mean=mu[j],
                                                                     cov=covariance[j], allow_singular=True).sum(),
                                          np.log(1e-8) * len(chunklets[i]))
            r = softmax(r)
            posterior[i, :] = r[:]

            p_next += r[:] * len(chunklets[i])
            N += len(chunklets[i])

        p_next /= N
        for i in range(C):
            p_next[i] = max(p_next[i], 1e-8)

        for j in range(C):  # compute mu_next
            Z = 0  # normalizer
            for i in range(L):
                Z += posterior[i, j] * len(chunklets[i])
                mu_next[j] += (chunklets[i].sum(axis=0) * posterior[i, j])
            mu_next[j] /= Z

        for j in range(C):  # compute coriance_next
            Z = 0  # normalizer
            for i in range(L):
                Z += posterior[i, j] * len(chunklets[i])
                centered_chunck = chunklets[i] - mu_next[j]
                covariance_next[j] += (centered_chunck.T @ centered_chunck) * posterior[i, j]
            covariance_next[j] /= Z

        p = p_next
        mu = mu_next
        covariance = covariance_next

        clean_chunklet_label = np.argmax(posterior[-1, :])

        if np.linalg.norm(last_posterior - posterior) < EPS: # early stop
            break
        last_posterior = posterior

    # hard label prediction for X, with the fitted model
    n = len(X)
    posterior = np.zeros((n, C))
    for i in range(n):  # compute posterior of each sample
        r = np.zeros(C)
        for j in range(C):
            r[j] = np.log(p[j]) + multivariate_normal.logpdf(x=X[i:i + 1], mean=mu[j],
                                                             cov=covariance[j], allow_singular=True).sum()
        r = softmax(r)
        posterior[i, :] = r[:]

    labels = np.argmax(posterior, axis=1)

    return p, mu, covariance, labels, clean_chunklet_label


# use the confused model to identify backdoor poison samples
def identify_poison_samples(inspection_set, clean_indices, model, num_classes):


    from scipy.stats import multivariate_normal

    kwargs = {'num_workers': 4, 'pin_memory': True}
    num_samples = len(inspection_set)

    other_indices = list(set(range(len(inspection_set))) - set(clean_indices))
    other_indices.sort()

    # main dataset we aim to cleanse
    inspection_split_loader = torch.utils.data.DataLoader(
        inspection_set,
        batch_size=128, shuffle=False, **kwargs)

    feats_inspection, class_labels_inspection, preds_inspection = get_features(inspection_split_loader, model)
    feats_inspection = np.array(feats_inspection)
    class_labels_inspection = np.array(class_labels_inspection)

    class_indices = [[] for _ in range(num_classes)]
    class_indices_in_clean_chunklet = [[] for _ in range(num_classes)]

    for i in range(num_samples):
        gt = class_labels_inspection[i]
        class_indices[gt].append(i)

    for i in clean_indices:
        gt = class_labels_inspection[i]
        class_indices_in_clean_chunklet[gt].append(i)

    for i in range(num_classes):
        class_indices[i].sort()
        class_indices_in_clean_chunklet[i].sort()

        if len(class_indices[i]) < 2:
            raise Exception('dataset is too small for class %d' % i)

        if len(class_indices_in_clean_chunklet[i]) < 2:
            raise Exception('clean chunklet is too small for class %d' % i)

    # apply cleanser, if the likelihood of two-clusters-model is twice of the likelihood of single-cluster-model
    threshold = 5.0
    suspicious_indices = []

    from sklearn.decomposition import PCA
    from sklearn.mixture import GaussianMixture

    class_likelihood_ratio = []

    for target_class in range(num_classes):

        print('num')

        num_samples_within_class = len(class_indices[target_class])

        print('class-%d : ' % target_class, num_samples_within_class)

        clean_chunklet_size = len(class_indices_in_clean_chunklet[target_class])

        clean_chunklet_indices_within_class = []
        pt = 0
        for i in range(num_samples_within_class):

            if pt == clean_chunklet_size:
                break

            if class_indices[target_class][i] < class_indices_in_clean_chunklet[target_class][pt]:
                continue
            else:
                clean_chunklet_indices_within_class.append(i)
                pt += 1

        temp_feats = torch.FloatTensor(feats_inspection[class_indices[target_class]])
        projector = PCA(n_components=8)
        projected_feats = projector.fit_transform(temp_feats)
        temp_feats = torch.FloatTensor(projected_feats)

        # whiten the data with approximated clean statistics
        clean_feats = temp_feats[clean_chunklet_indices_within_class]
        clean_covariance = torch.cov(clean_feats.T)
        temp_feats -= clean_feats.mean(dim=0)
        L, V = torch.linalg.eig(clean_covariance)
        L, V = L.real, V.real
        L = (L + 0.0001) ** (-1 / 2)
        L = torch.diag(L)
        normalizer = torch.matmul(V, torch.matmul(L, V.T))
        temp_feats = torch.matmul(normalizer, temp_feats.T).T

        # reduce dimensionality
        projector = PCA(n_components=2)
        projected_feats = projector.fit_transform(temp_feats)
        projected_feats = torch.FloatTensor(projected_feats)
        projected_feats_clean = projected_feats[clean_chunklet_indices_within_class]

        # unconstrained gmm => we use it as the init state for the latter constrained gmm
        init_mu = torch.zeros(2, 2)
        init_sigma = torch.zeros((2, 2, 2))
        init_pi = torch.zeros(2)

        init_mu[0, :] = projected_feats_clean.mean(dim=0)
        init_sigma[0, :] = torch.cov(projected_feats_clean.T)

        dis_array = torch.norm(projected_feats - init_mu[0], dim=1)
        _, ids = torch.topk(dis_array, num_samples_within_class // 10)

        psudo_outliers = projected_feats[ids]
        init_mu[1, :] = psudo_outliers.mean(dim=0)
        init_sigma[1, :] = torch.cov(psudo_outliers.T)

        init_pi[0] = 0.9
        init_pi[1] = 0.1

        init_gmm = [init_pi, init_mu, init_sigma]

        chunklets = []
        chunklets_ids_to_sample_ids = []
        num_chunklets = 0

        for i in range(num_samples_within_class):
            if i not in clean_chunklet_indices_within_class:
                chunklets.append(
                    projected_feats[i:i + 1].numpy())  # unconstrained points : each single point forms a chunklet
                chunklets_ids_to_sample_ids.append([i])
                num_chunklets += 1

        chunklets.append(
            projected_feats_clean.numpy())  # constraint : the known clean set should be in the same cluster
        chunklets_ids_to_sample_ids.append(clean_chunklet_indices_within_class)
        num_chunklets += 1

        p, mu, covariance, labels, clean_cluster = constrained_GMM(init=init_gmm, chunklets=chunklets,
                                                                   X=projected_feats, C=2)

        # likelihood ratio test
        single_cluster_likelihood = 0
        two_clusters_likelihood = 0
        for i in range(num_samples_within_class):
            single_cluster_likelihood += multivariate_normal.logpdf(x=projected_feats[i:i + 1], mean=mu[clean_cluster],
                                                                    cov=covariance[clean_cluster],
                                                                    allow_singular=True).sum()
            two_clusters_likelihood += multivariate_normal.logpdf(x=projected_feats[i:i + 1], mean=mu[labels[i]],
                                                                  cov=covariance[labels[i]], allow_singular=True).sum()

        likelihood_ratio = np.exp((two_clusters_likelihood - single_cluster_likelihood) / num_samples_within_class)

        class_likelihood_ratio.append(likelihood_ratio)

        print('likelihood_ratio = ', likelihood_ratio)

    max_ratio = np.array(class_likelihood_ratio).max()

    for target_class in range(num_classes):
        likelihood_ratio = class_likelihood_ratio[target_class]

        if likelihood_ratio == max_ratio and likelihood_ratio > 2.0:  # a lower conservative threshold for maximum ratio

            print('[class-%d] class with maximal ratio %f!. Apply Cleanser!' % (target_class, max_ratio))

            for i in class_indices[target_class]:
                if preds_inspection[i] == target_class:
                    suspicious_indices.append(i)

        elif likelihood_ratio > threshold:
            print('[class-%d] likelihood_ratio = %f > threshold = %f. Apply Cleanser!' % (
                target_class, likelihood_ratio, threshold))

            for i in class_indices[target_class]:
                if preds_inspection[i] == target_class:
                    suspicious_indices.append(i)

        else:
            print('[class-%d] likelihood_ratio = %f <= threshold = %f. Pass!' % (
            target_class, likelihood_ratio, threshold))

    return suspicious_indices


# pretraining on the poisoned datast to learn a prior of the backdoor
def pretrain(args, debug_packet, arch, num_classes, weight_decay, pretrain_epochs, distilled_set_loader, criterion,
             inspection_set_dir, confusion_iter):

    ######### Pretrain Base Model ##############
    model = arch(num_classes=num_classes)

    if confusion_iter != 0:
        ckpt_path = os.path.join(inspection_set_dir, 'base_%d_seed=%d.pt' % (confusion_iter-1, args.seed))
        model.load_state_dict( torch.load(ckpt_path) )

    model = nn.DataParallel(model)
    model = model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), 0.1,  momentum=0.9, weight_decay=weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=[30, 45], gamma=0.1)

    for epoch in range(1, pretrain_epochs + 1):  # pretrain backdoored base model with the distilled set
        model.train()

        for batch_idx, (data, target) in enumerate(distilled_set_loader):
            optimizer.zero_grad()
            data, target = data.cuda(), target.cuda()  # train set batch
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            print('<Round-{} : Pretrain> Train Epoch: {}/{} \tLoss: {:.6f}'.format(confusion_iter, epoch, pretrain_epochs, loss.item()))
            if args.debug_info:
                model.eval()
                tools.test(model=model, test_loader=debug_packet['test_set_loader'], poison_test=True,
                           poison_transform=debug_packet['poison_transform'], num_classes=num_classes,
                           source_classes=debug_packet['source_classes'])
        scheduler.step()

    base_ckpt = model.module.state_dict()
    torch.save(base_ckpt, os.path.join(inspection_set_dir, 'base_%d_seed=%d.pt' % (confusion_iter, args.seed)))
    print('save : ', os.path.join(inspection_set_dir, 'base_%d_seed=%d.pt' % (confusion_iter, args.seed)))

    return model


# confusion training : joint training on the poisoned dataset and a randomly labeled small clean set (i.e. confusion set)
def confusion_train(args, debug_packet, distilled_set_loader, clean_set_loader, confusion_iter, arch,
                    num_classes, inspection_set_dir, weight_decay, criterion_no_reduction,
                    momentum, lamb, freq, lr, batch_factor):

    ######### Distillation Step ################
    model = arch(num_classes=num_classes)
    #print('load : ', os.path.join(inspection_set_dir, 'base_%d_seed=%d.pt' % (confusion_iter, args.seed)))
    model.load_state_dict(
                torch.load(os.path.join(inspection_set_dir, 'base_%d_seed=%d.pt' % (confusion_iter, args.seed))))
    model = nn.DataParallel(model)
    model = model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay,
                                momentum=momentum)


    distilled_set_iters = iter(distilled_set_loader)
    clean_set_iters = iter(clean_set_loader)
    distillation_iters = 6000

    for batch_idx in range(distillation_iters):
        model.train()

        try:
            data_shift, target_shift = next(clean_set_iters)
        except Exception as e:
            clean_set_iters = iter(clean_set_loader)
            data_shift, target_shift = next(clean_set_iters)
        data_shift, target_shift = data_shift.cuda(), target_shift.cuda()
        target_clean = (target_shift + num_classes - 1) % num_classes

        s = len(target_clean)
        target_confusion = torch.randint(high=num_classes, size=(s,)).cuda()
        for i in range(s):
            if target_confusion[i] == target_clean[i]:
                # make sure the confusion set is never correctly labeled
                target_confusion[i] = (target_confusion[i] + 1) % num_classes

        if batch_idx % batch_factor == 0:

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

            loss_confusion_batch_all = loss_mix[:boundary]
            loss_confusion_batch = loss_confusion_batch_all.mean()

            loss_inspection_batch_all = loss_mix[boundary:]
            target_inspection_batch_all = target_mix[boundary:]
            inspection_batch_size = len(loss_inspection_batch_all)
            loss_inspection_batch = 0
            normalizer = 0
            for i in range(inspection_batch_size):
                gt = target_inspection_batch_all[i].item()
                loss_inspection_batch += (loss_inspection_batch_all[i] / freq[gt])
                normalizer += (1 / freq[gt])
            loss_inspection_batch = loss_inspection_batch / normalizer

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

        if (batch_idx + 1) % 1000 == 0:
            print('<Round-{} : Distillation Step> Batch_idx: {}, lr: {}, lamb : {}, moment : {}, Loss: {:.6f}'.format(
                confusion_iter, batch_idx + 1, optimizer.param_groups[0]['lr'], lamb, momentum,
                weighted_loss.item()))
            print('inspection_batch_loss = %f, confusion_batch_loss = %f' %
                  (loss_inspection_batch, loss_confusion_batch))

            if args.debug_info:
                model.eval()
                tools.test(model=model, test_loader=debug_packet['test_set_loader'],
                           poison_test=True,
                           poison_transform=debug_packet['poison_transform'],
                           num_classes=num_classes,
                           source_classes=debug_packet['source_classes'])

    torch.save( model.module.state_dict(),
               os.path.join(inspection_set_dir, 'confused_%d_seed=%d.pt' % (confusion_iter, args.seed)) )
    print('save : ', os.path.join(inspection_set_dir, 'confused_%d_seed=%d.pt' % (confusion_iter, args.seed)))

    return model


# restore from a certain iteration step
def distill(args, params, inspection_set, n_iter, criterion_no_reduction):

    kwargs = params['kwargs']
    inspection_set_dir = params['inspection_set_dir']
    num_classes = params['num_classes']
    num_samples = len(inspection_set)
    arch = params['arch']
    distillation_ratio = params['distillation_ratio']
    num_confusion_iter = len(distillation_ratio) + 1

    model = arch(num_classes=num_classes)
    ckpt = torch.load(os.path.join(inspection_set_dir, 'confused_%d_seed=%d.pt' % (n_iter, args.seed)))
    model.load_state_dict(ckpt)
    model = nn.DataParallel(model)
    model = model.cuda()
    inspection_set_loader = torch.utils.data.DataLoader(inspection_set, batch_size=params['batch_size'],
                                                            shuffle=False, **kwargs)

    """
        Collect loss values for inspected samples.
    """
    loss_array = []
    correct_instances = []
    model.eval()
    st = 0
    with torch.no_grad():

        for data, target in inspection_set_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            preds = torch.argmax(output, dim=1)
            batch_loss = criterion_no_reduction(output, target)

            this_batch_size = len(target)

            for i in range(this_batch_size):
                loss_array.append(batch_loss[i].item())
                if preds[i] == target[i]:
                    correct_instances.append(st + i)

            st += this_batch_size

    loss_array = np.array(loss_array)
    sorted_indices = np.argsort(loss_array)

    """
        Distill samples with low loss values from the inspected set.
    """

    if n_iter < num_confusion_iter - 1:

        if distillation_ratio[n_iter] is None:
            distilled_samples_indices = head = correct_instances
        else:
            num_expected = int(distillation_ratio[n_iter] * num_samples)
            head = sorted_indices[:num_expected]
            head = list(head)
            distilled_samples_indices = head

        median_sample_indices = None

    else:

        distilled_samples_indices = head = correct_instances

        median_sample_rate = params['median_sample_rate']

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

    """
        Report statistics of the distillation results.
    """
    if args.debug_info:

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

        num_poison = len(poison_indices)
        num_collected = len(correct_instances)
        pt = 0

        recall = 0
        for idx in correct_instances:
            while (idx > poison_indices[pt] and pt + 1 < num_poison): pt += 1
            if poison_indices[pt] == idx:
                recall += 1

        fpr = num_collected - recall
        print('recall = %d/%d = %f, fpr = %d/%d = %f' % (recall, num_poison, recall / num_poison,
                                                             fpr, num_samples - num_poison,
                                                             fpr / (num_samples - num_poison)))

    return distilled_samples_indices, median_sample_indices