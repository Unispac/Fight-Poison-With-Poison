<h1 align='center' style="text-align:center; font-weight:bold; font-size:2.0em;letter-spacing:2.0px;"> Towards A Proactive ML Approach for Detecting Backdoor Poison Samples </h1>
<p align='center' style="text-align:center;font-size:1.25em;">
    <a href="https://unispac.github.io/" target="_blank" style="text-decoration: none;">Xiangyu Qi</a>&nbsp;,&nbsp;
    <a href="http://vtu.life/" target="_blank" style="text-decoration: none;">Tinghao Xie</a>&nbsp;,&nbsp;
    <a href="https://tianhaowang.netlify.app/" target="_blank" style="text-decoration: none;">Jiachen T. Wang</a>&nbsp;,&nbsp;
    <a href="https://tongwu2020.github.io/tongwu/" target="_blank" style="text-decoration: none;">Tong Wu</a><br>
    <a href="https://scholar.google.com/citations?user=kW-hl3YAAAAJ&hl=en" target="_blank" style="text-decoration: none;">Saeed Mahloujifar</a>&nbsp;,&nbsp;
    <a href="https://www.princeton.edu/~pmittal/" target="_blank" style="text-decoration: none;">Prateek Mittal</a>&nbsp;&nbsp; 
    <br/> 
Princeton University<br/> 
</p>

<p align='center';>
<b>
<em>USENIX Security 2023</em> <br>
</b>
</p>

<p align='center' style="text-align:center;font-size:2.5 em;">
<b>
    <a href="https://arxiv.org/abs/2205.13616" target="_blank" style="text-decoration: none;">arXiv</a>&nbsp;
</b>
</p>
----------------------------------------------------------------------



Official repostory for (USENIX 2023) [Towards A Proactive ML Approach for Detecting Backdoor Poison Samples](https://arxiv.org/abs/2205.13616).

![](assets/overview.png)

Adversaries can embed backdoors in deep learning models by introducing backdoor poison samples into training datasets. In this work, we investigate how to detect such poison samples to mitigate the threat of backdoor attacks. 
1. First, we uncover a **post-hoc workflow** underlying most prior work, where defenders **passively** allow the attack to proceed and then leverage the characteristics of the post-attacked model to uncover poison samples. We reveal that this workflow does not fully exploit defenders’ capabilities, and defense pipelines built on it are **prone to failure or performance degradation** in many scenarios.
2. Second, we suggest a paradigm shift by promoting a **proactive mindset** in which defenders engage proactively with the entire model training and poison detection pipeline, directly enforcing and magnifying distinctive characteristics of the post-attacked model to facilitate poison detection. Based on this, we formulate a unified framework and provide practical insights on designing detection pipelines that are more robust and generalizable. 
3. Third, we introduce the technique of **Confusion Training (CT)** as a concrete instantiation of our framework. CT applies an additional poisoning attack to the already poisoned dataset, actively **decoupling benign correlation while exposing backdoor patterns to detection**. Empirical evaluations on 4 datasets and 14 types of attacks validate the superiority of CT over 14 baseline defenses.

---

> This is a brief introduction to get you start with our code. Refer to [misc/reproduce.md](misc/reproduce.md) for more details to reproduce our major results.



## Hardware

Our artifact is compatible with common hardware settings, only specifically requiring NVIDIA GPU support. We recommend a computing node equipped with Intel CPU (≥32 cores) and ≥2 Nvidia A100 GPUs.



## Dependency

Our experiments are conducted with PyTorch 1.12.1, and should be compatible with PyTorch of newer versions. To reproduce our defense, first manually install PyTorch with CUDA, and then install other packages via `pip install -r requirement.txt`.



## TODO before You Start

- Original CIFAR10 and GTSRB datasets would be automatically downloaded. ImageNet should be separated downloaded from [Kaggle](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data) or other available sources, while Ember can be downloaded from [here](https://github.com/elastic/ember).
- Before any experiments, first initialize the clean reserved data and validation data using command `python create_clean_set.py -dataset=$DATASET -clean_budget $N`, where `$DATASET = cifar10, gtsrb, ember, imagenet`, `$N = 2000` for `cifar10, gtsrb`, `$N = 5000` for `ember, imagenet`.
- Before launching `clean_label` attack, run [data/cifar10/clean_label/setup.sh](data/cifar10/clean_label/setup.sh).
- Before launching `dynamic` attack, download pretrained generators `all2one_cifar10_ckpt.pth.tar` and `all2one_gtsrb_ckpt.pth.tar` to [models/](models/) from https://drive.google.com/file/d/1vG44QYPkJjlOvPs7GpCL2MU8iJfOi0ei/view?usp=sharing and https://drive.google.com/file/d/1x01TDPwvSyMlCMDFd8nG05bHeh1jlSyx/view?usp=sharing.
- `SPECTRE` baseline defense is implemented in Julia. To compare our defense with `SPECTRE`, you must install Julia and install dependencies before running SPECTRE, see [other_cleansers/spectre/README.md](other_cleansers/spectre/README.md) for configuration details.
- `Frequency` baseline defense is based on Tensorflow. If you would like to reproduce their results, please install Tensorflow (code is tested with Tensorflow 2.8.1 and should be compatible with newer versions) manually, after installing all the dependencies upon.



## A Gentle Start on CIFAR10

To help readers get to know the overall pipeline of our artifact, we first illustrate an example by showing how to launch and defend against BadNet attack on CIFAR10 (corresponding to BadNet lines in Table 1 and Table 2 of the paper).

> All our scripts adopt command-line options using `argparse`. 

### **Step 1**: Create a poisoned training set.
```bash
python create_poisoned_set.py -dataset=cifar10 -poison_type=badnet -poison_rate=0.003
```

### **Step 2**: Train a backdoored model on this poisoned training set.
```bash
python train_on_poisoned_set.py -dataset=cifar10 -poison_type=badnet -poison_rate=0.003
```
The model checkpoint will be automatically saved to [poisoned_train_set/cifar10/badnet_0.003_poison_seed=0/full_base_aug_seed=2333.pt](poisoned_train_set/cifar10/badnet_0.003_poison_seed=0/full_base_aug_seed=2333.pt).

After training, you may evaluate the trained model's performance (ACC & ASR) via:
```bash
python test_model.py -dataset=cifar10 -poison_type=badnet -poison_rate=0.003
```

You may also visualize the latent space of the backdoor model (like Fig 2) w.r.t. clean and poison samples via:

```bash
python visualize.py -method=tsne -dataset=cifar10 -poison_type=badnet -poison_rate=0.003
```

### **Step 3**: Defend against the BadNet attack.

To launch our **confusion training** defense, run script:
```bash
# Cleanse the poisoned training set (results in Table 1)
python ct_cleanser.py -dataset=cifar10 -poison_type=badnet -poison_rate=0.003 -devices=0,1 -debug_info

# Retrain a benign model on the cleansed training set (results in Table 2)
python train_on_cleansed_set.py -cleanser=CT -dataset=cifar10 -poison_type=badnet -poison_rate=0.003
```

To launch baseline defenses (poison set cleanser), run script:
```bash
# Cleanse the poisoned training set (results in Table 1)
python other_cleanser.py -cleanser=$CLEANSER -dataset=cifar10 -poison_type=badnet -poison_rate=0.003 # $CLEANSER = ['SCAn', 'AC', 'SS', 'Strip', 'SPECTRE', 'SentiNet', 'Frequency']

# Retrain a benign model on the cleansed training set (results in Table 2)
python train_on_cleansed_set.py -cleanser=$CLEANSER -dataset=cifar10 -poison_type=badnet -poison_rate=0.003
```

And to launch other baseline defenses (not poison set cleanser), run script:
```bash
# (results in Table 2)
python other_defense.py -defense=$DEFENSE -dataset=cifar10 -poison_type=badnet -poison_rate=0.003 # $DEFENSE = ['ABL', 'NC', 'NAD', 'FP']
```

### Defending Other Attacks That We Implement

Replace `-poison_type=badnet` with `-poison_type=$attack`, where `$attack=['badnet', 'blend', 'dynamic', 'clean_label', 'TaCT', 'SIG', 'WaNet', 'ISSBA', 'adaptive_blend', 'adaptive_patch', 'none', 'trojan']` can be any one of the 11 attacks in our main tables (Table-1, Table-2). We can also vary `-poison_rate=$rate` to test attacks with different poison rates.

### Experiments on GTSRB

Replace `-dataset cifar10` with `-dataset gtsrb`



## Experiments on ImageNet and Ember

### ImageNet

> On Imagenet, we use seperate scripts to manage the poisoned dataset creation and confusion training pipeline.

An example on Imagenet:

```bash
python create_clean_set.py -dataset imagenet -clean_budget 5000 # reserved clean set for CT
python create_poisoned_set_imagenet.py -poison_type badnet -poison_rate 0.01 # a seperate script for creating poisoned dataset
python train_on_poisoned_set.py -dataset=imagenet -poison_type=badnet -poison_rate=0.01
python ct_cleanser_imagenet.py -poison_type=badnet -poison_rate=0.01 -devices=0,1 -debug_info # a seperate script for managing confusion training
python train_on_cleansed_set.py -cleanser=CT -dataset=imagenet -poison_type=badnet -poison_rate=0.01
```

### Ember

> On Ember, we use the original code from https://github.com/ClonedOne/MalwareBackdoors to generate poisoned dataset.

We consider "constrained" and "unconstrained" versions of the attack. The poison rate is 1% for both attacks. For the constrainted attack, the trigger watermark size is 17, with attack strategy "LargeAbsSHAP x MinPopulation"; for the unconstrained attack, the trigger watermark size is 32, with attack strategy "Combined Feature Value Selector".

After the generation of the poisoned dataset, the constrained and unconstrained versions of the should be placed at `./poisoned_train_set/ember/$type` where `$type = ['constrained', 'unconstrained', 'none']`. Particularly, 'none' corresponds to the clean dataset without attack. For ease of usage, we also upload the poisoned dataset we generated [here]().

Example: Run Confusion Training against Ember Unconstrained Attack:

```bash
python create_clean_set.py -dataset ember -clean_budget 5000 # reserved clean set for Ember
python train_on_poisoned_set.py -dataset=ember -ember_options=unconstrained
python ct_cleanser_ember.py -ember_options=unconstrained -debug_info # a seperate script for managing confusion training
python train_on_cleansed_set.py -cleanser=CT -dataset=ember -ember_options=unconstrained
```
