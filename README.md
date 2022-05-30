# Confusion Training

Official Implementation for [Fight Poison with Poison: Detecting Backdoor Poison Samples via Decoupling Benign Correlations](https://arxiv.org/abs/2205.13616)




## Attacks

See [poison_tool_box/](poison_tool_box/).

**Adaptive**
- `adaptive_blend`: adaptive attack with a single blending trigger
- `adaptive_k`: adaptive attack with `k`=4 different triggers

**Others**

- `badnet`: basic attack with badnet patch trigger
- `blend`: basic attack with a single blending trigger
- `dynamic`
- `clean_label`
- `SIG`
- `TaCT`: source specific attack





## Cleansers

**Ours**

* confusion_training.py

- run "poison_cleanser_iter.py" to launch

**Others**

See [other_cleanses/](other_cleansers/).

- `SCAn`
- `AC`: activation clustering
- `SS`: spectral signature
- `SPECTRE`
- `Strip`





## Visualization

See [visualize.py](visualize.py).

- `tsne`
- `pca`
- `oracle`





## Quick Start

To launch and defend an Adaptive-Blend attack:
```bash
# Create a clean set
python create_clean_set.py -dataset=cifar10

# Create a poisoned training set
python create_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005

# Train on the poisoned training set
python train_on_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005
python train_on_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -no_aug

# Visualize
## $METHOD = ['pca', 'tsne', 'oracle']
python visualize.py -method=$METHOD -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005

# **Cleanse with Confusion Training**
python poison_cleander_iter.py -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005

# Cleanse with other cleansers
## $CLEANSER = ['SCAn', 'AC', 'SS', 'Strip', 'SPECTRE']
python other_cleanser.py -cleanser=$CLEANSER -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005

# Retrain on cleansed set
## $CLEANSER = ['CT', 'SCAn', 'AC', 'SS', 'Strip', 'SPECTRE']
python train_on_cleansed_set.py -cleanser=$CLEANSER -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005
```

**Notice**:
- `SPECTRE` is implemented in Julia. So you must install Julia and install dependencies before running SPECTRE, see [other_cleansers/spectre/README.md](other_cleansers/spectre/README.md) for configuration details.
- For `clean_label` attack, run 'data/cifar10/clean_label/setup.sh' before the first time launching it.
- For `dynamic` attack, download pretrained generators `all2one_cifar10_ckpt.pth.tar` and `all2one_gtsrb_ckpt.pth.tar` to `[models/](models/) from https://github.com/VinAIResearch/input-aware-backdoor-attack-release before the first time launching it.


Poisoning attacks we evaluate in our papers:
```bash
# No Poison
python create_poisoned_set.py -dataset=cifar10 -poison_type=none -poison_rate=0
# BadNet
python create_poisoned_set.py -dataset=cifar10 -poison_type=badnet -poison_rate=0.01
# Blend
python create_poisoned_set.py -dataset=cifar10 -poison_type=blend -poison_rate=0.01
# Blend
python create_poisoned_set.py -dataset=cifar10 -poison_type=dynamic -poison_rate=0.01
# Clean Label
python create_poisoned_set.py -dataset=cifar10 -poison_type=clean_label -poison_rate=0.005
# SIG
python create_poisoned_set.py -dataset=cifar10 -poison_type=SIG -poison_rate=0.02
# TaCT
python create_poisoned_set.py -dataset=cifar10 -poison_type=TaCT -poison_rate=0.02 -cover_rate=0.01
# Adaptive Blend
python create_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005
# Adaptive K
python create_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_k -poison_rate=0.005 -cover_rate=0.01
```

You can also:
- specify details on the trigger (for `blend`, `clean_label`, `adaptive_blend` and `TaCT` attacks) via
    - `-alpha=$ALPHA`, the opacity of the trigger.
    - `-trigger=$TRIGGER_NAME`, where `$TRIGGER_NAME` is the name of a 32x32 trigger mark image in [triggers/](triggers). If another image named `mask_$TRIGGER_NAME` also exists in [triggers/](triggers), it will be used as the trigger mask. Otherwise by default, all black pixels of the trigger mark are not applied.
- train a vanilla model via
    ```bash
    python train_vanilla.py
    ```
- test a trained model via
    ```bash
    python test_model.py -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005
    # other options include: -no_aug, -cleanser=$CLEANSER, -model_path=$MODEL_PATH, see our code for details
    ```
- enforce a fixed running seed via `-seed=$SEED` option
- change dataset to GTSRB via `-dataset=gtsrb` option
- see more configurations in `config.py`
