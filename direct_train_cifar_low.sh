# shadow01

nohup python -u train_on_poisoned_set.py -dataset cifar10 -poison_type badnet -poison_rate 0.003 -devices 0 -seed 2333 > logs/cifar/poison_train/2333/badnet_0.003.out 2>&1 &
nohup python -u train_on_poisoned_set.py -dataset cifar10 -poison_type dynamic -poison_rate 0.003 -devices 1 -seed 2333 > logs/cifar/poison_train/2333/dynamic_0.003.out 2>&1 &
nohup python -u train_on_poisoned_set.py -dataset cifar10 -poison_type SIG -poison_rate 0.02 -devices 2 -seed 2333 > logs/cifar/poison_train/2333/SIG.out 2>&1 &
nohup python -u train_on_poisoned_set.py -dataset cifar10 -poison_type none -devices 3 -seed 2333 > logs/cifar/poison_train/2333/none.out 2>&1 &
nohup python -u train_on_poisoned_set.py -dataset cifar10 -poison_type ISSBA -poison_rate 0.02 -devices 4 -seed 2333 > logs/cifar/poison_train/2333/ISSBA_0.02.out 2>&1 &
nohup python -u train_on_poisoned_set.py -dataset cifar10 -poison_type clean_label -poison_rate 0.003 -devices 5 -seed 2333 > logs/cifar/poison_train/2333/clean_label_0.003.out 2>&1 &
#nohup python -u train_on_poisoned_set.py -dataset cifar10 -poison_type badnet_all_to_all -poison_rate 0.05 -devices 9 -seed 2333 > logs/cifar/poison_train/2333/badnet_all_to_all_0.05.out 2>&1 &


# shadow04
nohup python -u train_on_poisoned_set.py -dataset cifar10 -poison_type TaCT -poison_rate 0.003 -cover_rate 0.003 -devices 2 -seed 2333 > logs/cifar/poison_train/2333/TaCT_0.003.out 2>&1 &
nohup python -u train_on_poisoned_set.py -dataset cifar10 -poison_type adaptive_blend -poison_rate 0.003 -cover_rate 0.003 -alpha 0.15 -test_alpha 0.2 -devices 3 -seed 2333 > logs/cifar/poison_train/2333/adaptive_blend.out 2>&1 &
nohup python -u train_on_poisoned_set.py -dataset cifar10 -poison_type adaptive_patch -poison_rate 0.003 -cover_rate 0.006 -devices 4 -seed 2333 > logs/cifar/poison_train/2333/adaptive_patch.out 2>&1 &
nohup python -u train_on_poisoned_set.py -dataset cifar10 -poison_type blend -poison_rate 0.003 -devices 5 -seed 2333 > logs/cifar/poison_train/2333/blend_0.003.out 2>&1 &
#nohup python -u train_on_poisoned_set.py -dataset cifar10 -poison_type badnet_all_to_all -poison_rate 0.05 -devices 9 -seed 2333 > logs/cifar/poison_train/2333/badnet_all_to_all_0.05.out 2>&1 &