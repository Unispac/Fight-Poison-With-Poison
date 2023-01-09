


# shadow02
nohup python -u train_on_poisoned_set.py -dataset gtsrb -poison_type badnet -poison_rate 0.01 -devices 0 -seed 2333 > logs/gtsrb/poison_train/2333/badnet_0.01.out 2>&1 &
nohup python -u train_on_poisoned_set.py -dataset gtsrb -poison_type blend -poison_rate 0.01 -devices 1 -seed 2333 > logs/gtsrb/poison_train/2333/blend_0.01.out 2>&1 &
nohup python -u train_on_poisoned_set.py -dataset gtsrb -poison_type SIG -poison_rate 0.02 -devices 2 -seed 2333 > logs/gtsrb/poison_train/2333/SIG_0.02.out 2>&1 &
nohup python -u train_on_poisoned_set.py -dataset gtsrb -poison_type TaCT -poison_rate 0.005 -cover_rate 0.005 -devices 2 -seed 2333 > logs/gtsrb/poison_train/2333/TaCT_0.005.out 2>&1 &
nohup python -u train_on_poisoned_set.py -dataset gtsrb -poison_type adaptive_blend -poison_rate 0.003 -cover_rate 0.003 -alpha 0.15 -test_alpha 0.2 -devices 4 -seed 2333 > logs/gtsrb/poison_train/2333/adaptive_blend.out 2>&1 &
nohup python -u train_on_poisoned_set.py -dataset gtsrb -poison_type adaptive_patch -poison_rate 0.005 -cover_rate 0.01 -devices 5 -seed 2333 > logs/gtsrb/poison_train/2333/adaptive_patch.out 2>&1 &



nohup python -u train_on_poisoned_set.py -dataset gtsrb -poison_type dynamic -poison_rate 0.003 -devices 3 -seed 2333 > logs/gtsrb/poison_train/2333/dynamic_0.003.out 2>&1 &
nohup python -u train_on_poisoned_set.py -dataset gtsrb -poison_type none -devices 7 -seed 2333 > logs/gtsrb/poison_train/2333/none.out 2>&1 &