nohup python -u other_cleanser.py -dataset cifar10 -poison_type badnet -poison_rate 0.003 -devices 0 -cleanser SPECTRE > logs/cifar/SPECTRE/badnet.out 2>&1 &

nohup python -u other_cleanser.py -dataset cifar10 -poison_type none -devices 1 -cleanser SPECTRE > logs/cifar/SPECTRE/none.out 2>&1 &
nohup python -u other_cleanser.py -dataset cifar10 -poison_type ISSBA -poison_rate 0.02 -devices 2 -cleanser SPECTRE > logs/cifar/SPECTRE/ISSBA.out 2>&1 &
nohup python -u other_cleanser.py -dataset cifar10 -poison_type SIG -poison_rate 0.02 -devices 4 -cleanser SPECTRE > logs/cifar/SPECTRE/SIG.out 2>&1 &

nohup python -u other_cleanser.py -dataset cifar10 -poison_type clean_label -poison_rate 0.003 -devices 5 -cleanser SPECTRE > logs/cifar/SPECTRE/clean_label.out 2>&1 &
nohup python -u other_cleanser.py -dataset cifar10 -poison_type dynamic -poison_rate 0.003 -devices 6 -cleanser SPECTRE > logs/cifar/SPECTRE/dynamic.out 2>&1 &
nohup python -u other_cleanser.py -dataset cifar10 -poison_type TaCT -poison_rate 0.003 -cover_rate 0.003 -devices 0 -cleanser SPECTRE > logs/cifar/SPECTRE/TaCT.out 2>&1 &

nohup python -u other_cleanser.py -dataset cifar10 -poison_type adaptive_blend -poison_rate 0.003 -cover_rate 0.003 -devices 4 -cleanser SPECTRE -alpha 0.15 -test_alpha 0.2 > logs/cifar/SPECTRE/adaptive_blend.out 2>&1 &
nohup python -u other_cleanser.py -dataset cifar10 -poison_type adaptive_patch -poison_rate 0.003 -cover_rate 0.006 -devices 6 -cleanser SPECTRE > logs/cifar/SPECTRE/adaptive_patch.out 2>&1 &
nohup python -u other_cleanser.py -dataset cifar10 -poison_type blend -poison_rate 0.003 -devices 5 -cleanser SPECTRE > logs/cifar/SPECTRE/blend.out 2>&1 &