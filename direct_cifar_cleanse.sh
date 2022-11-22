nohup python -u ct_cleanser.py -dataset cifar10 -poison_type badnet -poison_rate 0.01 -devices 0,1 -debug_info > logs/cifar/badnet.out 2>&1 &
nohup python -u ct_cleanser.py -dataset cifar10 -poison_type blend -poison_rate 0.01 -devices 2,3 -debug_info > logs/cifar/blend.out 2>&1 &
nohup python -u ct_cleanser.py -dataset cifar10 -poison_type clean_label -poison_rate 0.005 -devices 4,5 -debug_info > logs/cifar/clean_label.out 2>&1 &
nohup python -u ct_cleanser.py -dataset cifar10 -poison_type dynamic -poison_rate 0.01 -devices 6,7 -debug_info > logs/cifar/dynamic.out 2>&1 &
nohup python -u ct_cleanser.py -dataset cifar10 -poison_type ISSBA -poison_rate 0.01 -devices 8,9 -debug_info > logs/cifar/ISSBA.out 2>&1 &


nohup python -u ct_cleanser.py -dataset cifar10 -poison_type SIG -poison_rate 0.02 -devices 0,1 -debug_info > logs/cifar/SIG.out 2>&1 &
nohup python -u ct_cleanser.py -dataset cifar10 -poison_type TaCT -poison_rate 0.02 -cover_rate 0.01 -devices 2,3 -debug_info > logs/cifar/TaCT.out 2>&1 &