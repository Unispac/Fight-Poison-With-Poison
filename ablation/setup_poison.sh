


python create_poisoned_set.py -dataset cifar10 -poison_type adaptive_blend -poison_rate 0.001 -cover_rate 0.001 -alpha 0.15
python create_poisoned_set.py -dataset cifar10 -poison_type adaptive_blend -poison_rate 0.003 -cover_rate 0.003 -alpha 0.15
python create_poisoned_set.py -dataset cifar10 -poison_type adaptive_blend -poison_rate 0.005 -cover_rate 0.005 -alpha 0.15
python create_poisoned_set.py -dataset cifar10 -poison_type adaptive_blend -poison_rate 0.01 -cover_rate 0.01 -alpha 0.15
python create_poisoned_set.py -dataset cifar10 -poison_type adaptive_blend -poison_rate 0.05 -cover_rate 0.05 -alpha 0.15
python create_poisoned_set.py -dataset cifar10 -poison_type adaptive_blend -poison_rate 0.1 -cover_rate 0.1 -alpha 0.15



python create_poisoned_set.py -dataset cifar10 -poison_type adaptive_patch -poison_rate 0.001 -cover_rate 0.002
python create_poisoned_set.py -dataset cifar10 -poison_type adaptive_patch -poison_rate 0.003 -cover_rate 0.006
python create_poisoned_set.py -dataset cifar10 -poison_type adaptive_patch -poison_rate 0.005 -cover_rate 0.01
python create_poisoned_set.py -dataset cifar10 -poison_type adaptive_patch -poison_rate 0.01 -cover_rate 0.02
python create_poisoned_set.py -dataset cifar10 -poison_type adaptive_patch -poison_rate 0.05 -cover_rate 0.1
python create_poisoned_set.py -dataset cifar10 -poison_type adaptive_patch -poison_rate 0.1 -cover_rate 0.2



python create_poisoned_set.py -dataset cifar10 -poison_type TaCT -poison_rate 0.001 -cover_rate 0.001
python create_poisoned_set.py -dataset cifar10 -poison_type TaCT -poison_rate 0.005 -cover_rate 0.005
python create_poisoned_set.py -dataset cifar10 -poison_type TaCT -poison_rate 0.01 -cover_rate 0.01
python create_poisoned_set.py -dataset cifar10 -poison_type TaCT -poison_rate 0.05 -cover_rate 0.05
python create_poisoned_set.py -dataset cifar10 -poison_type TaCT -poison_rate 0.1 -cover_rate 0.1



python create_poisoned_set.py -dataset cifar10 -poison_type dynamic -poison_rate 0.001
python create_poisoned_set.py -dataset cifar10 -poison_type dynamic -poison_rate 0.005
python create_poisoned_set.py -dataset cifar10 -poison_type dynamic -poison_rate 0.01
python create_poisoned_set.py -dataset cifar10 -poison_type dynamic -poison_rate 0.05
python create_poisoned_set.py -dataset cifar10 -poison_type dynamic -poison_rate 0.1


python create_poisoned_set.py -dataset cifar10 -poison_type clean_label -poison_rate 0.001
python create_poisoned_set.py -dataset cifar10 -poison_type clean_label -poison_rate 0.005
python create_poisoned_set.py -dataset cifar10 -poison_type clean_label -poison_rate 0.01
python create_poisoned_set.py -dataset cifar10 -poison_type clean_label -poison_rate 0.05
python create_poisoned_set.py -dataset cifar10 -poison_type clean_label -poison_rate 0.1


python create_poisoned_set.py -dataset cifar10 -poison_type badnet -poison_rate 0.001
python create_poisoned_set.py -dataset cifar10 -poison_type badnet -poison_rate 0.005
python create_poisoned_set.py -dataset cifar10 -poison_type badnet -poison_rate 0.01
python create_poisoned_set.py -dataset cifar10 -poison_type badnet -poison_rate 0.05
python create_poisoned_set.py -dataset cifar10 -poison_type badnet -poison_rate 0.1


python create_poisoned_set.py -dataset cifar10 -poison_type blend -poison_rate 0.001
python create_poisoned_set.py -dataset cifar10 -poison_type blend -poison_rate 0.005
python create_poisoned_set.py -dataset cifar10 -poison_type blend -poison_rate 0.01
python create_poisoned_set.py -dataset cifar10 -poison_type blend -poison_rate 0.05
python create_poisoned_set.py -dataset cifar10 -poison_type blend -poison_rate 0.1