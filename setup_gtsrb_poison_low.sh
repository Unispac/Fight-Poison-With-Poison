python create_poisoned_set.py -dataset gtsrb -poison_type badnet -poison_rate 0.01
python create_poisoned_set.py -dataset gtsrb -poison_type blend -poison_rate 0.01
python create_poisoned_set.py -dataset gtsrb -poison_type dynamic -poison_rate 0.003
python create_poisoned_set.py -dataset gtsrb -poison_type SIG -poison_rate 0.02

python create_poisoned_set.py -dataset gtsrb -poison_type none
python create_poisoned_set.py -dataset gtsrb -poison_type TaCT -poison_rate 0.005 -cover_rate 0.005
python create_poisoned_set.py -dataset gtsrb -poison_type adaptive_blend -poison_rate 0.003 -cover_rate 0.003 -alpha 0.15
python create_poisoned_set.py -dataset gtsrb -poison_type adaptive_patch -poison_rate 0.005 -cover_rate 0.01
#python create_poisoned_set.py -dataset gtsrb -poison_type badnet_all_to_all -poison_rate 0.05