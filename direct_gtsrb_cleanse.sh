nohup python -u ct_cleanser.py -dataset gtsrb -poison_type adaptive_blend -poison_rate 0.003 -cover_rate 0.003 -devices 4,5 -debug_info -alpha 0.15 -test_alpha 0.2 > logs/gtsrb/adaptive_blend.out 2>&1 &
nohup python -u ct_cleanser.py -dataset gtsrb -poison_type adaptive_patch -poison_rate 0.005 -cover_rate 0.01 -devices 6,7 -debug_info > logs/gtsrb/adaptive_patch.out 2>&1 &
nohup python -u ct_cleanser.py -dataset gtsrb -poison_type badnet_all_to_all -poison_rate 0.01 -devices 8,9 -debug_info > logs/gtsrb/badnet_all_to_all.out 2>&1 &
nohup python -u ct_cleanser.py -dataset gtsrb -poison_type badnet -poison_rate 0.01 -devices 8,9 -debug_info > logs/gtsrb/badnet.out 2>&1 &
nohup python -u ct_cleanser.py -dataset gtsrb -poison_type blend -poison_rate 0.01 -devices 2,3 -debug_info > logs/gtsrb/blend.out 2>&1 &

nohup python -u ct_cleanser.py -dataset gtsrb -poison_type dynamic -poison_rate 0.003 -devices 6,7 -debug_info > logs/gtsrb/dynamic_0.003.out 2>&1 &
nohup python -u ct_cleanser.py -dataset gtsrb -poison_type TaCT -poison_rate 0.005 -cover_rate 0.005 -devices 4,5 -debug_info > logs/gtsrb/TaCT_0.005.out 2>&1 &

nohup python -u ct_cleanser.py -dataset gtsrb -poison_type SIG -poison_rate 0.02 -devices 8,9 -debug_info > logs/gtsrb/SIG.out 2>&1 &

nohup python -u ct_cleanser.py -dataset gtsrb -poison_type none -devices 4,5 -debug_info > logs/gtsrb/none.out 2>&1 &