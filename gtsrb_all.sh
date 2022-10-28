nohup python -u new_cleanser.py -dataset gtsrb -poison_type adaptive_blend -poison_rate 0.003 -cover_rate 0.003 -devices 0,1 -debug_info -alpha 0.15 -test_alpha 0.2 > logs/gtsrb/cleanse_adaptive_blend.out 2>&1 &
nohup python -u new_cleanser.py -dataset gtsrb -poison_type adaptive_k -poison_rate 0.003 -cover_rate 0.006 -devices 2,3 -debug_info > logs/gtsrb/cleanse_adaptive_k.out 2>&1 &
nohup python -u new_cleanser.py -dataset gtsrb -poison_type badnet -poison_rate 0.01 -devices 4,5 -debug_info > logs/gtsrb/cleanse_badnet.out 2>&1 &
nohup python -u new_cleanser.py -dataset gtsrb -poison_type blend -poison_rate 0.01 -devices 6,7 -debug_info > logs/gtsrb/cleanse_blend.out 2>&1 &
nohup python -u new_cleanser.py -dataset gtsrb -poison_type dynamic -poison_rate 0.01 -devices 8,9 -debug_info > logs/gtsrb/cleanse_dynamic.out 2>&1 &
nohup python -u new_cleanser.py -dataset gtsrb -poison_type SIG -poison_rate 0.02 -devices 8,9 -debug_info > logs/gtsrb/cleanse_SIG.out 2>&1 &
nohup python -u new_cleanser.py -dataset gtsrb -poison_type TaCT -poison_rate 0.02 -cover_rate 0.01 -devices 6,7 -debug_info > logs/gtsrb/cleanse_TaCT.out 2>&1 &