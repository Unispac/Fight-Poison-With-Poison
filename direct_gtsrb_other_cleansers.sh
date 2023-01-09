
nohup python -u other_cleanser.py -dataset gtsrb -poison_type ISSBA -poison_rate 0.005 -devices 6 -cleanser SS > logs/gtsrb/SS/ISSBA.out 2>&1 &


nohup python -u other_cleanser.py -dataset gtsrb -poison_type none -devices 6 -cleanser SPECTRE > logs/gtsrb/SPECTRE/none.out 2>&1 &

nohup python -u other_cleanser.py -dataset gtsrb -poison_type SIG -poison_rate 0.02 -devices 7 -cleanser SPECTRE > logs/gtsrb/SPECTRE/SIG.out 2>&1 &

nohup python -u other_cleanser.py -dataset gtsrb -poison_type badnet -poison_rate 0.01 -devices 4 -cleanser SPECTRE > logs/gtsrb/SPECTRE/badnet.out 2>&1 &
nohup python -u other_cleanser.py -dataset gtsrb -poison_type blend -poison_rate 0.01 -devices 5 -cleanser SPECTRE > logs/gtsrb/SPECTRE/blend.out 2>&1 &
nohup python -u other_cleanser.py -dataset gtsrb -poison_type dynamic -poison_rate 0.003 -devices 6 -cleanser SPECTRE > logs/gtsrb/SPECTRE/dynamic.out 2>&1 &

nohup python -u other_cleanser.py -dataset gtsrb -poison_type TaCT -poison_rate 0.005 -cover_rate 0.005 -devices 1 -cleanser SPECTRE > logs/gtsrb/SPECTRE/TaCT.out 2>&1 &
nohup python -u other_cleanser.py -dataset gtsrb -poison_type adaptive_blend -poison_rate 0.003 -cover_rate 0.003 -devices 2 -cleanser SPECTRE -alpha 0.15 -test_alpha 0.2 > logs/gtsrb/SPECTRE/adaptive_blend.out 2>&1 &
nohup python -u other_cleanser.py -dataset gtsrb -poison_type adaptive_patch -poison_rate 0.005 -cover_rate 0.01 -devices 3 -cleanser SPECTRE > logs/gtsrb/SPECTRE/adaptive_patch.out 2>&1 &