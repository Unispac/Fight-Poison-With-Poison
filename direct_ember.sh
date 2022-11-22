nohup python -u ct_cleanser_ember.py -ember_options unconstrained -debug_info -devices 9 > logs/ember/unconstrained.out  2>&1 &
nohup python -u ct_cleanser_ember.py -ember_options constrained -debug_info -devices 8 > logs/ember/constrained.out  2>&1 &
nohup python -u ct_cleanser_ember.py -ember_options none -debug_info -devices 7 > logs/ember/none.out  2>&1 &