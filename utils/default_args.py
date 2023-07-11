'''default arguments for data poisoning setup
'''
parser_choices = {
    'dataset': ['gtsrb', 'cifar10', 'ember', 'imagenet'],
    'poison_type': ['badnet', 'blend', 'dynamic', 'clean_label', 'TaCT', 'SIG', 'WaNet', 'ISSBA',
                    'adaptive_blend', 'adaptive_patch', 'none', 'trojan'],
    'poison_rate': [i / 1000.0 for i in range(0, 500)],
    'cover_rate': [i / 1000.0 for i in range(0, 500)],
}

parser_default = {
    'dataset': 'cifar10',
    'poison_type': 'badnet',
    'poison_rate': 0,
    'cover_rate': 0,
    'alpha': 0.2,
}

seed = 2333 # 999, 999, 666 (1234, 5555, 777)