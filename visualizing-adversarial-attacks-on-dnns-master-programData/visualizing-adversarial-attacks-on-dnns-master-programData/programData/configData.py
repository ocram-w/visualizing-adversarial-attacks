from .__init__ import *

CONFIG_DATA={
    'WRAPPERS':{"CIFAR_10":Cifar10Wrapper, "CIFAR_100":Cifar100Wrapper, "SVHN":SVHNWrapper, "CelebA":CelebAWrapper,
                    "TinyImageNet":TinyImageNetWrapper,"ImageNet":ImageNetWrapper,"RestrictedImageNet":RestrictedImageNetWrapper},
    
    'DATASETS':{'CIFAR_10':Cifar10, 'CIFAR_100':Cifar100, 'SVHN':SVHN},
    
    'ATTACKS':{'FGM':FGM,'PGD':PGD,'MonotonePGD':MonotonePGD,'ArgminPGD':ArgminPGD,'DummyAttack':DummyAttack},
    
    'NOISE_GENERATORS':{'None':None,'Uniform':UniformNoiseGenerator(),'Normal':NormalNoiseGenerator()},
    
    'ARCHITECTURES':{'ResNet18':ResNet18, 'ResNet34':ResNet34, 'ResNet50':ResNet50, 'ResNet101':ResNet101,
                     'ResNet152':ResNet152},
    
    'MAX_INFO_BEFORE_LOGGER_CLEAROUT':20,
    
}