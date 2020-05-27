: '
    parser.add_argument('-dataset', help="dataset to use", choices=['mnist', 'cifar'])
    parser.add_argument('-model', help="target model to attack", choices=['vgg16', 'resnet20', 'lenet1', 'lenet4', 'lenet5', 'svhn_model', 'svhn_first', 'svhn_second'])

'

python all.py  -dataset mnist  -model lenet1
#python all.py  -dataset mnist  -model lenet4
#python all.py  -dataset mnist  -model lenet5
#
#python all.py  -dataset cifar  -model vgg16
#python all.py  -dataset cifar  -model resnet20
#
#python all.py  -dataset svhn  -model svhn_model
#python all.py  -dataset svhn  -model svhn_first
#python all.py  -dataset svhn  -model svhn_second



echo 'Finish!'

