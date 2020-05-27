: '
    parser.add_argument('-dataset', help="dataset to use", choices=['mnist', 'cifar'])
    parser.add_argument('-model', help="target model to attack", choices=['vgg16', 'resnet20', 'lenet1', 'lenet4', 'lenet5', 'svhn_model', 'svhn_first', 'svhn_second'])

'

#: 'attack'
python attack.py  -dataset mnist  -model lenet1 -attack PGD -batch_size 128
python attack.py  -dataset mnist  -model lenet4 -attack PGD -batch_size 128
python attack.py  -dataset mnist  -model lenet5 -attack PGD -batch_size 128

python attack.py  -dataset cifar  -model vgg16 -attack PGD -batch_size 128
python attack.py  -dataset cifar  -model resnet20 -attack PGD -batch_size 128

python attack.py  -dataset svhn  -model svhn_model -attack PGD -batch_size 128
python attack.py  -dataset svhn  -model svhn_first -attack PGD -batch_size 128
python attack.py  -dataset svhn  -model svhn_second -attack PGD -batch_size 128

## DH
python deephunter_attack.py  -dataset mnist  -model lenet1
python deephunter_attack.py  -dataset mnist  -model lenet4
python deephunter_attack.py  -dataset mnist  -model lenet5

python deephunter_attack.py  -dataset cifar  -model vgg16
python deephunter_attack.py  -dataset cifar  -model resnet20

python deephunter_attack.py  -dataset svhn  -model svhn_model
python deephunter_attack.py  -dataset svhn  -model svhn_first
python deephunter_attack.py  -dataset svhn  -model svhn_second

# criteria
python criteria.py  -dataset mnist  -model lenet1
python criteria.py  -dataset mnist  -model lenet4
python criteria.py  -dataset mnist  -model lenet5

python criteria.py  -dataset cifar  -model vgg16
python criteria.py  -dataset cifar  -model resnet20

python criteria.py  -dataset svhn  -model svhn_model
python criteria.py  -dataset svhn  -model svhn_first
python criteria.py  -dataset svhn  -model svhn_second



echo 'Finish!'

