: '
    parser.add_argument('-dataset', help="dataset to use", choices=['mnist', 'cifar'])
    parser.add_argument('-model', help="target model to attack", choices=['vgg16', 'resnet20', 'lenet1', 'lenet4', 'lenet5'])
    parser.add_argument('-attack', help="attack model", choices=['CW', 'PGD'])
    parser.add_argument('-batch_size', help="attack batch size", type=int, default=32)
'

#: 'model train'
#python models.py  -d mnist -e 50 -b 128

#python attack.py  -dataset mnist  -model model_9 -attack PGD -batch_size 128
#python attack_evaluate.py  -dataset mnist  -model model_9 -attack PGD

#: 'attack'
#python attack.py  -dataset mnist  -model lenet1 -attack CW -batch_size 128
#python attack.py  -dataset mnist  -model lenet1 -attack PGD -batch_size 128
#
#python attack.py  -dataset mnist  -model lenet4 -attack CW -batch_size 128
#python attack.py  -dataset mnist  -model lenet4 -attack PGD -batch_size 128
#
#python attack.py  -dataset mnist  -model lenet5 -attack CW -batch_size 128
#python attack.py  -dataset mnist  -model lenet5 -attack PGD -batch_size 128

#python attack.py  -dataset cifar  -model vgg16 -attack CW -batch_size 128
#python attack.py  -dataset cifar  -model vgg16 -attack PGD -batch_size 128
#
#python attack.py  -dataset cifar  -model resnet20 -attack CW -batch_size 128
#python attack.py  -dataset cifar  -model resnet20 -attack PGD -batch_size 128

#python attack.py  -dataset svhn  -model svhn_model -attack CW -batch_size 128
#python attack.py  -dataset svhn  -model svhn_model -attack PGD -batch_size 128
#
#python attack.py  -dataset svhn  -model svhn_first -attack CW -batch_size 128
#python attack.py  -dataset svhn  -model svhn_first -attack PGD -batch_size 128
#
#python attack.py  -dataset svhn  -model svhn_second -attack CW -batch_size 128
#python attack.py  -dataset svhn  -model svhn_second -attack PGD -batch_size 128

#
#: 'AttackEvaluate'
#python attack_evaluate.py  -dataset mnist  -model lenet1 -attack CW
#python attack_evaluate.py  -dataset mnist  -model lenet1 -attack PGD

#python attack_evaluate.py  -dataset mnist  -model lenet4 -attack CW
#python attack_evaluate.py  -dataset mnist  -model lenet4 -attack PGD
#
#python attack_evaluate.py  -dataset mnist  -model lenet5 -attack CW
#python attack_evaluate.py  -dataset mnist  -model lenet5 -attack PGD

#python attack_evaluate_cifar.py  -dataset cifar  -model vgg16 -attack CW
#python attack_evaluate_cifar.py  -dataset cifar  -model vgg16 -attack PGD
###
#python attack_evaluate_cifar.py  -dataset cifar  -model resnet20 -attack CW
#python attack_evaluate_cifar.py  -dataset cifar  -model resnet20 -attack PGD

#python attack_evaluate.py  -dataset svhn  -model svhn_model -attack CW
#python attack_evaluate.py  -dataset svhn  -model svhn_model -attack PGD

#python attack_evaluate.py  -dataset svhn  -model svhn_first -attack CW
#python attack_evaluate.py  -dataset svhn  -model svhn_first -attack PGD
#
#python attack_evaluate.py  -dataset svhn  -model svhn_second -attack CW
#python attack_evaluate.py  -dataset svhn  -model svhn_second -attack PGD


#: 'coverage'
#python coverage.py  -dataset mnist  -model lenet1 -attack CW -layer 8
#python coverage.py  -dataset mnist  -model lenet1 -attack PGD -layer 8
#
#python coverage.py  -dataset mnist  -model lenet4 -attack CW -layer 9
#python coverage.py  -dataset mnist  -model lenet4 -attack PGD -layer 9
#
#python coverage.py  -dataset mnist  -model lenet5 -attack CW -layer 10
#python coverage.py  -dataset mnist  -model lenet5 -attack PGD -layer 10
#
#python coverage_cifar.py  -dataset cifar  -model vgg16 -attack CW  -layer 55 -start_layer 0
#python coverage_cifar.py  -dataset cifar  -model vgg16 -attack PGD -layer 55 -start_layer 0
#
#python coverage_cifar.py  -dataset cifar  -model resnet20 -attack CW -layer 70 -start_layer 1
#python coverage_cifar.py  -dataset cifar  -model resnet20 -attack PGD -layer 70 -start_layer 1
#
#python coverage.py  -dataset svhn  -model svhn_model -attack CW -layer 18
#python coverage.py  -dataset svhn  -model svhn_model -attack PGD -layer 18
#
#python coverage.py  -dataset svhn  -model svhn_first -attack CW -layer 14
#python coverage.py  -dataset svhn  -model svhn_first -attack PGD -layer 14
#
#python coverage.py  -dataset svhn  -model svhn_second -attack CW -layer 32
#python coverage.py  -dataset svhn  -model svhn_second -attack PGD -layer 32



#: 'attack of adv_models'
#python attack.py  -dataset mnist  -model adv_lenet1 -attack CW -batch_size 128
#python attack.py  -dataset mnist  -model adv_lenet1 -attack PGD -batch_size 128
#
#python attack.py  -dataset mnist  -model adv_lenet4 -attack CW -batch_size 128
#python attack.py  -dataset mnist  -model adv_lenet4 -attack PGD -batch_size 128
#
#python attack.py  -dataset mnist  -model adv_lenet5 -attack CW -batch_size 128
#python attack.py  -dataset mnist  -model adv_lenet5 -attack PGD -batch_size 128

#python attack.py  -dataset cifar  -model adv_vgg16 -attack CW -batch_size 128
#python attack.py  -dataset cifar  -model adv_vgg16 -attack PGD -batch_size 128
##
#python attack.py  -dataset cifar  -model adv_resnet20 -attack CW -batch_size 128
#python attack.py  -dataset cifar  -model adv_resnet20 -attack PGD -batch_size 128

#python attack.py  -dataset svhn  -model adv_svhn_model -attack CW -batch_size 128
#python attack.py  -dataset svhn  -model adv_svhn_model -attack PGD -batch_size 128
#
#python attack.py  -dataset svhn  -model adv_svhn_first -attack CW -batch_size 128
#python attack.py  -dataset svhn  -model adv_svhn_first -attack PGD -batch_size 128
#
#python attack.py  -dataset svhn  -model adv_svhn_second -attack CW -batch_size 128
#python attack.py  -dataset svhn  -model adv_svhn_second -attack PGD -batch_size 128


#: 'AttackEvaluate of adv_models'
#python attack_evaluate.py  -dataset mnist  -model adv_lenet1 -attack CW
#python attack_evaluate.py  -dataset mnist  -model adv_lenet1 -attack PGD
#
#python attack_evaluate.py  -dataset mnist  -model adv_lenet4 -attack CW
#python attack_evaluate.py  -dataset mnist  -model adv_lenet4 -attack PGD
#
#python attack_evaluate.py  -dataset mnist  -model adv_lenet5 -attack CW
#python attack_evaluate.py  -dataset mnist  -model adv_lenet5 -attack PGD

python attack_evaluate_cifar.py  -dataset cifar  -model adv_vgg16 -attack CW
python attack_evaluate_cifar.py  -dataset cifar  -model adv_vgg16 -attack PGD
#
python attack_evaluate_cifar.py  -dataset cifar  -model adv_resnet20 -attack CW
python attack_evaluate_cifar.py  -dataset cifar  -model adv_resnet20 -attack PGD

#python attack_evaluate.py  -dataset svhn  -model adv_svhn_model -attack CW
#python attack_evaluate.py  -dataset svhn  -model adv_svhn_model -attack PGD
#
#python attack_evaluate.py  -dataset svhn  -model adv_svhn_first -attack CW
#python attack_evaluate.py  -dataset svhn  -model adv_svhn_first -attack PGD
#
#python attack_evaluate.py  -dataset svhn  -model adv_svhn_second -attack CW
#python attack_evaluate.py  -dataset svhn  -model adv_svhn_second -attack PGD

#: 'coverage of adv_models'
#python coverage.py  -dataset mnist  -model adv_lenet1 -attack CW -layer 8
#python coverage.py  -dataset mnist  -model adv_lenet1 -attack PGD -layer 8
#
#python coverage.py  -dataset mnist  -model adv_lenet4 -attack CW -layer 9
#python coverage.py  -dataset mnist  -model adv_lenet4 -attack PGD -layer 9
#
#python coverage.py  -dataset mnist  -model adv_lenet5 -attack CW -layer 10
#python coverage.py  -dataset mnist  -model adv_lenet5 -attack PGD -layer 10

#python coverage_cifar.py  -dataset cifar  -model adv_vgg16 -attack CW  -layer 55 -start_layer 0
#python coverage_cifar.py  -dataset cifar  -model adv_vgg16 -attack PGD -layer 55 -start_layer 0
#
#python coverage_cifar.py  -dataset cifar  -model adv_resnet20 -attack CW -layer 70 -start_layer 1
#python coverage_cifar.py  -dataset cifar  -model adv_resnet20 -attack PGD -layer 70 -start_layer 1

#python coverage.py  -dataset svhn  -model adv_svhn_model -attack CW -layer 18
#python coverage.py  -dataset svhn  -model adv_svhn_model -attack PGD -layer 18
#
#python coverage.py  -dataset svhn  -model adv_svhn_first -attack CW -layer 14
#python coverage.py  -dataset svhn  -model adv_svhn_first -attack PGD -layer 14
#
#python coverage.py  -dataset svhn  -model adv_svhn_second -attack CW -layer 32
#python coverage.py  -dataset svhn  -model adv_svhn_second -attack PGD -layer 32

echo 'Finish!'

