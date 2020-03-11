# CovTesting 

A tool for adversarial training, model attack and test. 

---

## Dependencies and Acknowledgement:

1. [clevenhans](https://github.com/tensorflow/cleverhans)
2. [adversarial-robustness-toolbox](https://github.com/IBM/adversarial-robustness-toolbox)
3. tensorflow, keras  

---

## Quick Start

- Step 0: dependencies.

We use [clevenhans](https://github.com/tensorflow/cleverhans) to construct attacks, and use [adversarial-robustness-toolbox](https://github.com/IBM/adversarial-robustness-toolbox) to do adversarial training. You can install the required dependencies according to the [requirements.txt](https://github.com/Jethro85/AETesting/blob/master/requirements.txt) file. What's more, the models and the datasets we use during our experiments are shared through: https://drive.google.com/drive/folders/14BeWcJrmcq1sCcBUMejn24-_hAQljbQ4?usp=sharing. Please download them and put them under the 'data' folder. 

- Step 1: Attack the models 

```$ python attack.py  -dataset mnist  -model lenet1 -attack CW -batch_size 128```

For `-dataset` , you can select from 'mnist' or 'cifar'. 

For `-model` , if the dataset is MNIST, you can select from 'lenet1', 'lenet4' and 'lenet5'; if the dataset is 'CIFAR-10', you can select from 'vgg16' and 'resnet20'. 

For `-attack` , you can select from 'CW' and 'PGD'.

For `-batch_size` , you can use any number. The default value of batch size is 2048.

You can modify the 'PGD' attack parameters [here](https://github.com/Jethro85/AETesting/blob/5ddb012a7efa59fd76419c0d908e6c6d80811bcc/attack.py#L162-L178) and 'CW' attack parameters [here](https://github.com/Jethro85/AETesting/blob/5ddb012a7efa59fd76419c0d908e6c6d80811bcc/attack.py#L125-L133). 

For more attack examples, you can find [here](https://github.com/Jethro85/AETesting/blob/5ddb012a7efa59fd76419c0d908e6c6d80811bcc/test_example.sh#L11-L25). 

The output results (adv test datasets) will be stored in folder 'data'.

- Step 2: Evaluate the attacks

`$ python attack_evaluate.py  -dataset mnist  -model lenet1 -attack CW`

For the choices of `-dataset` `-model` and `-attack`, you can refer to Step 1. For more attack evaluation examples, you can find [here](https://github.com/Jethro85/AETesting/blob/5ddb012a7efa59fd76419c0d908e6c6d80811bcc/test_example.sh#L27-L41). The evaluate results will be stored in "attack_evaluate_result.txt" file. 

- Step 3: Calculate the Neuron Coverage of the model

`$ python coverage.py  -dataset mnist  -model lenet1 -attack CW -layer 4`

For the choices of `-dataset` `-model` and `-attack`, you can refer to Step 1. For `-layer`, you can select any layer. For more Coverage calculation examples, you can find [here](https://github.com/Jethro85/AETesting/blob/5ddb012a7efa59fd76419c0d908e6c6d80811bcc/test_example.sh#L43-L57). The Neuron Coverage results will be stored in "coverage_result.txt" file. 

- Step 4: Adversarial training the model

`$ python adv_train_example.py`

You can modify the adversarial training parameters [here](https://github.com/Jethro85/AETesting/blob/5ddb012a7efa59fd76419c0d908e6c6d80811bcc/adv_train_example.py#L56). The adv trained model will be stored in 'data' folder. 

- Step 5: Evaluate the attacks of the adv models

`$ python attack_evaluate.py  -dataset mnist  -model adv_lenet1 -attack CW`

The choices of `-dataset`  and `-attack`  are just like them in Step 2. But for `-model` , if the dataset is MNIST, you should select from 'adv_lenet1', 'adv_lenet4' and 'adv_lenet5'; if the dataset is 'CIFAR-10', you should select from 'adv_vgg16' and 'adv_resnet20'. For more attack evaluation examples, you can find [here](https://github.com/Jethro85/AETesting/blob/5ddb012a7efa59fd76419c0d908e6c6d80811bcc/test_example.sh#L76-L90). The results will also be stored in "attack_evaluate_result.txt" file. 

- Step 6: Calculate the Neuron Coverage of the adv model

`$ python coverage.py  -dataset mnist  -model adv_lenet1 -attack CW -layer 4`

The choices of `-dataset`, `-attack`  and `-model` are just like those in Step 5. For `-layer`, you can select any layer. For more Coverage calculation examples of adv models, you can find [here](https://github.com/Jethro85/AETesting/blob/5ddb012a7efa59fd76419c0d908e6c6d80811bcc/test_example.sh#L92-L106). 





