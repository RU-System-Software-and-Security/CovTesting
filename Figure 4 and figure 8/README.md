# Figure 4 and figure 8

## Quick Start

We have put all data we use to generate figure 4 and figure 8 in 'correlation.xlsx'. 

1. Download the dataset from google drive link:                  and put them in the 'data' folder. 

2. Run 'correlation.py' to calculate the correlation between NC, TKNC, TKNP, KMNC, NBC, SNAC and robustness criteria. We have put the results of this step in the third sheet of 'correlation.xlsx'. 
3. Run 'correlation_sadl.py' to calculate the correlation between DSA, LSA and robustness criteria. We also put the results of this step in the third sheet of 'correlation.xlsx'. 
4. Run 'hotpicture.py' to use results from step 2 and step 3 (the third sheet of 'correlation.xlsx') to generate the figure 4. The figure will be stored as 'ALL_ALL.eps'.
5. Run 'inner_correlation.py' to calculate the correlation between coverage criteria. We have put the results of this step in the fifth sheet of 'correlation.xlsx'. 
6. Run 'inner_hotpicture.py' to use results from step 5 (the fifth sheet of 'correlation.xlsx') to generate the figure 8. The figure will be stored as 'cov_ALL_ALL.eps'.



## General Steps:

- Step 0: dependencies.

We use [clevenhans](https://github.com/tensorflow/cleverhans) to construct attacks, and use [adversarial-robustness-toolbox](https://github.com/IBM/adversarial-robustness-toolbox) to do adversarial training. You can install the required dependencies according to the [requirements.txt](https://github.com/Jethro85/AETesting/blob/master/requirements.txt) file. What's more, the models and the datasets we use during our experiments are shared through: https://drive.google.com/drive/folders/14BeWcJrmcq1sCcBUMejn24-_hAQljbQ4?usp=sharing. Please download them and put them under the 'data' folder. 

- Step 1: Attack the models 

```$ python attack.py  -dataset mnist  -model lenet1 -attack CW -batch_size 128```

For `-dataset` , you can select from 'mnist', 'cifar' or 'svhn'. 

For `-model` , if the dataset is 'mnist', you can select from 'lenet1', 'lenet4' and 'lenet5'; if the dataset is 'cifar', you can select from 'vgg16' and 'resnet20';  if the dataset is 'svhn', you can select from 'svhn_model', 'model_first' and 'model_second'. 

For `-attack` , you can select from 'CW' and 'PGD'.

For `-batch_size` , you can use any number. The default value of batch size is 2048.

You can modify the 'PGD' attack parameters [here](https://github.com/DNNTesting/CovTesting/blob/8103124d7a5a5280ad845d46d35116716ade3185/criteria/attack.py#L163-L187) and 'CW' attack parameters [here](https://github.com/DNNTesting/CovTesting/blob/8103124d7a5a5280ad845d46d35116716ade3185/criteria/attack.py#L125-L134). 

For more attack examples, you can find [here](https://github.com/DNNTesting/CovTesting/blob/e9911c8f65dadc6e55aed0fbedc4e373325c7800/criteria/test_example.sh#L15-L37). 

The output results (adv test datasets) will be stored in folder 'data'.

- Step 2: Evaluate the attacks

`$ python attack_evaluate.py  -dataset mnist  -model lenet1 -attack CW`

For the choices of `-dataset` `-model` and `-attack`, you can refer to Step 1. For more attack evaluation examples, you can find [here](https://github.com/DNNTesting/CovTesting/blob/e9911c8f65dadc6e55aed0fbedc4e373325c7800/criteria/test_example.sh#L41-L63). The evaluate results will be stored in "attack_evaluate_result.txt" file. 

- Step 3: Calculate the Neuron Coverage of the model

`$ python coverage.py  -dataset mnist  -model lenet1 -attack CW -layer 4`

For the choices of `-dataset` `-model` and `-attack`, you can refer to Step 1. For `-layer`, you can select any layer. For more Coverage calculation examples, you can find [here](https://github.com/DNNTesting/CovTesting/blob/e9911c8f65dadc6e55aed0fbedc4e373325c7800/criteria/test_example.sh#L67-L89). The Neuron Coverage results will be stored in "coverage_result.txt" file. 

- Step 4: Adversarial training the model

`$ python adv_train_example.py`

You can modify the adversarial training parameters [here](https://github.com/DNNTesting/CovTesting/blob/e9911c8f65dadc6e55aed0fbedc4e373325c7800/criteria/adv_train_example.py#L69). The adv trained model will be stored in 'data' folder. 

- Step 5: Evaluate the attacks of the adv models

`$ python attack_evaluate.py  -dataset mnist  -model adv_lenet1 -attack CW`

The choices of `-dataset`  and `-attack`  are just like them in Step 2. But for `-model` , if the dataset is 'mnist', you should select from 'adv_lenet1', 'adv_lenet4' and 'adv_lenet5'; if the dataset is 'cifar', you should select from 'adv_vgg16' and 'adv_resnet20'; if the dataset is 'svhn', you can select from 'adv_svhn_model', 'adv_model_first' and 'adv_model_second'. For more attack evaluation examples, you can find [here](https://github.com/DNNTesting/CovTesting/blob/e9911c8f65dadc6e55aed0fbedc4e373325c7800/criteria/test_example.sh#L120-L142). The results will also be stored in "attack_evaluate_result.txt" file. 

- Step 6: Calculate the Neuron Coverage of the adv model

`$ python coverage.py  -dataset mnist  -model adv_lenet1 -attack CW -layer 4`

The choices of `-dataset`, `-attack`  and `-model` are just like those in Step 5. For `-layer`, you can select any layer. For more Coverage calculation examples of adv models, you can find [here](https://github.com/DNNTesting/CovTesting/blob/e9911c8f65dadc6e55aed0fbedc4e373325c7800/criteria/test_example.sh#L144-L167). 



