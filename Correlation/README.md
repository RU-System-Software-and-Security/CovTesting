# Correlation (Figure 4 and figure 8)

## Quick Start Based on Our Data:

We have put all data we use to generate figure 4 and figure 8 in 'correlation.xlsx'. 

1. Download the dataset from google drive link: https://drive.google.com/drive/folders/1WXqnuBT0FISMyYuYGShbjGwOxK7nHdGK?usp=sharing and put them in the 'data' folder. 

2. Calculate the correlation between NC, TKNC, TKNP, KMNC, NBC, SNAC and robustness criteria:

   ```$ python correlation.py ``` 

   We have put the results of this step in the third sheet of 'correlation.xlsx'. 

3. Calculate the correlation between DSA, LSA and robustness criteria:

   ```$ python correlation_sadl.py ``` 

   We also put the results of this step in the third sheet of 'correlation.xlsx'. 

4. Use results from step 2 and step 3 (the third sheet of 'correlation.xlsx') to generate the figure 4:

   ```$ python hotpicture.py ``` 

   The figure will be stored as 'ALL_ALL.eps'.

5. Calculate the correlation between coverage criteria:

   ```$ python inner_correlation.py ``` 

   We have put the results of this step in the fifth sheet of 'correlation.xlsx'. 

6. Use results from step 5 (the fifth sheet of 'correlation.xlsx') to generate the figure 8:

   ```$ python inner_hotpicture.py ``` 

   The figure will be stored as 'cov_ALL_ALL.eps'.



## Experiment on Your Own Data:

1. Dependencies:

   The models and the datasets we use during our experiments are shared through: https://drive.google.com/drive/folders/1WXqnuBT0FISMyYuYGShbjGwOxK7nHdGK?usp=sharing. Please download them and put them under the 'data' folder. You can train your own models and attack them and test the attack and coverage according to the following steps.

   We also put commands in 'test_example.sh' script for reference.

2. Attack the models:

   Please run the following commands to attack different models:

   ```python
   python attack.py  -dataset mnist  -model lenet1 -attack CW -batch_size 128
   python attack.py  -dataset mnist  -model lenet1 -attack PGD -batch_size 128
   
   python attack.py  -dataset mnist  -model lenet4 -attack CW -batch_size 128
   python attack.py  -dataset mnist  -model lenet4 -attack PGD -batch_size 128
   
   python attack.py  -dataset mnist  -model lenet5 -attack CW -batch_size 128
   python attack.py  -dataset mnist  -model lenet5 -attack PGD -batch_size 128
   
   python attack.py  -dataset cifar  -model vgg16 -attack CW -batch_size 128
   python attack.py  -dataset cifar  -model vgg16 -attack PGD -batch_size 128
   
   python attack.py  -dataset cifar  -model resnet20 -attack CW -batch_size 128
   python attack.py  -dataset cifar  -model resnet20 -attack PGD -batch_size 128
   
   python attack.py  -dataset svhn  -model svhn_model -attack CW -batch_size 128
   python attack.py  -dataset svhn  -model svhn_model -attack PGD -batch_size 128
   
   python attack.py  -dataset svhn  -model svhn_first -attack CW -batch_size 128
   python attack.py  -dataset svhn  -model svhn_first -attack PGD -batch_size 128
   
   python attack.py  -dataset svhn  -model svhn_second -attack CW -batch_size 128
   python attack.py  -dataset svhn  -model svhn_second -attack PGD -batch_size 128
   ```

   For `-dataset` , you can select from 'mnist', 'cifar' or 'svhn'. 

   For `-model` , if the dataset is 'mnist', you can select from 'lenet1', 'lenet4' and 'lenet5'; if the dataset is 'cifar', you can select from 'vgg16' and 'resnet20';  if the dataset is 'svhn', you can select from 'svhn_model', 'model_first' and 'model_second'. 

   For `-attack` , you can select from 'CW' and 'PGD'.

   For `-batch_size` , you can use any number. The default value of batch size is 2048.

   You can modify the 'PGD' attack parameters [here](https://github.com/DNNTesting/CovTesting/blob/f1be13587df8ae74bc36a02f0c48870013691bd3/Figure%204%20and%20figure%208/attack.py#L163-L186) and 'CW' attack parameters [here](https://github.com/DNNTesting/CovTesting/blob/f1be13587df8ae74bc36a02f0c48870013691bd3/Figure%204%20and%20figure%208/attack.py#L125-L134). 

   The output results (adv test datasets) will be stored in folder 'data'.

3. Evaluate the attacks:

   Please run the following commands to evaluate different models:

   ```python
   python attack_evaluate.py  -dataset mnist  -model lenet1 -attack CW
   python attack_evaluate.py  -dataset mnist  -model lenet1 -attack PGD
   
   python attack_evaluate.py  -dataset mnist  -model lenet4 -attack CW
   python attack_evaluate.py  -dataset mnist  -model lenet4 -attack PGD
   
   python attack_evaluate.py  -dataset mnist  -model lenet5 -attack CW
   python attack_evaluate.py  -dataset mnist  -model lenet5 -attack PGD
   
   python attack_evaluate_cifar.py  -dataset cifar  -model vgg16 -attack CW
   python attack_evaluate_cifar.py  -dataset cifar  -model vgg16 -attack PGD
   
   python attack_evaluate_cifar.py  -dataset cifar  -model resnet20 -attack CW
   python attack_evaluate_cifar.py  -dataset cifar  -model resnet20 -attack PGD
   
   python attack_evaluate.py  -dataset svhn  -model svhn_model -attack CW
   python attack_evaluate.py  -dataset svhn  -model svhn_model -attack PGD
   
   python attack_evaluate.py  -dataset svhn  -model svhn_first -attack CW
   python attack_evaluate.py  -dataset svhn  -model svhn_first -attack PGD
   
   python attack_evaluate.py  -dataset svhn  -model svhn_second -attack CW
   python attack_evaluate.py  -dataset svhn  -model svhn_second -attack PGD
   ```

   For the choices of `-dataset` `-model` and `-attack`, you can refer to Step 2. The evaluate results will be stored in "attack_evaluate_result.txt" file. 

4. Calculate the Neuron Coverage of the model:

   Please run the following commands to calculate coverage of different models:

   ```python
   python coverage.py  -dataset mnist  -model lenet1 -attack CW -layer 8
   python coverage.py  -dataset mnist  -model lenet1 -attack PGD -layer 8
   
   python coverage.py  -dataset mnist  -model lenet4 -attack CW -layer 9
   python coverage.py  -dataset mnist  -model lenet4 -attack PGD -layer 9
   
   python coverage.py  -dataset mnist  -model lenet5 -attack CW -layer 10
   python coverage.py  -dataset mnist  -model lenet5 -attack PGD -layer 10
   
   python coverage_cifar.py  -dataset cifar  -model vgg16 -attack CW  -layer 55 -start_layer 0
   python coverage_cifar.py  -dataset cifar  -model vgg16 -attack PGD -layer 55 -start_layer 0
   
   python coverage_cifar.py  -dataset cifar  -model resnet20 -attack CW -layer 70 -start_layer 1
   python coverage_cifar.py  -dataset cifar  -model resnet20 -attack PGD -layer 70 -start_layer 1
   
   python coverage.py  -dataset svhn  -model svhn_model -attack CW -layer 18
   python coverage.py  -dataset svhn  -model svhn_model -attack PGD -layer 18
   
   python coverage.py  -dataset svhn  -model svhn_first -attack CW -layer 14
   python coverage.py  -dataset svhn  -model svhn_first -attack PGD -layer 14
   
   python coverage.py  -dataset svhn  -model svhn_second -attack CW -layer 32
   python coverage.py  -dataset svhn  -model svhn_second -attack PGD -layer 32
   ```

   For the choices of `-dataset` `-model` and `-attack`, you can refer to Step 2. For `-layer`, you can select any layer. The Neuron Coverage results will be stored in "coverage_result.txt" file. 


5. Adversarial training the model:

   `$ python adv_train_example.py`

   To train different models, you have to modify the [dataset and model_name](https://github.com/DNNTesting/CovTesting/blob/f1be13587df8ae74bc36a02f0c48870013691bd3/Figure%204%20and%20figure%208/adv_train_example.py#L36-L37). You can modify the adversarial training parameters [here](https://github.com/DNNTesting/CovTesting/blob/f1be13587df8ae74bc36a02f0c48870013691bd3/Figure%204%20and%20figure%208/adv_train_example.py#L69-L79). We have provided the better parameters of retraining for different datasets (see our [comments](https://github.com/DNNTesting/CovTesting/blob/f1be13587df8ae74bc36a02f0c48870013691bd3/Figure%204%20and%20figure%208/adv_train_example.py#L69-L79)). Feel free to try different parameters. The adv trained model will be [stored in 'data' folder](https://github.com/DNNTesting/CovTesting/blob/f1be13587df8ae74bc36a02f0c48870013691bd3/Figure%204%20and%20figure%208/adv_train_example.py#L82). 

6. Attack the adv models:

   Please run the following commands to attack different adv models:

   ```python
   python attack.py  -dataset mnist  -model adv_lenet1 -attack CW -batch_size 128
   python attack.py  -dataset mnist  -model adv_lenet1 -attack PGD -batch_size 128
   
   python attack.py  -dataset mnist  -model adv_lenet4 -attack CW -batch_size 128
   python attack.py  -dataset mnist  -model adv_lenet4 -attack PGD -batch_size 128
   
   python attack.py  -dataset mnist  -model adv_lenet5 -attack CW -batch_size 128
   python attack.py  -dataset mnist  -model adv_lenet5 -attack PGD -batch_size 128
   
   python attack.py  -dataset cifar  -model adv_vgg16 -attack CW -batch_size 128
   python attack.py  -dataset cifar  -model adv_vgg16 -attack PGD -batch_size 128
   
   python attack.py  -dataset cifar  -model adv_resnet20 -attack CW -batch_size 128
   python attack.py  -dataset cifar  -model adv_resnet20 -attack PGD -batch_size 128
   
   python attack.py  -dataset svhn  -model adv_svhn_model -attack CW -batch_size 128
   python attack.py  -dataset svhn  -model adv_svhn_model -attack PGD -batch_size 128
   
   python attack.py  -dataset svhn  -model adv_svhn_first -attack CW -batch_size 128
   python attack.py  -dataset svhn  -model adv_svhn_first -attack PGD -batch_size 128
   
   python attack.py  -dataset svhn  -model adv_svhn_second -attack CW -batch_size 128
   python attack.py  -dataset svhn  -model adv_svhn_second -attack PGD -batch_size 128
   ```

   The choices of `-dataset` `-model`  `-attack` and `-batch_size` are just like those in Step 2. But for `-model` , if the dataset is 'mnist', you should select from 'adv_lenet1', 'adv_lenet4' and 'adv_lenet5'; if the dataset is 'cifar', you should select from 'adv_vgg16' and 'adv_resnet20'; if the dataset is 'svhn', you can select from 'adv_svhn_model', 'adv_model_first' and 'adv_model_second'. The output results (test datasets) will be stored in folder 'data'

7. Evaluate the attacks of the adv models:

   Please run the following commands to evaluate different adv models:

   ```python
   python attack_evaluate.py  -dataset mnist  -model adv_lenet1 -attack CW
   python attack_evaluate.py  -dataset mnist  -model adv_lenet1 -attack PGD
   
   python attack_evaluate.py  -dataset mnist  -model adv_lenet4 -attack CW
   python attack_evaluate.py  -dataset mnist  -model adv_lenet4 -attack PGD
   
   python attack_evaluate.py  -dataset mnist  -model adv_lenet5 -attack CW
   python attack_evaluate.py  -dataset mnist  -model adv_lenet5 -attack PGD
   
   python attack_evaluate_cifar.py  -dataset cifar  -model adv_vgg16 -attack CW
   python attack_evaluate_cifar.py  -dataset cifar  -model adv_vgg16 -attack PGD
   
   python attack_evaluate_cifar.py  -dataset cifar  -model adv_resnet20 -attack CW
   python attack_evaluate_cifar.py  -dataset cifar  -model adv_resnet20 -attack PGD
   
   python attack_evaluate.py  -dataset svhn  -model adv_svhn_model -attack CW
   python attack_evaluate.py  -dataset svhn  -model adv_svhn_model -attack PGD
   
   python attack_evaluate.py  -dataset svhn  -model adv_svhn_first -attack CW
   python attack_evaluate.py  -dataset svhn  -model adv_svhn_first -attack PGD
   
   python attack_evaluate.py  -dataset svhn  -model adv_svhn_second -attack CW
   python attack_evaluate.py  -dataset svhn  -model adv_svhn_second -attack PGD
   ```

   Same, for `-model` , if the dataset is 'mnist', you should select from 'adv_lenet1', 'adv_lenet4' and 'adv_lenet5'; if the dataset is 'cifar', you should select from 'adv_vgg16' and 'adv_resnet20'; if the dataset is 'svhn', you can select from 'adv_svhn_model', 'adv_model_first' and 'adv_model_second'.  The results will also be stored in "attack_evaluate_result.txt" file. 

8. Calculate the Neuron Coverage of the adv models:

   Please run the following commands to calculate coverage of different models:

   ```python
   python coverage.py  -dataset mnist  -model adv_lenet1 -attack CW -layer 8
   python coverage.py  -dataset mnist  -model adv_lenet1 -attack PGD -layer 8
   
   python coverage.py  -dataset mnist  -model adv_lenet4 -attack CW -layer 9
   python coverage.py  -dataset mnist  -model adv_lenet4 -attack PGD -layer 9
   
   python coverage.py  -dataset mnist  -model adv_lenet5 -attack CW -layer 10
   python coverage.py  -dataset mnist  -model adv_lenet5 -attack PGD -layer 10
   
   python coverage_cifar.py  -dataset cifar  -model adv_vgg16 -attack CW  -layer 55 -start_layer 0
   python coverage_cifar.py  -dataset cifar  -model adv_vgg16 -attack PGD -layer 55 -start_layer 0
   
   python coverage_cifar.py  -dataset cifar  -model adv_resnet20 -attack CW -layer 70 -start_layer 1
   python coverage_cifar.py  -dataset cifar  -model adv_resnet20 -attack PGD -layer 70 -start_layer 1
   
   python coverage.py  -dataset svhn  -model adv_svhn_model -attack CW -layer 18
   python coverage.py  -dataset svhn  -model adv_svhn_model -attack PGD -layer 18
   
   python coverage.py  -dataset svhn  -model adv_svhn_first -attack CW -layer 14
   python coverage.py  -dataset svhn  -model adv_svhn_first -attack PGD -layer 14
   
   python coverage.py  -dataset svhn  -model adv_svhn_second -attack CW -layer 32
   python coverage.py  -dataset svhn  -model adv_svhn_second -attack PGD -layer 32
   ```

   The Neuron Coverage results will also be stored in "coverage_result.txt" file. 

9. Put all data from "coverage_result.txt" and "attack_evaluate_result.txt" in 'correlation.xlsx' and repeat the step 2- step 6 in "**Quick Start Based on Our Data**" section to get figure 4 and figure 8. 

   We have put our "coverage_result.txt" and "attack_evaluate_result.txt" gotten from step 1 - step 7 in 'result_mnist', 'result_cifar' and 'result_svhn' folders. Then we input all of them into the 'correlation.xlsx'  (see the first, the second and the fourth sheets). The step 2 and step 3 in "**Quick Start Based on Our Data**" section will use the first and the second sheets to calculate the correlation between coverage and robustness criteria (we put the correlation in the third sheet) and the step 4 will draw figure 4. The step 5 in "**Quick Start Based on Our Data**" section will use the fourth sheet to calculate the inner correlation between coverage criteria (we put the inner correlation in the fifth sheet) and the step 6 will draw figure 8. 

> Tips:
>
> 1. For the measurement of LSA and DSA criteria, please refer to https://github.com/coinse/sadl. We use their original codes to test LSA and DSA of our models. 
> 2. Feel free to use your own model, adversarial train different models and use different attacks to do different experiments. We are happy to see more results from you. 
>







