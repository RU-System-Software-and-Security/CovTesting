# Table 3

---
## Quick Start:

1. Download the dataset and new models from google drive link:                  and put them in the 'data' and 'new_model' folder, respectively. 

2. We have put [commands](https://github.com/DNNTesting/CovTesting/blob/d462c59c1cbc00c2add20ee0eaf7a9966859788b/Table%202/test_example.sh#L32-L41) in 'test_example.sh'. Please uncomment the line comment and run 'test_example.sh' to get the corresponding result for model. The results will be stored in 'result.txt'. To understand the meaning of the numbers in 'result.txt', we can take 'the result of cifar vgg16' as an example. The numbers after 'Benign-dh' is the numbers at the 8th row and the 3rd column in Table 3. The numbers after 'DH-dh' is the numbers at the 8th row and the 4th column. The numbers after 'PGD-dh' is the numbers at the 8th row and the 5th column. The numbers after 'DH-pgd' is the numbers at the 10th row and the 4th column. As for the numbers at the 10th row and the 3rd column and the numbers at the 10th row and the 5th column in Table 3, they can be gotten from the data used to generate Figure 4. 

   (We only keep the original models and datasets we used to get the results of CIFAR VGG16 and SVHN SADL-1. But users can generate their own models and datasets to get the similar results according to the steps in the following 'General Steps' section. We will take the steps to generate the results of MNIST LeNet-1 as an example. The difference between our results and the new results is acceptable and it will not affect our conclusion.)



## General Steps:

1. Download the dataset and new models from google drive link:                  and put them in the 'data' and 'new_model' folder, respectively. 

2. Run 'fuzzing.py' to generate new training data to be added to the original training dataset. To generate T1-T10, please modify the [order_number](https://github.com/DNNTesting/CovTesting/blob/250ac8148a1532f900d4129ac24423fba3c3b1cf/Figure%202%20and%20figure%203/fuzzing.py#L336) from 0 to 9 in sequence. All results will be stored at  'fuzzing/nc_index_{}.npy'.format(order_number). (fuzzing: get DH new training dataset)

3. Run 'retrain_robustness.py' to use the new training datasets T1-T10 to retrain the model. You can change the value of [T](https://github.com/DNNTesting/CovTesting/blob/d59eaac69cfb9013221463db1c52283c2200b99e/Figure%202%20and%20figure%203/retrain_robustness.py#L381) (select from 1-10) to get the models retrained by T1-T10, respectively. The retrained models are saved at ('new_model/' + dataset +'/model_{}.h5'.format(0-9)). After every  retraining, this file will also measure the robustness of the corresponding retrained model and results are stored in 'attack_evaluate_result.txt', which can be used to generate figure 3. (refer to 'Figure 2 and figure 3.xlsx')

   (retrain_robustness:  get DH retrained model)

4. pgd_attack: PGD testing dataset

5. Run 'fuzzing_testset.py' and 'exchange_testing_dataset.py' in sequence to generate the testing dataset. When running 'fuzzing_testset.py', please modify the [order_number](https://github.com/DNNTesting/CovTesting/blob/d59eaac69cfb9013221463db1c52283c2200b99e/Figure%202%20and%20figure%203/fuzzing_testset.py#L336) from 0 to 1 in sequence. The results will be stored as 'fuzzing/nc_index_test_0.npy' and 'fuzzing/nc_index_test_1.npy'. After running 'exchange_testing_dataset.py', the new testing dataset will be generated and stored as 'x_test_new.npy' at the main folder.  (deephunter_attack: DH testing dataset)

6. all: (bash)

   (depends on the training parameters and the size of new training datasets. We provide one trained model for verification)

   

   