# Coverage and Robustness VS. Training Datasets (Figure 2 and figure 3)

## Quick Start Based on Our Data:

1. Download the dataset from google drive link: https://drive.google.com/drive/folders/1WXqnuBT0FISMyYuYGShbjGwOxK7nHdGK?usp=sharing and merge them to the 'data' folder. The fuzzing data we use to generate figure 2 and figure 3 are stored in 'fuzzing' folder. The retrained model we use to generate figure 3 are stored in 'new_model' folder. 

2. Generate the data used to get figure 2:

   ```$ python compare_coverage.py```  

   You can change the value of [T](https://github.com/DNNTesting/CovTesting/blob/3c73af15df594657dbc67034496b46736c7fcf13/Coverage%20and%20Robustness%20VS.%20Training%20Datasets/compare_coverage.py#L253) (select from 1-10) to get the coverage values for T1-T10. All of the results are stored in 'coverage_result.txt'. We have put all results in 'Figure 2 and figure 3.xlsx' and use them to draw figure 2.

3. Generate the data used to get figure 3:

   ```$ python robustness.py``` 

   You can change the value of [T](https://github.com/DNNTesting/CovTesting/blob/3c73af15df594657dbc67034496b46736c7fcf13/Coverage%20and%20Robustness%20VS.%20Training%20Datasets/robustness.py#L381) (select from 1-10) to get the coverage values for T1-T10. All of the results are stored in 'attack_evaluate_result.txt'. We also put all results from this step in 'Figure 2 and figure 3.xlsx' and use them to draw figure 3.

   

## Experiment on Your Own Data:

1. Download the dataset from google drive link: https://drive.google.com/drive/folders/1WXqnuBT0FISMyYuYGShbjGwOxK7nHdGK?usp=sharing and merge them to the 'data' folder.

2. Generate new training data to be added to the original training dataset:

   ```$ python fuzzing.py```  

   To generate T1-T10, please modify the [order_number](https://github.com/DNNTesting/CovTesting/blob/fd2a5c649fb73b24826c80ee060e5a0250527e61/Coverage%20and%20Robustness%20VS.%20Training%20Datasets/fuzzing.py#L336) from 0 to 9 in sequence. All results will be stored at  'fuzzing/nc_index_{}.npy'.format(order_number).

3. Test the coverage for T1-T10:

   ```$ python compare_coverage.py``` 

   You can change the value of [T](https://github.com/DNNTesting/CovTesting/blob/fd2a5c649fb73b24826c80ee060e5a0250527e61/Coverage%20and%20Robustness%20VS.%20Training%20Datasets/compare_coverage.py#L253) (select from 1-10) to get the coverage values for T1-T10, respectively. All of the results are stored in 'coverage_result.txt' and can be used to draw figure 2. (refer to 'Figure 2 and figure 3.xlsx')

4. Generate the testing dataset:

   ```$ python fuzzing_testset.py``` 

   ```$ python exchange_testing_dataset.py``` 

   When running 'fuzzing_testset.py', please modify the [order_number](https://github.com/DNNTesting/CovTesting/blob/fd2a5c649fb73b24826c80ee060e5a0250527e61/Coverage%20and%20Robustness%20VS.%20Training%20Datasets/fuzzing_testset.py#L336) from 0 to 1 in sequence. The results will be stored as 'fuzzing/nc_index_test_0.npy' and 'fuzzing/nc_index_test_1.npy'. After running 'exchange_testing_dataset.py', the new testing dataset will be generated and stored as 'x_test_new.npy' at the main folder. 

5. Use the new training datasets T1-T10 to retrain the model:

   ```$ python retrain_robustness.py``` 

   You can change the value of [T](https://github.com/DNNTesting/CovTesting/blob/fd2a5c649fb73b24826c80ee060e5a0250527e61/Coverage%20and%20Robustness%20VS.%20Training%20Datasets/retrain_robustness.py#L381) (select from 1-10) to get the models retrained by T1-T10, respectively. The retrained models are saved at ('new_model/' + dataset +'/model_{}.h5'.format(0-9)). After every  retraining, this file will also measure the robustness of the corresponding retrained model and results are stored in 'attack_evaluate_result.txt', which can be used to generate figure 3. (refer to 'Figure 2 and figure 3.xlsx')





