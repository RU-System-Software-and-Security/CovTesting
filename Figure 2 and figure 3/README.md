# Figure 2 and figure 3:

## Quick Start:

1. Download the dataset from google drive link:     and put them in the 'data' folder. The fuzzing data we use to generate figure 2 and figure 3 are stored in 'fuzzing' folder. The retrained model we use to generate figure 3 are stored in 'new_model' folder. 

2. To generate the data used to get figure 2, please run 'compare_coverage.py'. You can change the value of [T](https://github.com/DNNTesting/CovTesting/blob/a7bd6da7833124796b9d7fcabce85055a097d1b1/Figure%202%20and%20figure%203/compare_coverage.py#L253) (select from 1-10) to get the coverage values for T1-T10. All of the results are stored in 'coverage_result.txt'. We have put all results in 'Figure 2 and figure 3.xlsx' and use them to draw figure 2.

3.  To generate the data used to get figure 3, please run 'robustness.py'. You can change the value of [T](https://github.com/DNNTesting/CovTesting/blob/5abea2564bb247e54caa2908248d44a956b914f7/Figure%202%20and%20figure%203/robustness.py#L381) (select from 1-10) to get the coverage values for T1-T10. All of the results are stored in 'attack_evaluate_result.txt'. We also put all results from this step in 'Figure 2 and figure 3.xlsx' and use them to draw figure 3.

   

## General Steps:

1. Run 'fuzzing.py' to generate new training data to be added to the original training dataset. To generate T1-T10, please modify the [order_number](https://github.com/DNNTesting/CovTesting/blob/250ac8148a1532f900d4129ac24423fba3c3b1cf/Figure%202%20and%20figure%203/fuzzing.py#L336) from 0 to 9 in sequence. All results will be stored at  'fuzzing/nc_index_{}.npy'.format(order_number).
2. Run 'compare_coverage.py' to test the coverage for T1-T10. You can change the value of [T](https://github.com/DNNTesting/CovTesting/blob/a7bd6da7833124796b9d7fcabce85055a097d1b1/Figure%202%20and%20figure%203/compare_coverage.py#L253) (select from 1-10) to get the coverage values for T1-T10, respectively. All of the results are stored in 'coverage_result.txt' and can be used to draw figure 2. (refer to 'Figure 2 and figure 3.xlsx')
3. Run 'fuzzing_testset.py' and 'exchange_testing_dataset.py' in sequence to generate the testing dataset. When running 'fuzzing_testset.py', please modify the [order_number](https://github.com/DNNTesting/CovTesting/blob/d59eaac69cfb9013221463db1c52283c2200b99e/Figure%202%20and%20figure%203/fuzzing_testset.py#L336) from 0 to 1 in sequence. The results will be stored as 'fuzzing/nc_index_test_0.npy' and 'fuzzing/nc_index_test_1.npy'. After running 'exchange_testing_dataset.py', the new testing dataset will be generated and stored as 'x_test_new.npy' at the main folder. 
4. Run 'retrain_robustness.py' to use the new training datasets T1-T10 to retrain the model. You can change the value of [T](https://github.com/DNNTesting/CovTesting/blob/d59eaac69cfb9013221463db1c52283c2200b99e/Figure%202%20and%20figure%203/retrain_robustness.py#L381) (select from 1-10) to get the models retrained by T1-T10, respectively. The retrained models are saved at ('new_model/' + dataset +'/model_{}.h5'.format(0-9)). After every  retraining, this file will also measure the robustness of the corresponding retrained model and results are stored in 'attack_evaluate_result.txt', which can be used to generate figure 3. (refer to 'Figure 2 and figure 3.xlsx')





