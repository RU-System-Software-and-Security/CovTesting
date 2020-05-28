# Figure 2 and figure 3:

## Quick Start:

1. Download the dataset from google drive link:     and put them in the 'data' folder. The fuzzing data we use to generate figure 2 and figure 3 are stored in 'fuzzing' folder. The retrained model we use to generate figure 3 are stored in 'new_model' folder. 

2. To generate the data used to get figure 2, please run 'compare_coverage.py'. You can change the value of [T](https://github.com/DNNTesting/CovTesting/blob/a7bd6da7833124796b9d7fcabce85055a097d1b1/Figure%202%20and%20figure%203/compare_coverage.py#L253) (select from 1-10) to get the coverage values for T1-T10. All of the results are stored in 'coverage_result.txt'. We have put all results in 'Figure 2 and figure 3.xlsx' and use them to draw figure 2.

3.  To generate the data used to get figure 3, please run 'robustness.py'. You can change the value of [T](https://github.com/DNNTesting/CovTesting/blob/5abea2564bb247e54caa2908248d44a956b914f7/Figure%202%20and%20figure%203/robustness.py#L381) (select from 1-10) to get the coverage values for T1-T10. All of the results are stored in 'attack_evaluate_result.txt'. We also put all results from this step in 'Figure 2 and figure 3.xlsx' and use them to draw figure 3.

   

## General Steps:

1. Run 'fuzzing.py' to generate new training data to be added to the original training dataset. To generate T1-T10, please modify the [order_number](https://github.com/DNNTesting/CovTesting/blob/250ac8148a1532f900d4129ac24423fba3c3b1cf/Figure%202%20and%20figure%203/fuzzing.py#L336) from 0 to 9 in sequence. All results will be stored at  'fuzzing/nc_index_{}.npy'.format(order_number).
2. Run 'compare_coverage.py' to test the coverage for T1-T10. You can change the value of [T](https://github.com/DNNTesting/CovTesting/blob/a7bd6da7833124796b9d7fcabce85055a097d1b1/Figure%202%20and%20figure%203/compare_coverage.py#L253) (select from 1-10) to get the coverage values for T1-T10. All of the results are stored in 'coverage_result.txt' and can be used to draw figure 2. (see 'Figure 2 and figure 3.xlsx')
3. 





fuzzing_testset.py + exchange_testing_dataset.py: Run 'fuzzing_testset.py' firstly, and then run 'exchange_testing_dataset.py'. generate the new testing dataset. 

retrain_robustness.py: add the generated training dataset to the original training dataset and then retrain the model. The retrained model are saved at ('new_model/' + dataset +'/model_{}.h5'.format()). Then this file will measure the Robustness of the model. (figure 3)





