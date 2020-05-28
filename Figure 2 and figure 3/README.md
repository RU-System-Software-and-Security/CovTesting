# Figure 2 and figure 3:

## Quick Start:

1. Download the dataset from google drive link:     and put them in the 'data' folder. The fuzzing data we use to generate figure 2 and figure 3 are stored in 'fuzzing' folder. The retrained model we use to generate figure 3 are stored in 'new_model' folder. 
2. To generate the data used to get figure 2, please run 'compare_coverage.py'. You can change the value of [T](https://github.com/DNNTesting/CovTesting/blob/a7bd6da7833124796b9d7fcabce85055a097d1b1/Figure%202%20and%20figure%203/compare_coverage.py#L253) (choose from 1~10) to get the coverage values for T1~T10. All of the results are stored in 'coverage_result.txt'. We have put all results in 'Figure 2 and figure 3.xlsx' and use them to draw figure 2.
3.  To generate the data used to get figure 3, please run 'compare_coverage.py'. You can change the value of [T](https://github.com/DNNTesting/CovTesting/blob/a7bd6da7833124796b9d7fcabce85055a097d1b1/Figure%202%20and%20figure%203/compare_coverage.py#L253) (choose from 1~10) to get the coverage values for T1~T10. All of the results are stored in 'coverage_result.txt'. We have put all results in 'Figure 2 and figure 3.xlsx' and use them to draw figure 2.





## General Steps:

fuzzing_testset.py + exchange_testing_dataset.py: Run 'fuzzing_testset.py' firstly, and then run 'exchange_testing_dataset.py'. generate the new testing dataset. 

fuzzing.py: use DeepHunter to generate new training data to be added to the original training dataset 

order_number 0~9, results will be stored at (fuzzing/nc_index_{}.npy)

compare_coverage.py: add the generated training dataset to the original training dataset and test the coverage. (figure 2)

retrain_robustness.py: add the generated training dataset to the original training dataset and then retrain the model. The retrained model are saved at ('new_model/' + dataset +'/model_{}.h5'.format()). Then this file will measure the Robustness of the model. (figure 3)





