

fuzzing_testset.py + exchange_testing_dataset.py: Run 'fuzzing_testset.py' firstly, and then run 'exchange_testing_dataset.py'. generate the new testing dataset. 

fuzzing.py: use DeepHunter to generate new training data to be added to the original training dataset 

order_number 0~9, results will be stored at (fuzzing/nc_index_{}.npy)

compare_coverage.py: add the generated training dataset to the original training dataset and test the coverage. (figure 2)

retrain_robustness.py: add the generated training dataset to the original training dataset and then retrain the model. The retrained model are saved at ('new_model/' + dataset +'/model_{}.h5'.format()). Then this file will measure the Robustness of the model. (figure 3)





