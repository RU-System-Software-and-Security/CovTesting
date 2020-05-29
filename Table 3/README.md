# Table 3

---
## Quick Start:

1. Download the dataset and new models from google drive link:                  and put them in the 'data' and 'new_model' folder, respectively. 

2. We have put [commands](https://github.com/DNNTesting/CovTesting/blob/d462c59c1cbc00c2add20ee0eaf7a9966859788b/Table%202/test_example.sh#L32-L41) in 'test_example.sh'. Please uncomment the line comment and run 'test_example.sh' to get the corresponding result for model. The results will be stored in 'result.txt'. To understand the meaning of the numbers in 'result.txt', we can take 'the result of cifar vgg16' as an example. The numbers after 'Benign-dh' is the numbers at the 8th row and the 3rd column in Table 3. The numbers after 'DH-dh' is the numbers at the 8th row and the 4th column. The numbers after 'PGD-dh' is the numbers at the 8th row and the 5th column. The numbers after 'DH-pgd' is the numbers at the 10th row and the 4th column. As for the numbers at the 10th row and the 3rd column and the numbers at the 10th row and the 5th column in Table 3, they can be gotten from the data used to generate Figure 4. 

   (We only keep the original models and datasets we used to get the results of CIFAR VGG16 and SVHN SADL-1. But users can generate their own models and datasets to get the similar results according to the steps in the following 'General Steps' section. We will take the steps to generate the results of MNIST LeNet-1 as an example. The difference between our results and the new results is acceptable and it will not affect our conclusion.)



## General Steps:

1. Download the dataset from google drive link:                  and put them in the 'data' folder. 

2. PGD attack the dataset: 

   We have put [commands](https://github.com/DNNTesting/CovTesting/blob/d462c59c1cbc00c2add20ee0eaf7a9966859788b/Table%202/test_example.sh#L8-L17) of PGD attacking in 'test_example.sh'. Please uncomment the line comment and run 'test_example.sh' to implement the corresponding PGD attack. Attacked dataset will be stored as ('./data/' + args.dataset + '_data/model/' + args.model  + _PGD.npy').

3. Use DeepHunter to generate test cases:
   We have put [commands](https://github.com/DNNTesting/CovTesting/blob/d462c59c1cbc00c2add20ee0eaf7a9966859788b/Table%202/test_example.sh#L20-L29) of DeepHunter in 'test_example.sh'. Please uncomment the line comment and run 'test_example.sh' to use DeepHunter to generate the corresponding test cases. Test cases will be stored as ('./data/' + args.dataset + '\_data/model/' + 'deephunter_adv_test_{}.npy'.format(args.model)).  

4. Use [commands](https://github.com/DNNTesting/CovTesting/blob/d462c59c1cbc00c2add20ee0eaf7a9966859788b/Table%202/test_example.sh#L32-L41) in 'test_example.sh' to compare DH and DP. Please uncomment the line comment and run 'test_example.sh' to get the corresponding result. The results will be stored in 'compare_result.txt'. 

   

   

   fuzzing: get DH new training dataset
   
   retrain_robustness:  get DH retrained model
   
   
   
   pgd_attack: PGD testing dataset
   
   deephunter_attack: DH testing dataset
   
   
   
   all: (bash)
   
   
   
   depends on the training parameters and the size of new training datasets. We provide one trained model for verifacation
   
   
   
   