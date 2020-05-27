# Table 2

---
## Quick Start 

1. Download the dataset from google drive link:                  and put them in the 'data' folder. 

2. We have put [commands](https://github.com/DNNTesting/CovTesting/blob/e60737edce6cc275d2044dbe097f92f28d11f2ac/Table%202/test_example.sh#L25-L34) in 'test_example.sh'. Please uncomment the line comment and run 'test_example.sh' to get the corresponding result. The results will be stored in 'compare_result.txt'. 

   (We only keep the original Images dataset for CIFAR and SVHN models. We re-attack the MNIST models and use the newly generated dataset to do testing. So the results of MNIST is slightly different from those shown in our paper. But as you can see, the difference is acceptable and it will affect our conclusion.)



## General Steps:

1. PGD attack the dataset: 

   We have put [commands](https://github.com/DNNTesting/CovTesting/blob/e60737edce6cc275d2044dbe097f92f28d11f2ac/Table%202/test_example.sh#L7-L22) of PGD attacking in 'test_example.sh'. Please uncomment the line comment and run 'test_example.sh' to implement the corresponding PGD attack. Attacked dataset will be stored in ('./data/' + args.dataset + '_data/model/' + args.model  + _PGD.npy').

2. Use DeepHunter to generate test cases:

   