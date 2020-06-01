# Table 2

---
## Quick Start:

1. Download the dataset from google drive link: https://drive.google.com/drive/folders/1WXqnuBT0FISMyYuYGShbjGwOxK7nHdGK?usp=sharing and merge them to the 'data' folder. Then download other data from google drive link: https://drive.google.com/drive/folders/1_j-z2FAd8tkfG6AbAEqg05C8InpAGUNm?usp=sharing and merge them to the 'data' folder again. There must be several files that have same names as the files from the first link. Please replace those from the first link. 

2. We have put [commands](https://github.com/DNNTesting/CovTesting/blob/d462c59c1cbc00c2add20ee0eaf7a9966859788b/Table%202/test_example.sh#L32-L41) in 'test_example.sh'. Please uncomment the line comment and run 'test_example.sh' to get the corresponding result. The results will be stored in 'compare_result.txt'. 

   (We only keep the original Images dataset for CIFAR and SVHN models. We re-attack the MNIST models and use the newly generated dataset to do testing. So the results of MNIST is slightly different from those shown in our paper. But as you can see, the difference is acceptable and it will not affect our conclusion.)



## General Steps:

1. Download the dataset from google drive link: https://drive.google.com/drive/folders/1WXqnuBT0FISMyYuYGShbjGwOxK7nHdGK?usp=sharing and merge them to the 'data' folder. Then download other data from google drive link: https://drive.google.com/drive/folders/1_j-z2FAd8tkfG6AbAEqg05C8InpAGUNm?usp=sharing and merge them to the 'data' folder again. There must be several files that have same names as the files from the first link. Please replace those from the first link. 

2. PGD attack the dataset: 

   We have put [commands](https://github.com/DNNTesting/CovTesting/blob/d462c59c1cbc00c2add20ee0eaf7a9966859788b/Table%202/test_example.sh#L8-L17) of PGD attacking in 'test_example.sh'. Please uncomment the line comment and run 'test_example.sh' to implement the corresponding PGD attack. Attacked dataset will be stored as ('./data/' + args.dataset + '_data/model/' + args.model  + _PGD.npy').

3. Use DeepHunter to generate test cases:
   We have put [commands](https://github.com/DNNTesting/CovTesting/blob/d462c59c1cbc00c2add20ee0eaf7a9966859788b/Table%202/test_example.sh#L20-L29) of DeepHunter in 'test_example.sh'. Please uncomment the line comment and run 'test_example.sh' to use DeepHunter to generate the corresponding test cases. Test cases will be stored as ('./data/' + args.dataset + '\_data/model/' + 'deephunter_adv_test_{}.npy'.format(args.model)).  

4. Use [commands](https://github.com/DNNTesting/CovTesting/blob/d462c59c1cbc00c2add20ee0eaf7a9966859788b/Table%202/test_example.sh#L32-L41) in 'test_example.sh' to compare DH and DP. Please uncomment the line comment and run 'test_example.sh' to get the corresponding result. The results will be stored in 'compare_result.txt'. 

   

   

   