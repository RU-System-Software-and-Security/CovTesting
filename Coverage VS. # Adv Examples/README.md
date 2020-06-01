# Coverage VS. # Adv Examples (Figure 6 and figure 7)

---
## General Steps:

1. Download the dataset from google drive link: https://drive.google.com/drive/folders/1WXqnuBT0FISMyYuYGShbjGwOxK7nHdGK?usp=sharing and merge them to the 'data' folder.

2. Get the results of coverage metrics change with numbers of adversarial examples (figure 6):

   ```$ python coverage_curve.py ``` 

   All results will be stored in 'coverage_result.txt'. It will also generate several '.npy' files and store them in 'Q2_original' folder for post-process. The excel file we use to generate figure 6 is 'figure 6.xlsx'. It includes the results and the figure 6 we got.

3. Get the change of TKNP with numbers of inputs (blue line in figure 7):

   ```$ python tknp_testing.py ``` 

   The results will be stored in 'testing_coverage_result.txt'. It will also generate a 'tknp_all.npy' file in 'Q2_original' folder  for post-process. To generate figure 7, we need the results from this step and "TKNP VS. # adversarial examples" related data from step 2 ('cov_tknp.npy' in 'Q2_original' folder, the red line in figure 7). The excel file we use to generate figure 7 is 'figure 7.xlsx'. It includes the data we use and the figure 7 we got.

