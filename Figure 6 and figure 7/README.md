# Figure 6 and figure 7

---
## Quick start

1. Download the dataset from google drive link:                  and put them in the 'data' folder. 

2. Run 'coverage_curve.py' to get the results of coverage metrics change with numbers of adversarial examples (figure 6). All results will be stored in 'coverage_result.txt'. It will also generate several '.npy' files and store them in 'Q2_original' folder for post-process. The excel file we use to generate figure 6 is 'figure 6.xlsx'. It includes the results and the figure 6 we got.
3. Run 'tknp_testing.py' to get the change of TKNP with numbers of inputs (blue line in figure 7). The results will be stored in 'testing_coverage_result.txt'. It will also generate several 'tknp_all.npy' file in 'Q2_original' folder  for post-process. To generate figure 7, we need the results from this step and "TKNP VS. # adversarial examples" related data from step 2 ('cov_tknp.npy' in 'Q2_original' folder, red line in  figure 7). The excel file we use to generate figure 7 is 'figure 7.xlsx'. It includes the data we use and the figure 7 we got.

