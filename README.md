# Correlations Between Deep Neural Network Model Coverage Criteria and Model Quality

<a href="https://doi.org/10.5281/zenodo.3908793"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.3908793.svg" alt="DOI"></a>

Codes for paper: Correlations Between Deep Neural Network Model Coverage Criteria and Model Quality



## Dependencies:

The experiments are run by python 3.6 or 3.7 under Ubuntu 18.04.  

Way 1: Get environment with preinstalled dependencies using docker: 

```python
# Get the environment (OS) to run the code
$ docker pull hao359510974/covtesting2:latest
$ docker run -it --mount type=bind,src=SRC_PATH,dst=DEST_PATH hao359510974/covtesting2:latest # Where SRC_PATH and DEST_PATH must be absolute paths; SRC_PATH is the path on your host machine, and DEST_PATH is the file path for where you want it to be stored in the Docker.
    
#Experiments (take Comparison of Attack Images (Table 2) as an example, for other experiments just use other folders.)
$ cd /data/Comparison\ of\ Attack\ Images/

Then run commands according to 'README' file to get corresponding results. 
```

Way 2: We also provide [requirements.txt](https://github.com/DNNTesting/CovTesting/blob/master/requirements.txt) file for users who want to establish the environment on their own machine:

```$ pip install -r requirements.txt```

## Structure:

CovTesting:

&emsp;&emsp;|

&emsp;&emsp;|-----------> ['Comparison of Attack Images' folder](https://github.com/DNNTesting/CovTesting/tree/master/Comparison%20of%20Attack%20Images): Codes for comparison of attack images (Table 2)

&emsp;&emsp;|-----------> ['Correlation' folder](https://github.com/DNNTesting/CovTesting/tree/master/Correlation): Codes for calculation of correlation (Figure 4 and figure 8)

&emsp;&emsp;|-----------> ['Coverage VS. # Adv Examples' folder](https://github.com/DNNTesting/CovTesting/tree/master/Coverage%20VS.%20%23%20Adv%20Examples): Codes for coverage VS. # adv examples (Figure 6 and figure 7)

&emsp;&emsp;|-----------> ['Coverage and Robustness VS. Training Datasets' folder](https://github.com/DNNTesting/CovTesting/tree/master/Coverage%20and%20Robustness%20VS.%20Training%20Datasets): Codes for coverage and robustness VS. training datasets (Figure 2 and figure 3)

&emsp;&emsp;|-----------> ['Model Accuracy under Different Scenarios' folder](https://github.com/DNNTesting/CovTesting/tree/master/Model%20Accuracy%20under%20Different%20Scenarios): Codes for model accuracy under different scenarios (Table 3)

&emsp;&emsp;|-----------> ['data' folder](https://github.com/DNNTesting/CovTesting/tree/master/data): Reserved data folder for 'Correlation', 'Coverage VS. # Adv Examples' and 'Coverage and Robustness VS. Training Datasets'

&emsp;&emsp;|-----------> CONTACT.md

&emsp;&emsp;|-----------> INSTALL.md

&emsp;&emsp;|-----------> LICENSE.md

&emsp;&emsp;|-----------> README.md

&emsp;&emsp;|-----------> STATUS.md

&emsp;&emsp;|-----------> [esecfse2020-paper53.pdf](https://github.com/DNNTesting/CovTesting/blob/master/esecfse2020-paper53.pdf): A copy of our accepted paper

&emsp;&emsp;|-----------> index.md

&emsp;&emsp;|-----------> [requirements.txt](https://github.com/DNNTesting/CovTesting/blob/master/requirements.txt): Required dependencies]



## Before Start (Prepare the Data Used in Experiments):

Step 1: Download data and codes:

Download data and codes through the DOI link: <a href="https://doi.org/10.5281/zenodo.3908793"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.3908793.svg" alt="DOI"></a>

You will get two zip files: 'all-data.zip' (12.2 G, the data file) and 'DNNTesting/CovTesting-v1.1.zip' (5.2 M, the codes file). Please unzip the codes and data files to get the codes and data for experiments. After unzipping 'all-data.zip', you will get three zip files named 'data.zip' (4.1 G), 'Table 2 data.zip' (3.5 G) and 'Table 3 data.zip' (4.6 G).

(We also provide a Google Drive link for users to download the data: https://drive.google.com/drive/folders/16w93LPkaF0AP9QxIty9Y6ipU-N4cbPUd?usp=sharing. )

Step 2: Put the data folders at corresponding locations

1. Unzip 'data.zip' and get three folders: 'cifar_data', 'mnist_data' and 'svhn_data'. Please put these three folders under the ['data' folder](https://github.com/DNNTesting/CovTesting/tree/master/data) in the main folder. 
2. Unzip 'Table 2 data.zip' and get three folders: 'cifar_data', 'mnist_data' and 'svhn_data'. Please put these three folders under the ['data' folder](https://github.com/DNNTesting/CovTesting/tree/master/Comparison%20of%20Attack%20Images/data) under the ['Comparison of Attack Images' folder](https://github.com/DNNTesting/CovTesting/tree/master/Comparison%20of%20Attack%20Images). 
3. Unzip 'Table 3 data.zip' and get two folders: 'data' and 'new_model'. Please merge these two folders into the ['data' folder](https://github.com/DNNTesting/CovTesting/tree/master/Model%20Accuracy%20under%20Different%20Scenarios/data) and the ['new_model' folder](https://github.com/DNNTesting/CovTesting/tree/master/Model%20Accuracy%20under%20Different%20Scenarios/new_model) under the ['Model Accuracy under Different Scenarios' folder](https://github.com/DNNTesting/CovTesting/tree/master/Model%20Accuracy%20under%20Different%20Scenarios), respectively. 

