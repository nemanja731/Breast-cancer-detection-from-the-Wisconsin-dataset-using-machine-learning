# Breast cancer detection using ML

You can see the entire report in the [projekatML.pdf](./projekatML.pdf)

The database used in this project is the **Wisconsin Breast Cancer Database** which can be found on [Kaggle](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data). There are 569 real samples in the database, and for each of them there are 30 attributes that describe it in more detail. Each sample has an assigned diagnosis class that indicates which tumor it is. The tumor can be malignant or benign. The goal of the project is to implement ML algorithms and try to use them to assess whether a new sample has a malignant or benign tumor.

<img src="./images/tumor.png" width="50%" align="center"/>

## Machine learning

The ML techniques used in this project are:
- Decision tree
- Random forrest
- AdaBoost
- XGBoost
- Logistic regression
- Naive Bayes

## Instructions
Firstly, install all basic dependencies:
```bash
pip install matplotlib
```
```bash
pip install numpy
```
```bash
pip install pandas
```
```bash
pip install seaborn
```
```bash
pip install xgboost
```
```bash
pip install scikit-learn
```

Then, you can easily run the code by copying the following command into the terminal:
```bash
python3 classification.py
```
