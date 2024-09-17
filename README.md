# p2p-lending-loan-default-prediction
This project utilized the LightGBM algorithm alongside the Boruta feature selection technique to identify the most critical features and develop a predictive model for forecasting loan defaults in the Peer-to-Peer (P2P) lending sector. The goal was to create a model that accurately predicts loan defaults, benefiting lenders and P2P lending platforms. 

The scope of this project will be focusing on the prediction of loan default in the Peer-to-Peer (P2P) industry. In this project, relevant data regarding the loan details of the P2P industry will be collected from the [Kaggle website](https://www.kaggle.com/datasets/wordsforthewise/lending-club). The targeted audiences would be the management of the P2P platforms, the lenders and investors that lend funds to borrowers. The outcome of the project would be in the form of a predictive model that could predict the loan default in the P2P lending industry. 

The codes to conduct the project are separated into several parts, with part 1 as the preprocessing of the data, part 2 as the model development and part 3 as the model evaluation.

# Problem Statement
The stability and profitability of P2P lending will be part of the concerns and risk management for the lenders and P2P platforms. The risk of loan defaults by borrowers will result in a threat to the confidence of lenders, this will cause potential disinvestment from lenders which further leads to customer churn in the P2P industry. The loan defaults would incur huge losses for the lenders and affect the development and confidence in the ecosystem of P2P lending (Setiawan et. al., 2019; Tritto et. al., 2020). Thus, a credit rating and risk assessment mechanism using machine learning models could be applied so that the risk faced by lenders could be reduced to as low as possible. Furthermore, some of the features might be redundant and do not contribute to the prediction of loan default. Therefore, there is a need to experiment to reduce the complexity of the model. This is also to reduce the resources and time required for the training of the loan default predictive model (Xu et. al, 2021). 

#Scope
The scope of this project will be focusing on the prediction of loan default in the Peer-to-Peer (P2P) industry. In this study, relevant data regarding the loan details of the P2P industry will be collected from the [Kaggle website](https://www.kaggle.com/datasets/wordsforthewise/lending-club). The targeted audiences would be the management of the P2P platforms, the lenders and investors that lend funds to borrowers. The outcome of the project would be in the form of a predictive model that could predict the loan default in the P2P industry. Other than that, a report with methodology, findings and recommendations would also be part of the deliverables.

# Project Methodolgy
## Data Collection
For this project, data from the world’s largest Peer-to-Peer (P2P) platform which is the Lending Club. The year of the data collected would be from the year 2006 to the year 2018. The data includes the loan status of individuals, information about the borrowers and so on. There will be around 2 million instances and 150 features in the dataset. Since the number of features is huge, initial filtration of features will be done using a literature review to select features that are useful and promising. This is also to reduce the effect of multicollinearity and complexity of the predictive models.

## Data Preparation, Preprocessing and Initial Data Exploration
Exploratory Data Analysis (EDA) will be conducted on the dataset to understand the patterns and identify missing values as well as inconsistency in the dataset. EDA will be performed simultaneously with the data preprocessing to ensure that no inconsistencies. In the data preprocessing step, the missing values in the numerical and categorical variables will be imputed with mean and mode values respectively. Data transformation, one-hot encoding and other necessary steps will also be conducted accordingly. The purpose of preprocessing is to make sure the data is of high quality and improve the accuracy of the predictive models.
The collection consisted of rejected and accepted loan applications but only the accepted loan application dataset will be used in this project.  The dataset was first imported to the working environment for further initial data analysis, data preprocessing and modelling. Initial data analysis found that there were 2,260,701 rows and 151 features in the imported dataset, with loan status as the target variable. It was found that some of the features contained more than 30% of missing values. Thus, those features will be dropped from the dataset as they could only provide limited information for the experiment in this project. Even if imputations of those missing values were done, they would still introduce significant biases. The Jupyter notebook for the preprocessing is shown as below.

# Feature Selection Technique
In machine learning, one of the challenges is dimensionality. As the dimensionality becomes greater, the search space volume will also increase at the same rate. This will cause the data to be sparse and result in model overfitting. One of the scopes and aims of this study is to use the feature selection technique to filter and select features that are useful in the prediction of loan default in the P2P industry. This study will particularly use the Boruta Algorithm as the feature selection technique to select and analyze important features in the Lending Club dataset before the model creation.

## Boruta Algorithm
A Boruta technique from the [GitHub repository](https://github.com/scikit-learn-contrib/boruta_py) that consists of the implementation of the Boruta R package in Python language and scikit-learn compatible will be used in the feature selection method. Boruta algorithm is a feature selection technique and algorithm that is generally used in the machine learning model. It is a wrapper feature selection technique that is designed around the Random Forest classification, and it can capture important features.

# Machine Learning Algorithms
After the feature selection, three different predictive models will be proposed for the prediction of loan default in the P2P industry. The models will be built, and their performances will be evaluated. The model with the best performance and accurate results will be chosen as the desired model. The three models to be proposed are as follows:
 - Logistic Regression
 - LightGBM
 - Support Vector Machine


To make this model accessible and user-friendly, I also built a simple yet effective web application using Streamlit. This demo app allows users to input loan-related data and receive real-time predictions on whether a loan is likely to default.

Key Components of the web app:
- **Machine Learning Model**: A LightGBM model, fine-tuned to accurately predict loan default risks based on normalized numerical and categorical data.
- **Interactive Web Application**: Developed with Streamlit, this app provides an easy-to-use interface for data input and prediction.
- **Numerical Data Normalization**: Ensures that input values are scaled appropriately to match the model’s training data.
- **Categorical Feature Handling**: Allows users to select from various categorical options, with the app managing unused options correctly.



## Demo
You can view the live demo of the application [here](https://p2p-lending-loan-default-prediction.streamlit.app/).

![Demo GIF](images/streamlit-demo.gif)

# Conclusion
It was found that the features related to the verification status of the applicants’ information, annual income, average current balance, and debt-to-income ratio appeared to be the influential features with high feature importance scores. The features contributed to the loan defaults were determined using the output of the Boruta algorithm and this achieved the first objective of the study. After the feature selection step, the LightGBM, Logistic Regression and SVM models were trained using the dataset. Base models for each of these models were firstly constructed and subsequently fine-tuned using GridSearchCV and RandomizedSearchCV techniques. The LightGBM model obtained an accuracy of 65.26% and was chosen as the most suitable predictive model for the prediction of loan default in the P2P lending domain. Furthermore, the LightGBM model outperformed the proposed Support Vector Machine and Logistic Regression models across all the evaluation metrics. This fully met the second and third objectives of the study.
 
In future studies, the number of iterations in the Boruta algorithm can be increased to comprehensively explore the features in the dataset. Moreover, statistical analysis can also be employed together with the Boruta feature selection technique to address the issue of multicollinearity and group features into latent factors. It is also recommended that future studies collect datasets from multiple sources to refine the predictive model. This is to improve the generalization of the model and the predictive performance of the model. The model can leverage diverse characteristics of data and reduce the dependence on a single source of data. Next, other strategies of hyperparameter optimization such as metaheuristic optimization techniques can be employed to explore the most optimal configuration of hyperparameters. Lastly, future studies can also involve the development of a user-friendly Graphical User Interface system that allows the usage of the loan default predictive model for users without any technical background.
