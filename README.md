## House Price Prediction

+ Tools - Python code in Jupyter notebook, Tableau for vizualization, Powerpoint, Excel.

+ Dataset - The dataset file contains both the raw data set and cleaned dataset.

+ Algorithms Used - KNeighborsRegressor, Linear Regression, Ridge, Lasso, Polynomial Regression.

  + KNeighborsRegressor - KNeighborsRegressor is a machine learning algorithm used for regression tasks. It is part of the scikit-learn library in Python and is based on the K-Nearest Neighbors (KNN) approach, similar to the KNeighborsClassifier for classification tasks. The KNN algorithm works by predicting the target variable of a new data point based on the average (or weighted average) of the target values of its k-nearest neighbors in the feature space.

  + Linear Regression - Linear Regression is a statistical method and a fundamental machine learning algorithm used for predicting a continuous outcome variable (also called the dependent variable) based on one or more predictor variables (independent variables). The relationship between the variables is assumed to be linear, meaning that a change in the predictor variables is associated with a linear change in the outcome.

  + idge - Ridge Regression, also known as Tikhonov regularization or L2 regularization, is a linear regression variant that introduces a regularization term to the standard linear regression objective function. The purpose of Ridge Regression is to prevent overfitting and handle multicollinearity, which occurs when predictor variables are highly correlated.

  + Lasso - Lasso Regression, short for Least Absolute Shrinkage and Selection Operator, is a linear regression technique that introduces a regularization term to the standard linear regression objective function. Like Ridge Regression, Lasso aims to prevent overfitting and handle multicollinearity in the presence of highly correlated predictor variables.

  + Polynomial Regression - Polynomial Regression is a type of linear regression in which the relationship between the independent variable (predictor variable) and the dependent variable is modeled as an nth-degree polynomial. It extends the simple linear regression model, allowing for a more flexible representation of the relationship between variables.
 
+ Methodology:
  
  •	Data Collection: The data used to build the model was provided in the csv format. It has different columns such as price, price per square foot, location, availability, area type, 
    number of bedrooms and bathrooms, etc.

  •	Data Pre-processing: Data pre-processing is a crucial step to ensure the quality and suitability of the dataset for training machine learning models. Checked for missing values.         Handled categorical data (if any).

  •	Feature Selection: Feature selection is a critical step to identify the most relevant variables that contribute to the predictive power of the model.

  •	Model Selection: In the model selection section, provide a detailed overview of the machine learning algorithms chosen for the predictive analysis. Explain the rationale behind the      selection of each algorithm and discuss how they align with the project objectives.  Used KNeighborsRegressor, Linear Regression, Ridge, Lasso, and Polynomial Regression.
  
  •	Model Training: In the model training section, the processed data is fit to train the selected model so that it is able to predict the future entered data. Split the dataset into        training and testing sets. Trained each model on the training set.

  •	Model Evaluation: In the model evaluation section, the performance of the trained machine learning models is assessed to select the best suited model for deployment. Evaluated the       models using accuracy, precision, recall, and F1-score.



