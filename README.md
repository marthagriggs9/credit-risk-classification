# credit-risk-classification

## Split the Data into Training and Testing Sets

1. read the `lending_data.csv` data from the Resources folder into a Pandas DataFrame.
2. Create the labels for set (`y`) from the 'loan_status' column and then create the features (`x`) DataFrame from the remaining columns. 
3. Split the data into training and testing datasets by using `train_test_split`. 


## Create a Logistic Regression Model with the Original Data

1. Fit a logistic regression model by using the training data (`X_train` and 'y_train`). 
2. Save the predictions for the testing data labels by using the testing feature data (`X_test`) and the fitted model. 
3. Evaluate the model's performance by doing the following:
   * Calculate the accuracy score of the model. 
   * Generate a confusion matrix. 
   * Print the classification report. 

4. Answer the following question:
   * How well does the logistic regression model predict both the 0 (healthy loan) and 1 (high-risk loan) labels?

## Create a Logistic Regression Model with Resampled Data

1. Use `RandomOverSampler` module from the imbalanced-learn library to resample the data.
   * Be sure to confirm that the labels have an equal number of data points.
2. Fit a logistic regression model by using the training data (`X_train` and 'y_train`). 
3. Save the predictions for the testing data labels by using the testing feature data (`X_test`) and the fitted model. 
4. Evaluate the model's performance by doing the following:
   * Calculate the accuracy score of the model. 
   * Generate a confusion matrix. 
   * Print the classification report. 

4. Answer the following question:
   * How well does the logistic regression model predict both the 0 (healthy loan) and 1 (high-risk loan) labels?


## Overview of the Analysis

In this section, describe the analysis you completed for the machine learning models used in this Challenge. 

The dataset used contained information on:
  * loan size
  * interest rate
  * borrower's income
  * debt to income ratio
  * number of accounts the borrower has
  * derogatory marks against the borrower
  * total debt of borrower

The dataset contained 77,536 data points and was split into training and testing groups. The training set was used to build a logistical regression model using `LogisticRegression` module. This was then applied to the testing dataset. The purpose of the model (Machine Learning Model 1) was to determine whether a loan issued to a borrower in the testing set would be low (healthy) or high-risk (unhealthy). 

The model that used the original data had 75,036 low-risk data points and only 2,500 high-risk data points. This data was then resampled to ensure that that the logistic regression model had an equal number of data points to learn from. The training dataset was resampled with `RandomOverSampler` module from imbalanced-learn. There were then 56,271 data points for both low-risk and high-risk loans. This resampled data was used to build a new logistic regression model. Again, the purpose of the model (Machine Learning Model 2) was to determine whether a loan issued to a borrower in the testing set would be low (healthy) or hgh-risk (unhealthy). 

## Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1 (Original Data):
  * Balanced Accuracy Score of 95.2%. 
  * Precision Score of 92% (an average -- in predicting low-risk loans, the model was 100% precise, but only 85% precision on predicting high-risk loans.)
  * Recall Score was 95% (an average --  the model had 99% recall in predicting healthy loans and 91% in predicting high-risk loans)


* Machine Learning Model 2 (Resampled Data):
  * Balanced Accuracy Score of 99.3%
  * Precision Score of 92% (an average -- in predicting healthy loans, the model was 100% precise. It was only 84% precise in predicting high-risk loans.)
  * Recall Score was 99% for predicting both healthy and high-risk loans. 

## Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:

* Machine Learning Model 2 has a higher balanced accuracy score (99.3%). This resampled data had less false predictions overall. Model 2's recall score showed that this model was more accurate at predicting high risk loans. This is important if the bank wants to minimize financial losses. Model 1 had a lower recall score for predicting high-risk loans which could end up being bad for the bank, especially if a large loan is issued to a 'high-risk' client.

* While neither model scored above 90% in precision when predicting high-risk loans, there is only the risk of losing potential customers, which to me is less of a loss than clients not being able to pay their loan back. Model 2 also had a more balanced dataset to learn from, which I feel trumps what Model 1 was able to do, considering its much higher number of healthy loans in the the training set used. 
