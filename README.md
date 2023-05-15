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

* Machine Learning Model 1 (Original Data):
  * Balanced Accuracy Score of 95.2%. 
  * Precision Score of 92% (an average -- in predicting low-risk loans, the model was 100% precise, but only 85% precision on predicting high-risk loans.)
  * Recall Score was 95% (an average --  the model had 99% recall in predicting healthy loans and 91% in predicting high-risk loans)
