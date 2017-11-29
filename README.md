# banking_fraud-
Detecting fraud in finance 

Banks and other financial institutions are now employing more machine learning techniques to detect suspicious activity on user accounts. In this spirit, we used two machine learning algorithms K-nearest neighbors and logistic regression to detect cases of fraud.

One key advantage of K-nearest neighbors is that it allows the algorithm to be used for both classification and regression problems. Using this approach, the PaySim dataset was first feature scaled to limit the range of variables (i.e. normalization). Using distance based methods (i.e. KNN classifier) on non-feature scaled data will skew our predictions because the variable with the largest range will dominate the outcome results.

For our purposes, the user’s transaction amount, old balance, new balance, old balance destination, and new balance destination acted as the predictors in the K-nearest neighbors classification model. Overall, the algorithm accurately predicted 99.9% of all frauds.

We also tested fraud classification with logistic regression. Logistic regression is a statistical method used to describe the relationship between the dependent binary variable (fraud or no fraud) and the predictor variables. this approach, we first standardized the dataset, which rescales the data from the original range so that all values are within range 0 to 1. The logistic regression model also correctly identified over 99% of fraud events.

Read article here: https://beyondtheaverage.wordpress.com/2017/11/28/where-is-the-fraud/

