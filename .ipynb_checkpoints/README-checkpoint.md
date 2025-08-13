# Understanding The Success of Marketing Campaigns

**OVERVIEW**

In this report, I am acting a consultant to a bank. The client has provided a dataset ([bank=additional-full.csv](data/bank-additional-full.csv)) that contains information on a large number of marketing campaigns. These were telephone marketing campaigns attempting to get customers or prospective customers to subscribe to a term deposit with the client bank.

The code and computation for this report is in the linked [Jupyter notebook](prompt_III.ipynb).

In approaching this problem, I am to follow the industry standard **CRISP-DM** framework for data science tasks:

<center>
    <img src = images/crisp.png width = 50%/>
</center>

First, we should start with a basic business understanding of what the business is trying to accomplish through this analysis.

### Business Understanding

The business objective of our task is to aid this banking institution in targeting customers who are likely to subscribe to this bank's term deposit account. As shown below, marketing campaigns of this type do not convert a large percentage of users; only about 11% of customers in this campaign went through with a subscription. We have lots of information about the demographics and financial information of targeted users, as well as information about when the potential customer was contacted. Our goal is to uncover patterns in the data, allowing the bank to more effectively choose who to contact, which hopefully in the future will lead to a higher percentage of customers successfully converted (in future campaigns).

### Data Understanding

(More information about the dataset is available [here](https://archive.ics.uci.edu/dataset/222/bank+marketing). See also this accompanying paper: [Using Data Mining for Bank Direct Marketing: An Application of the CRISP-DM Methodology](CRISP-DM-BANK.pdf))

An investigation of the data reveals the information that we have to work with. This is quite a nicely prepared dataset; there are no missing values, and an evaluation of the individual columns indicates that they all seem at least potentially useful. The columns contain various demographic information about contacted customers, as well as information about the time when the user was contacted (and how many times, etc.), as well as current economic conditions.

This data comes from 17 marketing campaigns.

### Data Preparation

There are three main transformations I perform on the data to prepare it for the modeling stage:

First, in order to attempt to potentially find certain non-linear patterns in the data, I created polynomial features that are combinations of the 9 numerical columns. I chose up to degree 3, which is perhaps too many, as this creates 277 additional features. However, this will hopefully increase the explanatory power of our model.

Second, I will scale the numerical features including the created polynomial combinations to be normally distributed using sklearn's StandardScaler, which will make it easier to work with for the various classification algorithms.

Finally, for the remaining categorical features, I use one-hot encoding to transform these into numerical columns.

Oh, and the last thing to do is to split the data into a training set and a test set to judge the various algorithms' performance on.

### Modeling

In the modeling stage, I built a variety of classification models and compared their performance.

First, I created a dummy classifier that simply predicted that every customer would reject the marketing campaign. Because about 88% of customers in the dataset rejected the marketing campaign, this dummy classifier has an accuracy of 88%. This is important to keep in mind, because it gives us a baseline to compare our actual classification algorithms against.

Next, I generated classifiers using four classification methods: linear regression, K-nearest neighbors, a decision tree, and support vector machines. The graph below shows the results of my initial test:

![Model Train Time](images/train_time.png)

![Model Train and Test Accuracy](images/model_accuracy.png)

This shows strong performance for logistic regression, which had the highest test accuracy. The support vector classifier approach was not far behind, but took over 10x as long to run. K-nearest neighbors was very fast and decently accurate, while the decision tree approach was heavily overfit.

To address the overfitting on the decision tree approach, I performed a grid search varying the max_depth parameter of the decision tree classifier, from 2-4-6-8. This essentially controls how complex the decision tree is allowed to be, reducing the potential for overfitting. The tree with max depth of 6 turned out to perform the best, and actually slightly outperformed the original logistic regression approach, with a 92% train accuracy and a 91.7% test accuracy.

I also tried to see if I could improve the performance of the K-nearest neighbors classifier by varying the n_neighbors parameter (default is 5). A grid search found n_neighbors=17 to be the optimal parameter value, however performance only improved marginally (from 90.2% test accuracy to 91.0% test accuracy).

I also compared the recall scores of the various approaches (discussed more in the [Jupyter notebook](prompt_III.ipynb)) and found that the decision tree with max depth of 6 performed the best in terms of recall score.

Recall scores:
Dummy classifier:         0.0
Logistic Regression:      0.492
KNN (k=5):                0.488
Decision Tree (defaults): 0.533
SVC:                      0.426
Decision Tree (depth6)  : 0.568
KNN (n=17)              : 0.484

### Evaluation

# Logistic Regression Classifier: Largest Coefficients (positive and negative):

 euribor3m^3                            2.028670
 emp.var.rate^3                         1.446931
 duration^3                             1.410260
 pdays euribor3m^2                      1.213136
 age euribor3m^2                        0.981491
 month_sep                              0.960668
 euribor3m^2 nr.employed                0.944269
 campaign emp.var.rate cons.conf.idx    0.930110
 euribor3m^2                            0.883121
 cons.price.idx euribor3m^2             0.874995
 
 month_apr                             -0.822581
 cons.conf.idx^3                       -0.894728
 month_jun                             -0.967512
 age^3                                 -1.083950
 duration^2 nr.employed                -1.143500
 month_may                             -1.248751
 cons.conf.idx euribor3m^2             -1.288728
 duration^2                            -1.389337
 duration^2 cons.price.idx             -1.410246
 age^2 cons.conf.idx                   -1.677975
 

For the evaluation, I find it easiest to look at the logistic regression classifier - one of the highest performing classifiers - and see which coefficients are largest (positive or negative), indicating which columns had the largest impact on the classifier. We see here that euribor3m - the 3 month interest rate for European banks - is one of the strongest indicators of the success of a marketing campaign. This and other major coefficients here would indicate that current economic conditions are of larger importance than demographic factors when conducting a marketing campaign like this one, which is an interesting and important insight.

Along similar lines, we find that the consumer confidence index has a strong negative correlation with wanting to subscribe to a savings account, which makes sense as well. We also see that time of year plays a major role: the campaigns were very unsuccessful in April, May, and June, while highly successful in September.

These insights go against my initial assumption that this kind of marketing campaign would be mostly about targeting users with the right demographics. However, it seems that while certain demographic qualities such as age are important, in general most demographic factors are overshadowed by global conditions and timeframes instead. This is good to know, as it means that the bank can decide to target their campaigns based primarily on public, global data, rather than having to invest in data collection about the demographics of targeted consumers, potentially saving a lot of resources while also increasing campaign effectiveness.