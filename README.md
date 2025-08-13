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

![Model Comparison Graph](images/model_comparison.png)

This shows strong performance for logistic regression, which had the highest test accuracy. The support vector classifier approach was not far behind, but took over 10x as long to run. K-nearest neighbors was very fast and decently accurate, while the decision tree approach was heavily overfit.

To address the overfitting on the decision tree approach, I performed a grid search varying the max_depth parameter of the decision tree classifier, from 2-4-6-8. This essentially controls how complex the decision tree is allowed to be, reducing the potential for overfitting. The tree with max depth of 6 turned out to perform the best, and actually slightly outperformed the original logistic regression approach, with a 92% train accuracy and a 91.7% test accuracy.

I also tried to see if I could improve the performance of the K-nearest neighbors classifier by varying the n_neighbors parameter (default is 5). A grid search found n_neighbors=17 to be the optimal parameter value, however performance only improved marginally (from 90.2% test accuracy to 91.0% test accuracy).

### Evaluation

The models that I created performed ok - a mae of \$6800 and a medae of \$4000 is respectable, but not super precise. I think that the messiness of the original dataset was quite difficult to deal with here. I had to throw out the model column almost entirely. I think that with some data that was better collected, or more precisely transcribed, it would have been easier to build a more precise model. In addition, with more time and resources I could have potentially undertaken a much more extensive data cleaning operation, revealing more useful information that the model could use to make its predictions more precise.

### Deployment

A lot of the feedback that I have for the client, as I have already described extensively throughout this report, has to do with data collection and transcription. A lot of the data here is quite messy and hard to work with, or missing completely. A standardized list of the most common car models that my client cars most about would be really helpful. If my client sells, for example, a lot of Ford F-150s, we could build an individual model for just Ford F-150s, and that would be far higher precision for those specialized cases than this general model. But doing that untargeted in this exercise would be building far too many models without that additional info from my client.

In other words, I guess that part of the takeaway here is that business understanding is extremely important for this kind of task. Because this is not a real report for a real client, I find that my business understanding of what my client is looking for is extremely vague, because all we know is that it is some generic used car dealership. In the real world, I imagine that I would want to ask a lot about exactly what their business looks like, where it is located, what kinds of cars they sell most often, and so on. Indeed, a dataset of cars that my client has sold in the past would help a lot.

Along similar lines, something that jumps out is the effect of antique cars on the used car market. Generally, as a car gets older, its value decreases - until you get to rare antiques, where the value skyrockets. If my client only ever sells 'normal' used cars, and doesn't sell rare antiques, we could exclude all of those from the dataset, and build a model that is much more sensitive to the price of 'normal' used cars. This model would perform much worse on rare antiques, but going back to the business understanding, the point of the model is to help the business, and it doesn't matter if the model performs poorly on a category of car that is irrelevant to the business. My client would happily take that tradeoff to increase modeling precision on its actual inventory.