# Tweet Virality Prediction

Sentiment analysis on social networks remains one of the most interesting fields to apply statistical tools and machine learning models, with the practical advantage of predicting trends and identifying patterns.  

In this project, we focus on **Twitter data** (now X) taken from the [Codecademy platform](https://www.codecademy.com) (course: *Build a Machine Learning Model*) and implement several machine learning models to predict whether a tweet becomes viral.  



### Project Overview

Using a **quartile-based division**, we classify tweet virality into four categories:  
`unpopular`, `average`, `popular`, and `super popular`.  

We then perform exploratory data analysis by plotting the distributions of relevant features, and conduct feature engineering to uncover potential hidden patterns in the tweets, such as:

- tweet length  
- number of hashtags 
- number of mentions 
- tweet language  


### Models Implemented

One of the main challenges in this analysis is the strong imbalance between virality classes, which can significantly affect evaluation metrics such as the F1-score.  

To address this issue, we implemented three different machine learning models:  
**K-Nearest Neighbor (KNN)**, **Random Forest**, and a **Sequential Neural Network** built with `PyTorch`.  

Each model was trained and evaluated on the same dataset to compare their performance in predicting tweet virality and handling class imbalance.


### Results Summary
**Neural Network Metrics**:                                        
              precision    recall  f1-score   support

           0       0.82      0.71      0.76       752
           1       0.48      0.07      0.12       365
           2       0.41      0.21      0.28       559
           3       0.36      0.82      0.50       544

    accuracy                           0.51      2220
   macro avg       0.52      0.45      0.42      2220
weighted avg       0.55      0.51      0.47      2220

**KNN Metrics**:
              precision    recall  f1-score   support

     average       0.39      0.17      0.24       366
     popular       0.34      0.48      0.39       561
   super_pop       0.41      0.46      0.43       576
       unpop       0.80      0.69      0.74       717

    accuracy                           0.49      2220
   macro avg       0.48      0.45      0.45      2220
weighted avg       0.51      0.49      0.49      2220

**Random Forest Metrics**:
              precision    recall  f1-score   support

     average       0.41      0.14      0.21       366
     popular       0.41      0.38      0.40       561
   super_pop       0.43      0.64      0.51       576
       unpop       0.76      0.74      0.75       717

    accuracy                           0.53      2220
   macro avg       0.50      0.48      0.47      2220
weighted avg       0.53      0.53      0.51      2220

- The **Neural Network** achieved **51% accuracy**, performing roughly twice as well as random selection (25% accuracy given the four classes).   
- However, its **F1-score for the minority class** (`average`) remained low at **12%**, indicating room for improvement.  
- The **KNN** model reached **49% accuracy**, but a higher **F1-score (24%)** for the minority class, suggesting a mild improvement in handling.
- The **Random Forest** model achieved the highest accuracy at **53%**, although its **F1-score for the minority class** remained slightly lower than that of **KNN** (**21%** vs. **24%**).
 

These results highlight that better handling of minority classes could lead to significant improvements in model performance across all metrics.



### Conclusion

This project demonstrates the potential of machine learning models to analyze social media data by predicting viral trends using a minimal set of features, while also highlighting the importance of addressing **class imbalance** for fairer and more reliable predictions.


