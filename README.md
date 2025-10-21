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

- The **Neural Network** achieved the highest accuracy at **51%**, performing roughly twice as well as random selection (25% accuracy given the four classes).   
- However, its **F1-score for the minority class** (`average`) remained low at **15%**, indicating room for improvement.  
- The **KNN** model reached **49% accuracy**, but a higher **F1-score (24%)** for the minority class, suggesting better balance.
- **Random Forest** generally has the lowest metrics, underperforming both the aforementioned models. 

These results highlight that better handling of minority classes could lead to significant improvements in model performance across all metrics.



### Conclusion

This project demonstrates the potential of machine learning models to analyze social media data by predicting viral trends using a minimal set of features, while also highlighting the importance of addressing **class imbalance** for fairer and more reliable predictions.


