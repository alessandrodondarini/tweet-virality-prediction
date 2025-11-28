# Tweet Virality Prediction

Data analysis of social networks remains one of the most compelling fields for applying statistical tools and machine learning models, offering the practical advantage of predicting trends and identifying behavioral patterns.  

Predicting viral trends is particularly valuable in contexts where companies and advertisers aim to **optimize ad diffusion strategies** by investing in the most shareable and engaging posts.  

For these reasons, this project focuses on **Twitter data** (now *X*), taken from the [Codecademy platform](https://www.codecademy.com) (course: *Build a Machine Learning Model*), and implements several machine learning models to predict whether a tweet becomes viral.




### Project Overview

Using a **quartile-based division** of the retweet count ditribution, we classify tweet virality into four categories:  
`unpopular`, `average`, `popular`, and `super popular`.  

We then perform exploratory data analysis by plotting the distributions of relevant features, and conduct feature engineering to uncover potential hidden patterns in the tweets, such as:

- tweet length  
- number of hashtags 
- number of mentions 
- tweet language

Importantly, the dataset naturally exhibits class imbalance due to the highly skewed distribution of retweet counts. Using a quartile-based division reflects this skew, so rather than enforcing artificial class balance, models were trained on the original distribution to maintain realism.

### Models Implemented

One of the main challenges in this analysis is the strong imbalance between virality classes, which can significantly affect evaluation metrics such as the F1-score.  

To address this issue, we implemented three different machine learning models:  
**K-Nearest Neighbor (KNN)**, **Random Forest**, and a **Sequential Neural Network** built with `PyTorch`.  

Each model was trained and evaluated on the same dataset to compare their performance in predicting tweet virality and handling class imbalance.


### Results Summary
![Model Comparison](results/model_comparison.png)
*Figure 1: Comparison of the performance of the three models. The blue bar shows the overall test accuracy, while the orange and green bars represent the F1-scores for the majority (`unpopular`) and minority (`average`) classes, respectively. The red and purple bars correspond to the F1-scores for the `popular` and `super-popular` classes. The baseline performance, obtained by randomly selecting a category, is indicated by the horizontal dashed line at 25%.*



- The **Neural Network** achieved **51% accuracy**, performing roughly twice as well as random selection (25% accuracy given the four classes).   
- However, its **F1-score for the minority class** (`average`) remained low at **12%**, indicating room for improvement.  
- The **KNN** model reached **49% accuracy**, but a higher **F1-score (24%)** for the minority class, suggesting a mild improvement in handling minority classes.
- The **Random Forest** model achieved the highest accuracy at **53%**, although its **F1-score for the minority class** remained slightly lower than that of **KNN** (**21%** vs. **24%**).
 

Although the test accuracies of the three models are comparable and the F1-scores for the majority class (`unpopular`) remain relatively high (around **75%**), the consistently low F1-scores for the `average` class reveal that better handling of minority categories could lead to significant performance improvements across all evaluation metrics.


### Conclusions

This project demonstrates the potential of machine learning models to analyze social media data and predict viral trends using a minimal set of features. It also highlights the importance of properly addressing **class imbalance** to achieve more reliable and generalizable predictions.  

As a natural next step, applying techniques such as **resampling** or **class-weight adjustments** could help mitigate imbalance effects and further improve model performance.  

The key takeaway of the project is the importance of enhancing the model’s ability to detect minority classes to more effectively identify high-potential content and optimize ad diffusion strategies.



