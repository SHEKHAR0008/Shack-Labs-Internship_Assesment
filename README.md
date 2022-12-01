# Shack-Labs-Internship_Assesment
## Completed the following 2 tasks as part of Shack Labs DS Internship Assignment:

* House Price Prediction
* Matching of Amazon and Flipkart Products

## Libraries and Framework Used.
 * **1). For Task one House Price Prediction Using Different Ml Model**
   * Matplotlib
   * Pandas
   * Numpy
   * Seaborn
   * SKLearn 
   
   * **MODELS USED FOR PREDICTION**
     * *Used 13 Different model for predicion.*
      * 1. LinearRegression
      * 2. DecisionTreeRegressor
      * 3. RandomForestRegressor
      * 4. BaggingRegressor
      * 5. GradientBoostingRegressor
      * 6. ExtraTreesRegressor
      * 7. HistGradientBoostingRegressor 
      * 8. KNeighborsRegressor
      * 9. SVR
      * 10. XGBRegressor
      * 11. LGBMRegressor
      * 12. CatBoostRegressor
      * 13. MLPRegressor
      
 * **2). For task two  Matching the products of amazon and flipkart having the same name.**
   * Matplotlib
   * Pandas
   * Sentence_Transformer
   * Pytorch
   * Util
   
# Task 1 Analysis.
##  Drawbacks of each technique's assumptions used for Task 1.
 * Linear regression -- It assumes a linear relationship between dependent and independent variables in the problem i.e a straight line dependence.
                        It also assumes independence between attributes which is a major drawback of this model.
 * Decision Tree Regressor -- It always need a big dataset in order to produce good results.
                              The chance of overfitting is high in this model.
 * Support vector Regressor -- Compulsory to apply feature scaling . If not applied  gives very poor result.
                               Also it is difficult to understand.
 * Random forest Regressor --  This is model very much suceptible to overfitting.
 * K Nearest Neighbours -- Accuracy depends on the quality of data.
                           Sensitive to irrelevwnt features.
                           Requires large memory as it stores all the training data.
 * XGB And CatBoost Regressor --These Are very sensitive to outliers.
## Final model selection for Task 1
 * RandomForestRegreesor Or ExtraTreeRegressor or Catboostregressor Any one of the three.
   * As We have not so much data 
   * Also our data is not so much linear.
   
   
# Task 2 Analysis
## **`sentence_transformers`**
* SentenceTransformers is a Python framework for state-of-the-art sentence, text and image embeddings
* This framework generates embeddings for each input sentence.
* Sentences are passed as a list of string.

## **`Torch`**
* PyTorch is a Python package that provides two high-level features:
   * Tensor computation (like NumPy) with strong GPU acceleration.
   * Deep neural networks built on a tape-based autograd system.
## **`all-MiniLM-L6-v2`**
* This is a sentence-transformers model: It maps sentences & paragraphs to a 384 dimensional dense vector space and can be used for tasks like clustering or semantic search.

## **`cosine similarity`**
* It is a measure of similarity between two sequences of numbers.

## **`convert_to_tensor`**
* This is set true so as to convert the product names into numerical form .
