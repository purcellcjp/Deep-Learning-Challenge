# Deep Learning Challenge: Charity Funding Predictor
## Solution
Solution to predict whether or not applicants for Charity Funding will be successful based the process of Deep Learning.



## Background

The non-profit foundation Alphabet Soup wants to create an algorithm to predict whether or not applicants for funding will be successful. With your knowledge of machine learning and neural networks, you’ll use the features in the provided dataset to create a binary classifier that is capable of predicting whether applicants will be successful if funded by Alphabet Soup.

From Alphabet Soup’s business team, you have received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as the following:

* **EIN** and **NAME**—Identification columns
* **APPLICATION_TYPE**—Alphabet Soup application type
* **AFFILIATION**—Affiliated sector of industry
* **CLASSIFICATION**—Government organization classification
* **USE_CASE**—Use case for funding
* **ORGANIZATION**—Organization type
* **STATUS**—Active status
* **INCOME_AMT**—Income classification
* **SPECIAL_CONSIDERATIONS**—Special consideration for application
* **ASK_AMT**—Funding amount requested
* **IS_SUCCESSFUL**—Was the money used effectively

## Instructions

### Step 1: Preprocess the data

Using your knowledge of Pandas and the Scikit-Learn’s `StandardScaler()`, you’ll need to preprocess the dataset in order to compile, train, and evaluate the neural network model later in Step 2

Using the information we have provided in the starter code, follow the instructions to complete the preprocessing steps.

1. Read in the charity_data.csv to a Pandas DataFrame, and be sure to identify the following in your dataset:
  * What variable(s) are considered the target(s) for your model?
  * What variable(s) are considered the feature(s) for your model?
2. Drop the `EIN` and `NAME` columns.
3. Determine the number of unique values for each column.
4. For those columns that have more than 10 unique values, determine the number of data points for each unique value.
6. Use the number of data points for each unique value to pick a cutoff point to bin "rare" categorical variables together in a new value, `Other`, and then check if the binning was successful.
7. Use `pd.get_dummies()` to encode categorical variables

### Step 2: Compile, Train, and Evaluate the Model

Using your knowledge of TensorFlow, you’ll design a neural network, or deep learning model, to create a binary classification model that can predict if an Alphabet Soup–funded organization will be successful based on the features in the dataset. You’ll need to think about how many inputs there are before determining the number of neurons and layers in your model. Once you’ve completed that step, you’ll compile, train, and evaluate your binary classification model to calculate the model’s loss and accuracy.

1. Continue using the jupter notebook where you’ve already performed the preprocessing steps from Step 1.
2. Create a neural network model by assigning the number of input features and nodes for each layer using Tensorflow Keras.
3. Create the first hidden layer and choose an appropriate activation function.
4. If necessary, add a second hidden layer with an appropriate activation function.
5. Create an output layer with an appropriate activation function.
6. Check the structure of the model.
7. Compile and train the model.
8. Create a callback that saves the model's weights every 5 epochs.
9. Evaluate the model using the test data to determine the loss and accuracy.
10. Save and export your results to an HDF5 file, and name it `AlphabetSoupCharity.h5`.

### Step 3: Optimize the Model

Using your knowledge of TensorFlow, optimize your model in order to achieve a target predictive accuracy higher than 75%. If you can't achieve an accuracy higher than 75%, you'll need to make at least three attempts to do so.

Optimize your model in order to achieve a target predictive accuracy higher than 75% by using any or all of the following:

* Adjusting the input data to ensure that there are no variables or outliers that are causing confusion in the model, such as:
  * Dropping more or fewer columns.
  * Creating more bins for rare occurrences in columns.
  * Increasing or decreasing the number of values for each bin.
* Adding more neurons to a hidden layer.
* Adding more hidden layers.
* Using different activation functions for the hidden layers.
* Adding or reducing the number of epochs to the training regimen.

**NOTE**: You will not lose points if your model does not achieve target performance, as long as you make three attempts at optimizing the model in your jupyter notebook.

1. Create a new Jupyter Notebook file and name it `AlphabetSoupCharity_Optimzation.ipynb`.
2. Import your dependencies, and read in the `charity_data.csv` to a Pandas DataFrame.
3. Preprocess the dataset like you did in Step 1, taking into account any modifications to optimize the model.
4. Design a neural network model, taking into account any modifications that will optimize the model to achieve higher than 75% accuracy.
5. Save and export your results to an HDF5 file, and name it `AlphabetSoupCharity_Optimization.h5`.

### Step 4: Write a Report on the Neural Network Model

For this part of the assignment, you’ll write a report on the performance of the deep learning model you created for
Alphabet Soup.
The report should contain the following:

1. Overview of the analysis: Explain the purpose of this analysis.
2. Results: Using bulleted lists and images to support your answers, address the following questions:
	* Data Preprocessing
		* What variable(s) are the target(s) for your model?
		* What variable(s) are the features for your model?
		* What variable(s) should be removed from the input data because they are neither targets nor features?
	* Compiling, Training, and Evaluating the Model
		* How many neurons, layers, and activation functions did you select for your neural network model, and why?
		* Were you able to achieve the target model performance?
		* What steps did you take in your attempts to increase model performance?
3. Summary: Summarize the overall results of the deep learning model. Include a recommendation for how a different model could solve this classi�cation problem, and then explain your recommendation.

----


## <a id="Final-Report"></a>

[Jupyter Notebook Initial Model](AlphabetSoupCharityInitial.ipynb)
[Jupyter Notebook Opimization 2](AlphabetSoupCharityOptimization2.ipynb)
[Jupyter Notebook optimization 3](AlphabetSoupCharityOptimization3.ipynb)

1. The purpose of the model was to create a machine learning model to assist Alphabet Soup in predicting whether or not an applicant's funding will be successful. The machine learning model was a binary classifier that predicted the success rate fairly accurately.

2. **Results**: Using bulleted lists and images to support your answers, address the following questions.

  * Data Preprocessing

    * What variable(s) are considered the target(s) for your model?
		* The Target variable is column `IS_SUCCESSFUL`.

    * What variable(s) are considered to be the features for your model?
		* The following columns were features for the model:
			* `NAME`
			* `APPLICATION_TYPE`
			* `AFFILIATION`
			* `CLASSIFICATION`
			* `USE_CASE`
			* `ORGANIZATION`
			* `STATUS`
			* `INCOME_AMT`
			* `SPECIAL_CONSIDERATIONS`
			* `ASK_AMT`
    
	* What variable(s) are neither targets nor features, and should be removed from the input data?
    
		* The Employer Idenification Number (EIN) can be removed from the dataset as a feature, since it's an assigned attribute that has no bearing of venture success.
    
  * Compiling, Training, and Evaluating the Model
  
    * How many neurons, layers, and activation functions did you select for your neural network model, and why?
	
	
	| Model          | Layer     | Activation | Neurons | Epochs | Loss   | Accuracy |
	| -------------- | --------- | ---------- | ------- | ------ | ------ | -------- |
	| Initial        | 1         | relu       | 45      | 100    | 0.5554 | 0.7294   |
	|                | 2         | relu       | 45      | 		 |        |          |
	| Optimization 1 | 1         | relu       | 11      | 100	 | 0.5551 | 0.7301   |
	|                | 2         | relu       | 26      | 		 |        |          |
	|                | 3         | relu       | 11      | 		 |        |          |	
	| Optimization 3 | 1         | relu       | 24      | 100	 | 0.4503 | 0.7924   |
	|                | 2         | relu       | 36      | 		 |        |          |
	|                | 3         | tanh       | 24      | 		 |        |          |
	
		* In the initial model, I used two layers to get a base line accuracy 72.9% of the model. I then added a third layer, in Optimization 1, and received a slight increace in accuracy to 73.0%. This minimal improvement in accuracy inicates that the feature set needs to examined for further optimization.
		
		* The epochs were all set to 100 to be consistent, since I did not observe any significant improvement in increasing the value while testing.    
    
    * Were you able to achieve the target model performance?
    
		* I was able to increase the accuracy from **72.9%** to **79.2%**.
    
    * What steps did you take to try and increase model performance?
	
		* By reexamining the feature set I was able to improve the accuracy of the model. In this case, `NAME` was retained and `EIN' was dropped. 
		* Plugged in different values of neurons and activation functions
	
   
3. **Summary**: Summarize the overall results of the deep learning model. Include a recommendation for how a different model could solve this classification problem, and explain your recommendation.

    **Summary and Recommendation**
		* The final model was able to predict the accuracy of the venture to almost 80% of the time. The key attibutes of a successful applicant are:
			* `NAME` occurs more than 5 occurences
			* `APPLICATION` is classified with greater than 500 occurences.
			* `CLASSIFICATION` codes have more than 500 occurences.
	
	
    **Alternative Method**
      * An alternate method to solve this problem would to use the Random Forest model. This algorithm could break down the dataset into smaller, simplier decision trees for further optimization.