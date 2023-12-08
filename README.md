# Logisitic Regression Based Song Genre Classification

### [Link to Presentation](https://docs.google.com/presentation/d/1Uf5R8ULaxdBA8AckcTLqVd0WbE3RM_-HcaL8Xlz130A/edit?usp=sharing)

### Alanna Wroten, awroten@westmont.edu

### License : MIT License
Found in LICENSE.txt.

### Structure

This project includes three source files: runner.py, data_prep.py, and logisitic_models.py.
On runner.py is where the filepath to the data is needed. The data, spotify_songs.csv, is stored 
in the data folder. Then, it calls a data preparation function from data_prep.py, which will then 
call a regression function from logisitic_models.py.

In short, data_prep.py functions modify and select the data needed for the regression model 
based on which regression models are contributing to the prediction in logistic_model.py functions.

This project can predict the genre of songs from all five classes (rap, edm, pop, latin, r&b) and 
from just edm and rap.

### Implementation

#### data_prep.py
There is a function respective to the classes included in the classification options. 
The first function is to classify between all five genres and the second function 
is to classify between edm and rap. Both functions prepare the data to be used in the logistic model.
First it makes select features into integers unique to the unique string values. It also stores the 
target, or y, values separately to be modified later. This is also where you choose which
features are used in the model by taking out whole columns of the array. Then both the target values
and the remaining data set is passed to the corresponding logistic_models.py function.

#### logisitic_models.py
There is a function respective to the classes included in the classification options. to all_classes_prep()
corresponds to logistic_all_classes() and edm_rap_prep() corresponds to logistic_edm_rap(). The logistic_models.py
functions split the data into train and test sets. Then, using make_y_binary(), the y value data is 
modified to be binary for each class used so that it can help fit the model. The data is also scaled and
transformed so that no raw value skews the model too much and so that there are no negatives messing up 
model. Using the unique set y values for each class, create a LogisticRegression() and fit it for each class.
Using create_predictions(), make a list of predictions for each instance in the test data. Finally, use 
calc_accuracy() to print the prediction accuracies.

###### make_binary()
Helper function to return a dictionary of the class and binary y training values, since the binary y values 
signify whether or not the song genre is that specific genre.

###### create_prediction()
Helper function to iterate through the test set and return a list of predictions. Predictions are 
made by getting the probability from each logistic regression model and taking the greatest probability.

###### calc_accuracy()
Helper function to cacluate accuracy of predictions list against true classification.

### Corpus Selection and Acknowledgments

For this assignment I am working with the 30000 Spotify Songs data set that I found on Kaggle.
It is a data set of songs with 23 features and a Database Contents License (DbCL) v1.0. It was 
authored by  Charlie Thompson, Josiah Parry, Donal Phipps, and Tom Wolff from the Spotify API.
I downloaded it from [Kaggle.com](https://www.kaggle.com/datasets/joebeachcapital/30000-spotify-songs/data)
and will be using the single csv file with all the data. The csv file has almost 30,000 songs with 5 genres.

#### Permissions
[Database Contents License (DbCL) v1.0](https://opendatacommons.org/licenses/dbcl/1-0/)

"The Licensor grants to You a worldwide, royalty-free, non-exclusive, perpetual,
irrevocable copyright license to do any act that is restricted by copyright over 
anything within the Contents, whether in the original medium or any other. These 
rights explicitly include commercial use, and do not exclude any field of endeavour. 
These rights include, without limitation, the right to sublicense the work."
#### Resources

I did not copy any significant code segments but used these sites to inform me of various methods to 
implement my goal, usually involving numpy arrays and skylearn.
 * [How to Read and Write With CSV Files in Python?](https://www.analyticsvidhya.com/blog/2021/08/python-tutorial-working-with-csv-file-for-data-science/#h-steps-to-read-csv-files-in-python-using-csv-reader)
 * [How to import csv data file into scikit-learn?](https://stackoverflow.com/questions/11023411/how-to-import-csv-data-file-into-scikit-learn) 
 * [Logistic Regression in Python - Preparing Data](https://www.tutorialspoint.com/logistic_regression_in_python/logistic_regression_in_python_preparing_data.htm)
 * [Logistic Regression using Python](https://www.geeksforgeeks.org/ml-logistic-regression-using-python/)
 * [sklearn.linear_model.LogisticRegression¶](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
 * [Delete rows and columns of NumPy ndarray](https://www.geeksforgeeks.org/delete-rows-and-columns-of-numpy-ndarray/)
