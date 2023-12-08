# Logisitic Regression Based Song Genre Classification

### [Link to Presentation](https://docs.google.com/presentation/d/1Uf5R8ULaxdBA8AckcTLqVd0WbE3RM_-HcaL8Xlz130A/edit?usp=sharing)

### License : 

### Alanna Wroten, nkirk@westmont.edu awroten@westmont.edu

### Structure

This project includes three files: runner.py, data_prep.py, and logisitic_models.py.
On runner.py is where the filepath to the data is needed. Then, it calls a data preparation
function from data_prep.py, which will then call a regression function from logisitic_models.py.
In short, data_prep.py functions modify and select the data needed for the regression model 
based on which regression models are contributing to the prediction in logistic_model.py functions.
This project can predict the genre of songs from all five classes (rap, edm, pop, latin, r&b) and 
from just edm and rap.

### Implementation

#### data_prep.py


#### logisitic_models.py


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

I did not copy any code segments but used these sites to inform me of various methods to 
implement my goal, usually involving numpy arrays and skylearn.
 * [How to Read and Write With CSV Files in Python?](https://www.analyticsvidhya.com/blog/2021/08/python-tutorial-working-with-csv-file-for-data-science/#h-steps-to-read-csv-files-in-python-using-csv-reader)
 * [How to import csv data file into scikit-learn?](https://stackoverflow.com/questions/11023411/how-to-import-csv-data-file-into-scikit-learn) 
 * [Logistic Regression in Python - Preparing Data](https://www.tutorialspoint.com/logistic_regression_in_python/logistic_regression_in_python_preparing_data.htm)
 * [Logistic Regression using Python](https://www.geeksforgeeks.org/ml-logistic-regression-using-python/)
 * [sklearn.linear_model.LogisticRegression¶](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
 * [Delete rows and columns of NumPy ndarray](https://www.geeksforgeeks.org/delete-rows-and-columns-of-numpy-ndarray/)
