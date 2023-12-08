__author__ =  "Alanna Wroten"
__copyright__ = "Copyright 2023, Westmont College, Alanna Wroten"
__credits__ = ["Alanna Wroten"]
__license__ = "MIT"
__email__ = "awroten@westmont.edu"

import pandas as pd
import numpy as np
from data_prep import all_classes_prep, edm_rap_prep

def main():
    """Takes a file path and runs two different logistic model predictions on the data. Predicts the genre of the song
     from factors chosen in the data_prep file and runs the predictions in the logistic_models file."""

    # opens csv files and reads the data into a numpy array
    file = "/home/awroten/assignment-5-predict-movie-profits-AlannaJW/data/spotify_songs.csv"
    f = pd.read_csv(file, header=0)
    f = np.array(f)

    # passes the numpy array into the data preparation functions to be modified and then passed into the respective
    ## logistic model functions
    all_classes_prep(f)
    edm_rap_prep(f)

main()

