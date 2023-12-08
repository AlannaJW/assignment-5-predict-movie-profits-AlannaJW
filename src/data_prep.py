__author__ =  "Alanna Wroten"
__copyright__ = "Copyright 2023, Westmont College, Alanna Wroten"
__credits__ = ["Alanna Wroten"]
__license__ = "MIT"
__email__ = "awroten@westmont.edu"

import numpy as np
from logistic_models import logistic_all_classes, logistic_edm_rap
def all_classes_prep(f):
    """Modifies the data to be used by the logistic model that predicts genre from all five genres.
    Seperates the genre attribute column to be modified and used to train the logistic model later on.
    Converts some string features to unique integers."""

    # creates dictionaries and counters so that each unique value of chosen features recieves a unique integer
    track_artist = {}
    artist_counter = 0
    track_album = {}
    album_counter = 0
    pl_n = {}
    pl_n_counter = 0
    sub_g = {}
    sub_g_counter = 0
    y_values = []

    # iterates through each row in the array
    # creates list of y values to be passed separately into the logistic model
    # reassigns select string features an integer for unique string values
    # Note: this reassignment does not guarantee its use in the model, just its availability
    for i in range(len(f)):
        y_values.append([f[i][9]])

        row = f[i][2] # track_artist feature
        if row not in track_artist.keys():
            track_artist[row] = artist_counter
            artist_counter += 1
        f[i][2] = track_artist[row]

        row = f[i][5] # track_album feature
        if row not in track_album.keys():
            track_album[row] = album_counter
            album_counter += 1
        f[i][5] = track_album[row]

        row = f[i][7] # playlist_name feature
        if row not in pl_n.keys():
            pl_n[row] = pl_n_counter
            pl_n_counter += 1
        f[i][7] = pl_n[row]

        row = f[i][10] # sub_genre feature
        if row not in sub_g.keys():
            sub_g[row] = sub_g_counter
            sub_g_counter += 1
        f[i][10] = sub_g[row]

    ## Choose which attributes not to keep
    counter = 0
    for i in range(23):
        if i in [0, 1, 4, 6, 8, 9, 10, 12, 13, 14, 15, 20]:
            f = np.delete(f, i - counter, 1)
            counter += 1
    """
    attribute : index
    track_id : 0, track_name : 1, track_artist : 2, track_popularity : 3, track_album_id : 4,
    track_album_name : 5, track_album_release_date : 6, playlist_name : 7, playlist_id : 8,
    playlist_genre : 9, playlist_subgenre : 10, danceability : 11, energy : 12, key : 13,
    loudness : 14, mode : 15, speechiness: 16, acousticness : 17, instumentalness : 18,
    liveness : 19, valence : 20, tempo : 21, duration : 22 

    Examples of combinations and their accuracy
    Keep : 2, 3, 5, 7, 21, 16, 11 --> 0.6110366670727251
    Keep : 2, 3, 5, 7, 11, 12, 14, 16, 17, 18, 19, 21, 22 --> 0.612985747350469
    Keep : 2, 3, 5, 7, 11, 16, 17, 18, 19, 21, 22 --> 0.6306492873675235
    Keep : 2, 3, 5, 7, 10, 11, 12, 14, 16, 17, 18, 19, 21, 22 --> 0.6310147399196004
    """

    # pass y values and total data to the logistic prediction function
    y_array = np.array(y_values)
    logistic_all_classes(f, y_array)

def edm_rap_prep(f):
    """Modifies the data to be used by the logistic model that predicts genre from either edm or rap.
        Takes only songs that are either edm or rap. Seperates the genre attribute column to be modified
        and used to train the logistic model later on.Converts some string features to unique integers."""

    # creates dictionaries and counters so that each unique value of chosen features recieves a unique integer
    track_artist = {}
    artist_counter = 0
    track_album = {}
    album_counter = 0
    pl_n = {}
    pl_n_counter = 0
    sub_g = {}
    sub_g_counter = 0
    y_values = []
    editted_f =[]

    # iterates through each row in the array
    # if the genre of the song is either edm or rap, then
    # creates list of y values to be passed separately into the logistic model
    # reassigns select string features an integer for unique string values
    # Note: this reassignment does not guarantee its use in the model, just its availability
    for i in range(len(f)):
        if f[i][9] == "edm" or f[i][9] == "rap":
            y_values.append([f[i][9]])

            row = f[i][2]
            if row not in track_artist.keys(): # track_artist feature
                track_artist[row] = artist_counter
                artist_counter += 1
            f[i][2] = track_artist[row]

            row = f[i][5]
            if row not in track_album.keys(): # track_album feature
                track_album[row] = album_counter
                album_counter += 1
            f[i][5] = track_album[row]

            row = f[i][7]
            if row not in pl_n.keys(): # playlist_name feature
                pl_n[row] = pl_n_counter
                pl_n_counter += 1
            f[i][7] = pl_n[row]

            row = f[i][10]
            if row not in sub_g.keys(): # subgenre feature
                sub_g[row] = sub_g_counter
                sub_g_counter += 1
            f[i][10] = sub_g[row]

            # creates a new data set with only the edm and rap genres
            editted_f.append(f[i])

    ## Choose which attributes not to keep
    counter = 0
    for i in range(23):
        if i in [0, 1, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 20]:
            editted_f = np.delete(editted_f, i - counter, 1)
            counter += 1
    """
    
    attribute : index
    track_id : 0, track_name : 1, track_artist : 2, track_popularity : 3, track_album_id : 4,
    track_album_name : 5, track_album_release_date : 6, playlist_name : 7, playlist_id : 8,
    playlist_genre : 9, playlist_subgenre : 10, danceability : 11, energy : 12, key : 13,
    loudness : 14, mode : 15, speechiness: 16, acousticness : 17, instumentalness : 18,
    liveness : 19, valence : 20, tempo : 21, duration : 22
    
    Examples of feature combinations and their accuracy
    Keep : 2, 3, 11, 16, 17, 18, 19, 21, 22 --> 0.9016282225237449
    Keep : 2, 3, 5, 11, 16, 17, 18, 19, 21, 22 --> 0.9484396200814111
    Keep : 2, 3, 5, 7, 10, 11, 16, 17, 18, 19, 21, 22 --> 0.9955902306648575
    """

    # pass y values and total data to the logistic prediction function
    y_array = np.array(y_values)
    logistic_edm_rap(editted_f, y_array)