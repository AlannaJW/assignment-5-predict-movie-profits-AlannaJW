
__author__ = "Alanna Wroten"
__copyright__ = "Copyright 2023, Westmont College, Alanna Wroten"
__credits__ = ["Alanna Wroten"]
__license__ = "MIT"
__email__ = "awroten@westmont.edu"

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def logistic_all_classes(data, y_data):
    """Predicts genre of song based on accuracy of separate logistic models for each class."""

    # dictionary that can hold the binary y values for each class/logistic model
    y_class_data = {"r&b" : 0, "pop" : 0, "edm" : 0, "rap" : 0, "latin" : 0}

    x_train, x_test, y_train, y_test = train_test_split(data, y_data, test_size=0.25, random_state=29)

    # modifies y values to be binary for each different class
    y_class_data = make_y_binary(y_class_data, y_train)
    # scales and fits data so it can be used in logistic model
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)

    # creates Logisitic Regression model and fits it for each class, stores in dictionary
    class_models = {"rnb": 0, "pop": 0, "edm": 0, "rap": 0, "latin": 0}
    pop_model = LogisticRegression()
    pop_model.fit(x_train, y_class_data["pop"])
    class_models["pop"] = pop_model
    #__________________________________________
    edm_model = LogisticRegression()
    edm_model.fit(x_train, y_class_data["edm"])
    class_models["edm"] = edm_model
    # __________________________________________
    rap_model = LogisticRegression()
    rap_model.fit(x_train, y_class_data["rap"])
    class_models["rap"] = rap_model
    # __________________________________________
    rnb_model = LogisticRegression()
    rnb_model.fit(x_train, y_class_data["r&b"])
    class_models["rnb"] = rnb_model
    # __________________________________________
    latin_model = LogisticRegression()
    latin_model.fit(x_train, y_class_data["latin"])
    class_models["latin"] = latin_model

    print("Hit all class model predictions")
    # use create_predictions() to get list of singular predictions
    class_pred = {"rnb" : 0, "pop" : 0, "edm" : 0, "rap" : 0, "latin" : 0}
    predictions = create_predictions(x_test, class_pred, class_models)
    print("Done with all class predictions!")

    print("Getting accuracy")
    accuracy = calc_accuracy(predictions, y_test)
    print("Logisitic  All Classes Model Accuracy : " + str(accuracy))

def logistic_edm_rap(data, y_data) :
    """Predicts genre of song based on accuracy of separate logistic models for edm and rap class."""

    # dictionary that can hold the binary y values for each class/logistic model
    y_class_data = {"edm": 0, "rap": 0}

    x_train, x_test, y_train, y_test = train_test_split(data, y_data, test_size=0.25, random_state=29)

    # make y values binary and x values scaled and good for logistic fitting
    y_class_data = make_y_binary(y_class_data, y_train)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)

    class_models = {"edm": 0, "rap": 0}
    edm_model = LogisticRegression()
    edm_model.fit(x_train, y_class_data["edm"])
    class_models["edm"] = edm_model
    # __________________________________________
    rap_model = LogisticRegression()
    rap_model.fit(x_train, y_class_data["rap"])
    class_models["rap"] = rap_model

    print("Hit edm and rap model predictions")
    # use create_predictions() to get list of singular predictions
    class_pred = {"edm": 0, "rap": 0}
    predictions = create_predictions(x_test, class_pred, class_models)
    print("Done with edm and rap predictions!")

    print("Getting accuracy")
    accuracy = calc_accuracy(predictions, y_test)
    print("Logisitic Edm and Rap Model Accuracy : " + str(accuracy))

def make_y_binary(y_class_data : dict, y_train) :
    """Converts y values to binary list for each class."""
    for clas in y_class_data.keys():
        new = []
        for row in range(len(y_train)):
            if y_train[row] == clas:
                new.append(1)
            else:
                new.append(0)
        y_class_data[clas] = new
    return y_class_data

def create_predictions(x_test, class_pred, class_models):
    """Makes a prediction for each song in the test set by comparing probabilities across logistic models."""

    predictions = []
    for row in x_test:
        instance = [row]

        #gets prediction probability from each model
        for clas in class_pred.keys():
            class_pred[clas] = class_models[clas].predict_proba(instance)[:,1]

        pred_list = list(class_pred.values())
        max_pred = max(pred_list)
        prediction = 0

        # gets highest prediction probability and adds class prediction to the list
        for i in class_pred.keys():
            if class_pred[i] == max_pred:
                prediction = i
        predictions.append(prediction)
    return predictions

def calc_accuracy(predictions, y_test):
    """Compares predictions to actual classification/genre of song and computes accuracy."""
    counter = 0
    for i in range(len(y_test)):
        if y_test[i] == predictions[i]:
            counter += 1
    accuracy = counter / (len(y_test))
    return accuracy










