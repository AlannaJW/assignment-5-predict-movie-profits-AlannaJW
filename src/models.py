
__author__ = "Alanna Wroten"
__copyright__ = "Copyright 2023, Westmont College, Alanna Wroten"
__credits__ = ["Alanna Wroten"]
__license__ = "MIT"
__email__ = "awroten@westmont.edu"

class Feature():
    def __init__(self, name, value=None):
        if not isInstance(name, str) or name is None or name == "":
            raise ValueError
        self._name = name
        self._value = value

    def name(self) :
        return self._name
    def value(self) :
        return self._value
    def __eq__(self, other) :
        if self is other :
            return True
        elif not isInstance(other, Feature):
            return False
        else :
            return self._name == other.name and self._value == other.value
    def __str__(self):
        return self.__repr()
    def __hash(self):
        return hash((self._name, self._value))

class FeatureSet():
    def __init__(self, feature_set : set[Feature], known_class=None):
        if feature_set is None:
            raise ValueError
        self._feat : set[Feature] = feature_set
        self._clas: str | None = known_class

    def feat(self):
        return self._feat
    def cla(self):
        return self._clas

    def build_features(cls, data, known_class=None) -> FeatureSet:
        """TODO: open file, create features for each chosen feature and instance of data"""
        pass

class ObjectClassifier():
    def __init__(self, trained_fs: Iterable[FeatureSet], probs: {}, class_probs: {}):
        self._fs = trained_fs
        self._probs = probs
        self._class_probs = class_probs

    def gamma(self, a_feature_set:FeatureSet):
        """TODO: use FeatureSet passed in to predict value using previously trained method"""
        pass

    def logistic_train(cls, training_set:Iterable[FeatureSet]) -> ObjectClassifier:
        pass

    def ridge_train(cls, training_set:Iterable[FeatureSet]) -> ObjectClassifier:
        pass

    def linear_train(cls, training_set:Iterable[FeatureSet]) -> ObjectClassifier:
        pass








