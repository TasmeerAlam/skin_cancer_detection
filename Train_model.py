# import the necessary packages
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import argparse
import pickle
import h5py

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--db", required=True,
    help="path HDF5 database")
ap.add_argument("-m", "--model", required=True,
    help="path to output model")
ap.add_argument("-j", "--jobs", type=int, default=-1,
    help="# of jobs to run when tuning hyperparameters")
args = vars(ap.parse_args())

# open the HDF5 database for reading then determine the index of
# the training and testing split, provided that this data was
# already shuffled *prior* to writing it to disk
db = h5py.File(args["db"], "r")
i = int(db["labels"].shape[0] * 0.75)

# define the set of parameters that we want to tune then start a
# grid search where we evaluate our model for each value of C
print("[INFO] tuning hyperparameters...")
#params = {'loss': [ 'hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'], 'alpha': [0.1, 0.01]}
model = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=True, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
model.fit(db["features"][:i], db["labels"][:i])
#print("[INFO] best hyperparameters: {}".format(model.best_params_))

# evaluate the model
print("[INFO] evaluating...")
preds = model.predict(db["features"][i:])
print(classification_report(db["labels"][i:], preds,
    target_names=db["label_names"]))

# serialize the model to disk
print("[INFO] saving model...")
f = open(args["model"], "wb")
f.write(pickle.dumps(model))
f.close()

# close the database
db.close()