import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt


# Data Collection
data = pd.read_csv("Heart_Disease_Prediction.csv")
# print(data.head())
# print()
# print(f"Features = {data.columns._data}")

# Acquire X/Y data
X = np.array(data.drop(["Heart Disease"], 1))
Y = np.array(data["Heart Disease"])


# Train RFE model
rfe_model = RFE(RandomForestClassifier())
rfe_model.fit_transform(X=X, y=Y)

# Obtain test/train data
x_train = X[:int(len(X)*.75)]
x_test = X[int(len(X)*.75):]

y_train = Y[:int(len(X)*.75)]
y_test = Y[int(len(X)*.75):]

# Optimization of n_estimators
# list_of_dicts = []
#
# estimator_to_score = {}
#
# # Trial 1
# for i in range(10, 120, 10):
# 	random_forest = RandomForestClassifier(n_estimators=i)
# 	random_forest.fit(x_train, y_train)
# 	estimator_to_score[i] = random_forest.score(x_test, y_test)
#
# list_of_dicts.append(estimator_to_score)
# estimator_to_score = {}
#
# # Trial 2
# for i in range(10, 120, 10):
# 	random_forest = RandomForestClassifier(n_estimators=i)
# 	random_forest.fit(x_train, y_train)
# 	estimator_to_score[i] = random_forest.score(x_test, y_test)
#
# list_of_dicts.append(estimator_to_score)
# estimator_to_score = {}
#
# # Trial 3
# for i in range(10, 120, 10):
# 	random_forest = RandomForestClassifier(n_estimators=i)
# 	random_forest.fit(x_train, y_train)
# 	estimator_to_score[i] = random_forest.score(x_test, y_test)
#
# list_of_dicts.append(estimator_to_score)
# estimator_to_score = {}
#
# # Print out results from each trial
# for i in range(len(list_of_dicts)):
# 	print(list_of_dicts[i])

# plt.xlabel("N_estimators")
# plt.ylabel("Testing Accuracy")
# plt.plot(range(10, 120, 10), scores, c='red')
# plt.show()

# Create optimized Forest Ensemble
forest_ensemble = RandomForestClassifier(n_estimators=30)

# Train & Test Ensemble
forest_ensemble.fit(x_train, y_train)

# Examination of model results
# print(f"Accuracy of forest: {forest_ensemble.score(x_test, y_test)}")
#
# prediction_results = forest_ensemble.predict(x_test)
#
# for i in range(len(prediction_results)):
# 	print(f"Predicted Value: {prediction_results[i]}; Actual value: {y_test[i]}")

# User Prompting

prompt_results = []

# Testing values
# prompt_results = [70,1,4,130,322,0,2,109,0,2.4,2,3,3]

prompt_results.append(int(input("What is your age? ")))
prompt_results.append(int(input("What is your sex? (0 for male, 1 for female) ")))
prompt_results.append(int(input("What kind of chest pains are you feeling? Enter 1 for typical angina, 2 for atypical angina"
                          ", 3 for non-typical anginal pain, or 4 if you're  asymptomatic: ")))
prompt_results.append(int(input("What is your blood pressure? ")))
prompt_results.append(int(input("What is your cholesterol? ")))
prompt_results.append(int(input("Is your fasting blood sugar level over 120 mg/dl? (1 for true and 0 for false): ")))
prompt_results.append(int(input("If any, what is the result of your EKG test?  ")))
prompt_results.append(int(input("When your heart is under stress, what's the highest rate at which it can beat? ")))
prompt_results.append(int(input("Do you experience angina chest pain when you exercise? (1 for yes, 0 for no): ")))
prompt_results.append(int(input("If you experience any levels of S/T Depression, indicate it now: ")))
prompt_results.append(int(input("Please enter the slope of your S/T Depression: (1 for upsloping, 2 for flat, "
                                "and 3 for downsloping) ")))
prompt_results.append(int(input("How many of your major vessels have been colored by fluoroscopy? (0-3) ")))
prompt_results.append(int(input("Please enter the results of any Thallium testing you may have had:"
                                "(3 for normal, 6 for fixed defect, 7 for reversable defect) ")))

prediction_result = forest_ensemble.predict([prompt_results])

print(f"Based on your current conditions, we've predicted that you ", end='')

if prediction_result[0] == "Presence":
    print("may have disease present in your body ")
else:
    print("are disease free! ")

# Beginning of web-scraping

