import pickle
import pandas as pd
from xgboost import XGBClassifier

# load data
data = pd.read_excel('input.xlsx', header=1)

# load different models
Adaboost_model = pickle.load(open('AdaBoostClassifier_Model', 'rb'))
XGBoost_model = XGBClassifier()
XGBoost_model.load_model("XGBoost_model.json")
voting_model = pickle.load(open('VotingClassifier_Model', 'rb'))

# prediction
Adaboost_prediction = Adaboost_model.predict(data)
voting_prediction = voting_model.predict(data)
XGBoost_prediction = XGBoost_model.predict(data)

# print the results
print(f'Cell behaviour prediction is: {Adaboost_prediction}')
print(f'Cell behaviour prediction is: {voting_prediction}')
print(f'Cell behaviour prediction is: {XGBoost_prediction}')

