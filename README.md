# MLATE

MLATE is the name of a series of ML-based predictive models that are developed for tissue engineering. 

MLATE_Cardica_Cell_Prediction_V1 is capable to predicts cell behavior on cardiac tissue engineering scaffolds.

## usage

install XGBoost

```bash
pip install xgboost
```

## Usage
Load the model using pickle and pass it materials, cell line, and fabrication method.

```python
import pickle

filename = 'MLATE_Cardica_Cell_Prediction_V1'
model = pickle.load(open(filename, 'rb'))


```

## cite our paper
You can find our paper online at [link will be provided soon]


