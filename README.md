# Salary
Demo of training and deploying a simple linear regression model built with Scikit-learn

Usage:

```
pip install -r requirements.txt
python train/train.py -i ./data/sal.csv -o ./model/salary
python deploy/infer.py -m ./model/salary/model.pkl
curl -X POST -H "Content-type: application/json" -d "{\"exp\":\"25\"}" http://localhost:8080/predict
```
