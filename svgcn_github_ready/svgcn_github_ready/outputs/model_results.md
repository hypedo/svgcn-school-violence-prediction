# Model Results

## Baseline

- 발생빈도수 평균: 4.44375
- Baseline MAE: 2.8816

## SVGCN Test Result

| Metric | Value |
|---|---:|
| Test MAE | 1.9983 |
| Test RMSE | 2.5895 |
| Test R² | 0.3559 |
| Risk Accuracy | 0.7500 |

## Confusion Matrix

```text
[[ 3  1  1]
 [ 0  5  1]
 [ 0  3 10]]
```

## Classification Report

```text
              precision    recall  f1-score   support

저위험             1.00      0.60      0.75         5
중위험             0.56      0.83      0.67         6
고위험             0.83      0.77      0.80        13

accuracy                               0.75        24
macro avg          0.80      0.73      0.74        24
weighted avg       0.80      0.75      0.76        24
```
