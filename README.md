## Introduction
In this project, we are going to explore the gradient contradiction behavior in underfitting model

## Install
1. Python 3.7
2. pip install -r requirement.txt

## Project Structure
```
|_src
  |_ dataset.py: define dataloader 
  |_ model.py: define model
  |_ train_test.py: define training logic
  |_ train_test_neighbour.py: model evolving script
  |_ gradient.py: get gradient dictionary
  |_ Visualize_gradient_contradict.ipynb: gradient contradiction analysis
  |_ vis.py: visualization script

```

## Several observations
- Adam is better than SGD (not sure why)
- BatchNorm does not work well for simple underfitting model (BatchNorm is designed to solve covariance shift problem, thus more relevant to complex overfitting networks)
