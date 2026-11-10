
# Task 6 - Model Analysis

## 6.1 Overall Accuracy and Class Imbalance
Always-N accuracy = 82.76%.
Our model achieves 97.81% test accuracy.
Overall accuracy alone can be misleading because
the majority class dominates the dataset.

## 6.2 Lowest Performing Class
The F (Fusion) class has the lowest accuracy
at 54.32%. It is most frequently mistaken for
N (Normal), with 54 misclassified samples.

## 6.3 Overfitting
There is a small gap between training and
validation loss. This indicates mild overfitting,
although validation performance remains strong.

## 6.4 Weight Initialization
Zero initialization causes neurons in the same
layer to learn identical features.
He initialization uses sqrt(2/fan-in) to help
maintain suitable variance with ReLU activation.

## 6.5 Momentum and Weight Decay
Momentum smooths updates when consecutive
gradients point in opposite directions.
Weight decay gradually reduces weights toward
zero and helps control model complexity.

## 6.6 Improving the Rare F Class
Oversampling the minority F class could provide
more training examples and improve recognition.
This change is suggested but not implemented.
