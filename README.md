# Neural Network from_scratch
# Fundamentals of DL course Assignment - 1 


### Install

This project requires **Python** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org/)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)
- [wandb](https://wandb.ai/site)


### Code
- #### Neural network Class (MyNN)



- #### One hot encoder
- #### Wandb configuration
- #### train sweep function
- #### Testing

### Run

In a terminal or command window, navigate to the top-level project directory `NN_from_scratch/` (that contains this README) and run one of the following commands:

```bash
ipython notebook MNIST_Classification.ipynb
```  
or
```bash
jupyter notebook MNIST_Classification.ipynb
```
### Data
The MNIST fashion dataset is downloaded directly from the Keras library using the following the command:
```python
from keras.datasets import fashion_mnist
```
#### Data Preprocessing
- The MNIST fashion dataset contains 80k image data. The data is split in the ratio of 90:10 for training and testing respectively.
- The training data is very split with ratio of 90:10 for training and validation. This is done to avoid Overfitting.
- All the test, train and validation data are normalized for better performance of the neural network
