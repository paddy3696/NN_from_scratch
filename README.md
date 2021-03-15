# Neural Network from_scratch
# Fundamentals of DL course Assignment - 1 


## Install

This project requires **Python** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org/)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)
- [wandb](https://wandb.ai/site)


## Code
- #### Neural network Class (MyNN)
The neural network class contains all the functions required for training the model. The function **fit** is the training function which contains the neural network pipeline (Forward_prog, compute_loss, back_prop, update_parameters). The activation functions (Sigmoid, tanh, relu) and weights, bias initialization (xavier, normal) are independently defined. 

The one hot encoding is performed to deal with categorial output variable. and it is defined as a seperate function.

The following line of code is an example to define a model using the MyNN class:
```python
model = MyNN(network_size=layers,network_fns=act,batch_size = 64,
             optimizer='NADAM',regularize= 'l2',alpha = 0, wb_init = 'xavier_uniform',
             learning_rate = 1e-3, max_epoch=5,verbose=1,seed=25)
```
After defining the model, the training of the model can be done using the following command:
```python
model.fit(X,Y,x_valid,y_valid)
```

- #### Wandb configuration
Wandb sweep 

- #### train sweep function
- #### Testing

## Run

In a terminal or command window, navigate to the top-level project directory `NN_from_scratch/` (that contains this README) and run one of the following commands:

```bash
ipython notebook MNIST_Classification.ipynb
```  
or
```bash
jupyter notebook MNIST_Classification.ipynb
```
## Data
The MNIST fashion dataset is downloaded directly from the Keras library using the following the command:
```python
from keras.datasets import fashion_mnist
```
### Data Preprocessing
- The MNIST fashion dataset contains 80k image data. The data is split in the ratio of 90:10 for training and testing respectively.
- The training data is very split with ratio of 90:10 for training and validation. This is done to avoid Overfitting.
- All the test, train and validation data are normalized for better performance of the neural network
