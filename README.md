Homemade MNIST neural network written in Java. For Dr. Dub's AI class


### note: training and testing data is assumed to be stored like this:
#### `<current directory>\test\mnist_test.csv`

# Compile and Run
compile: 
```
javac -d . Network\Net.java
javac -d . Network\Layer.java
javac -d . Network\Model.java
javac -d . Main.java
```
run:

```
java Main
```

# Makefile route
```
make
java Main
```

# Options
## 1 - Train the network
* Default set to:
    * Epochs: 30
    * Hidden layer(s): 1 with 30 nodes
    * Batch size: 10 instances
    * Learning rate: 3

## 2 - Load pre-trained network
* accepts filepaths
* type the name without the `.model`
    * this was included so gitignore could ignore these files

## 3 - Training data accuracy
* tests current model on the data it was trained on

## 4 - Testing data accuracy
* tests the current model on data it wasn't trained on

## 5 - Save network state
* saves the model to `<filename>.model`

## 6 - Print misclassified images
* ^this is the goal
* currently prints all the images
