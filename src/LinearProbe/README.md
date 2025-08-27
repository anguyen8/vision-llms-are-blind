# Summary
The big picture of how this code works is that we extract: 
1) the features from the vision encoder
2) the features after linear projection, when they are understandable by the LLM

Then, we train multiple classifiers on this data to determine if the information needed to solve the task is present at either of these stages.

# Data
We train two models (Phi3.5 and llava-onevision-qwen2-0.5b-si) on two different datasets from "VLMs are Blind" (Touching Circles and Intersecting Lines). We partition the data based on increasing difficulty (closeness of circles or lines). The lines dataset has class labels of `0 - n` and the touching circles dataset has class labels of either `0 or 1`. 

# Code
## extract_features
For Llava, the extraction process is relatively straightforward. There are simple functions to pull out the features before and after projection. 

### Models
For Phi 3.5, this extraction is not super simple. Phi 3.5 does a complex cropping strategy with CLIP that is not well documented. Their codebase is annoying to deal with when you want to extract the features before the cropping and projection. You need to have their code repo downloaded and then import the relevant parts. In comparison, LLaVa is easier to work with.

## train_on_features
The code enables you to train either a Logistic Regression model or a simple linear projection model. Both of these work equally well. 

To train the linear projection model, we use `MLPClassifier` from `sklearn.neural_network`. If you set the hidden sizes of the `MLPClassifier` to be empty, it will train a simple input-output neural network. Don't be concerned by the fact that it is called an MLP even though it is technically not really an MLP due to having zero hidden layers. 

The code is set up to allow you to do a hyperparameter search for the best settings, but that part of the code is currently commented out.