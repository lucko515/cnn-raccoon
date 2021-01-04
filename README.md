<link rel="stylesheet" type="text/css" media="all" href="images/readme.css" />

# CNN Raccoon v0.9.5

<p align="center">
  <img src="https://raw.githubusercontent.com/lucko515/cnn-raccoon/master/cnn_raccoon/static/images/ui/cnn_logo.png">
</p>

[![Downloads](https://pepy.tech/badge/cnn-raccoon)](https://pepy.tech/project/cnn-raccoon)
   
<h4 style="text-align: center;">Create interactive dashboards for your Convolutional Neural Networks (CNNs) with a single line of code!</h4>

---
__CNN Raccoon__ helps you with inspecting what's going on inside your Convolutional Neural Networks, in visual and easy to understand way.



## How to use it?

### TensorFlow mode

For the clustering example, let's import KMeans and demonstrate how to use it with the ML Tutor library.
Notice that you can train/test it just like any `sklearn` algorithm.

Each algorithm has several arguments you can provide, but the unique one across all of them is `visual_training`. 
If you set this to `True`, you will see the whole training process inside your IDE.

```python
model = tf.keras.models.Sequential([ ... ])
model.compile(...)
# You define and compile model in the same way

# Let's use Cifar-10 for this example, but can be any dataset
from tensorflow.keras.datasets import cifar10

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# CNN Raccoon magic!
from cnn_raccoon import inspector

# In a single line of code send your model to the Inspector
inspector(model=model, images=X_train[:10], number_of_classes=10, engine="keras")
```

![](images/kmeans-vt.gif)

### PyTorch mode

For the clustering example, let's import KMeans and demonstrate how to use it with the ML Tutor library.
Notice that you can train/test it just like any `sklearn` algorithm.

Each algorithm has several arguments you can provide, but the unique one across all of them is `visual_training`. 
If you set this to `True`, you will see the whole training process inside your IDE.

```python
# For PyTorch you define the model in the same way as before
model = Net()

# Load dataset using data loaders
transform = transforms.Compose(
    [ transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)
dataiter = iter(trainloader)
images, labels = dataiter.next()

# CNN Raccoon magic!
from cnn_raccoon import inspector

# In a single line of code send your model to the Inspector
inspector(model=model, images=images, number_of_classes=10, engine="keras")
```

![](images/kmeans-vt.gif)



### Weights inspector

Every algorithm has method `.how_it_works()` which generates a blog post directly inside your IDE.
Every blog is written by somebody from the community, not myself, and in the end, they get a shout out for the great material.
```python
from ml_tutor.classification.knn import KNeighbourClassifier

clf = KNeighbourClassifier()
clf.how_it_works()
```

![](images/th.gif)


### GradCam

If you call `.interview_questions()` on any algorithm, it will generate resources with interview questions for the algorithm.

```python
from ml_tutor.classification.knn import KNeighbourClassifier

clf = KNeighbourClassifier()
clf.interview_questions()
```

![](images/inter-q.png)

### Siliency Maps

Since this is the library for education and not for production, you'll need to learn how to use these algorithms with the battle-tested library `sklearn`. Just call `.sklearn_version()` on any algorithm, and it will generate code for you!

NOTE: For now, this method only works in Jupyter Notebook!

```python
from ml_tutor.classification.knn import KNeighbourClassifier

clf = KNeighbourClassifier()
clf.sklearn_version()
```

![](images/sklearn.gif)


## Installation

### Installation with `pip`

You can install CNN Raccoon directly from the PyPi repository using `pip` (or `pip3`): 

```bash
pip install cnn-raccoon
```

### Manual installation

If you prefer to install it from source:

1. Clone this repository

```bash
git clone https://github.com/lucko515/cnn-raccoon
```

2. Go to the library folder and run

```bash
pip install .
```

### Requirements

#### PyTorch version 

#### TensorFlow version

## TODO

If you want to contribute to the CNN Raccoon, here is what's on the TODO list:

- [ ] Silency Map for the __TensorFlow__ mode
- [ ] Make dashboard responsive on smaller screens (< 1100px)

## Contact

Add me on [LinkedIn](https://www.linkedin.com/in/luka-anicin/)