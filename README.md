# Linear Graph Convolutional Model for Diagnosing Brain Disorders

## Requirements

* [Python](https://www.python.org/downloads/) - version 3.5+
* [PyTorch](https://pytorch.org/get-started/locally/)

## How to run

* Clone the forked SGC repository from (https://github.com/code-anonymous-submission/SGC) into the main folder of this project. 

* Run a python script `train.py`. Additionally, you can run one of the baseline models by passing one of the parameters (*original*, *graph_no_features*, *graph_random*, *graph_identity*) for graph_type: 

	python train.py --graph-type [*graph-type*]
	
  
