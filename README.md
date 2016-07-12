# LearnedNormPooling
Attempt at an implementation of http://arxiv.org/pdf/1311.1780v7.pdf in Theano/Lasagne.

## How to run
Make sure you have [Lasgane and Theano](http://lasagne.readthedocs.io/en/latest/user/installation.html) properly installed. 
Then: 

``` python mnist_test.py ```

## TODO
Scan function/symbolic tensor functions are not optimized! Takes way too long on a CPU. Next step, try one of the following: 
- Refactor code to work in GPU, test runtime
- Consider alternative implementation that overrides the Pool and PoolGrad op classes. 
