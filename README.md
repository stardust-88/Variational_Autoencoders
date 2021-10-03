# Variational_Autoencoders (VAEs)

Pytorch implementation of Variational Autoencoders - a popular <b>deep generative probabilistic graphical model.<\b>
<hr>

### Run:
To train:  ```python train.py```<br>
To sample from the saved model: ```python sample.py```

<hr>

### A bit about VAEs:

Variational Autoencoders (VAEs) are true generative models which differ from the regular autoencoders in a sense that a VAE generates a probability distribution 
over the hidden variable z given the input data x, i.e., P(z/X), rather than just learning the deterministic mapping from input data to hidden representation 
as is the case with regular autoencoders. The goal is to learn the probability distribution over the observed variable X, and this is facilitated by learning to 
generate the distribution P(X/z).

