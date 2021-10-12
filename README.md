# GaussMixtureProject
Repository for the paper [*Learning Gaussian Mixtures with Generalised Linear Models: Precise Asymptotics in High-dimensions*](https://arxiv.org/abs/2106.03791).

<p float="left">
  <img src="https://github.com/gsicuro/GaussMixtureProject/blob/main/plots/animation_logistic.gif" height="250" />
  <img src="https://github.com/gsicuro/GaussMixtureProject/blob/main/plots/GenErr.jpg" height="250">
  <img src="https://github.com/gsicuro/GaussMixtureProject/blob/main/plots/TrainErr.jpg" height="250">
</p>

*Left: logistic classification of three clusters with ridge regularisation for different values of the regularisation's strength λ. Center and right: test error and training error performing a ridge classification of a mixture of K=3 clusters with diagonal covariance in the high dimensional limit, with thoretical predictions compared with the results of numerical simulations.*

# Structure

In this repository we provide the code and some guided example to help the reader to reproduce the figures of the paper. The repository is structured as follows.

| File                          | Description                                                                                                                                                    |
|-------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ```/multiK/``` | Solver for the fixed point equations of the order parameters in the case of classification tasks on K Gaussian clusters. The notebook ```how_to.ipynb``` provides a step-by-step explanation on how to use the package                                     |
| ```/real_data/mnist_scattering.ipynb``` | Notebook reproducing real-data curves, see Fig. 4 of the paper.  |
| ```/gan_data/synthetic_data_pipeline.ipynb ```         | Notebook explaining pipeline to assign labels for GAN generated data.                                                               |
| ```/gan_data/monte_carlo.ipynb ```         | Notebook explaining how to estimate population covariances for features from GAN generated data.                                                               |
| ```/gan_data/learning_curves.ipynb ```         | Notebook reproducing learning curves for GAN generated data, see Fig. 3 of the paper.                                                              |

The notebooks are self-explanatory. You will also find some auxiliary files such as `simulations.py` in `/real_data` wrapping the code for running the simulations, and `dcgan.py`, `teachers.py`, `teacherutils.py`, `simulate_gan.py` in `/gan_data/` wrapping the different torch models for the pre-trained generators and teachers.

Note that for running the examples in ```/gan_data``` you will need the weights of the generator, the teacher and the covariances. A folder can be downloaded [here](https://drive.google.com/file/d/1XMm5NDFm3Ol0eqLjvgN5XriQcSNtI3ZN/view?usp=sharing) in a single folder ```/data```.

# Reference

[1]: *Learning Gaussian Mixtures with Generalised Linear Models: Precise Asymptotics in High-dimensions*,
B Loureiro, G. Sicuro, C Gerbelot, A. Pacco, F Krzakala, L Zdeborová, [arXiv: 2106.03791](https://arxiv.org/abs/2106.03791) [stat.ML]
