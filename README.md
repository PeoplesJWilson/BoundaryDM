# BoundaryDM: sklearn style library for manifold learning on manifolds with boundary

This readme is primarily meant to direct you to tutortial.ipynb, where you can find instructions on how to use this manifold learning library locally.

In short, all you need to do is clone this repo, and run `pip install -r requirements.txt` from the commandline. The libraries required to use this code are numpy, scipy, and matplotlib, all of which can be installed with pip.

This project contains sklearn style implementations of nonlinear dimensionality reduction techniques [Diffusion Maps](https://www.sciencedirect.com/science/article/pii/S1063520306000546) and [TGL](https://arxiv.org/abs/2110.06988). The latter is an extension of the former developed by me and my advisor. In short, Diffusion Maps is a dimensionality reduction technique where the underlying low dimensional structure of the data is suspected to approximate a manifold, like this:

When the manifold has a boundary, the Diffusion Maps algorithm only captures some information of the underlying structure. Manifolds with boundary look like the following image.

TGL is an alteration on Diffusion Maps which captures the missing structure. For a brief introduction on using the classes contained in this project, see tutorial.ipynb. Detailed comments found in BoundaryDM.py can assist you further.
