.. AiSpace documentation master file, created by
   sphinx-quickstart on Sun Feb  2 16:11:47 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to AiSpace's documentation!
===================================

AiSpace provides highly configurable framework for deep learning model development, deployment and
conveniently use of pre-trained models (bert, albert, opt, etc.).

Features
-----------------

* Highly configurable, we manage all hyperparameters with inheritable Configuration files.
* All modules are registerable, including models, dataset, losses, optimizers, metrics, etc.
* Standardized process
* Multi-GPU Training
* Integrate multiple pre-trained models, including chinese
* Simple and fast deployment using [BentoML](https://github.com/bentoml/BentoML)
* Integrated Chinese benchmarks [CLUE](https://github.com/CLUEbenchmark/CLUE)

.. toctree::
   :maxdepth: 2
   :caption: Notes:

   quickstart
   config
   deployment
   examples



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
