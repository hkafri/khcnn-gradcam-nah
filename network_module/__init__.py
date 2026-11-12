"""KHCNN network package.

Contains the physics-informed KHCNN architecture by M. Olivieri
(Olivieri et al., Sensors 2021, doi:10.3390/s21237834):

- ``khcnn_network``: full KHCNN model (CNN + Kirchhoff-Helmholtz propagation)
- ``cnn_models``: encoder/decoder CNN building blocks
- ``kh_discretisation_tf``: discretised Kirchhoff-Helmholtz integral in TensorFlow
"""
