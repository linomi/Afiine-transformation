# 5D Affine Transformation Layer
This TensorFlow layer implements an affine image transformation on 5D image batches, useful for data augmentation and spatial transformations in computer vision models.

## Usage
The layer takes a 5D image batch tensor of shape (batch_size, frames, height, width, channels) and a transformation parameter tensor of shape (batch_size, frames, 6) containing the affine transformation matrices per image.

It transforms the image grids accordingly and returns the transformed 5D batch.

Example:

python

images = tf.random.normal(shape=(5, 10, 64, 64, 3))
transforms = tf.random.uniform(shape=(5, 10, 6)) 

affine_layer = AffineTransformation5D(name='affine_trans')
transformed = affine_layer([images, transforms])
Key functions:

_interpolate: Bilinear sampling on input images
_transform_grid: Affine transformation of the sampling grid
call: Apply grid transformations and sampling per batch
Install
pip install affine-transformer-5D

Resources
Based on Spatial Transformer Networks: https://arxiv.org/abs/1506.02025
