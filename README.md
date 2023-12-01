# 5D Affine Transformation Layer
This TensorFlow layer implements an affine image transformation on 5D image batches, useful for data augmentation and spatial transformations in computer vision models.

## Usage
The layer takes a 5D image batch tensor of shape (batch_size, frames, height, width, channels) and a transformation parameter tensor of shape (batch_size, frames, 6) containing the affine transformation matrices per image.

It transforms the image grids accordingly and returns the transformed 5D batch.

usage: 
standlone function: 

~~~
from affine import AF
images = tf.random.normal(shape=(5, 10, 64, 64, 3))
transforms = tf.random.uniform(shape=(5, 10, 6)) 

affine_layer = AF(name='affine_trans')
transformed = affine_layer([images, transforms])

~~~
you can also use this as any other keras layer in your model: 
~~~
from affine import AF
from tensorflow.keras.layer imoort Input
from tensorflow.keras imoort Model

image_in = Input((batch,frames,h,w,c))
affine_in = Input((batch,frames,6))
affine_layer = AF()([image_in,affine_in])
model = Model([image_in,affine_in],affine_layer)
~~~

Key functions:

~~~
_interpolate: Bilinear sampling on input images
_transform_grid: Affine transformation of the sampling grid
call: Apply grid transformations and sampling per batch
~~~

Resources
Based on Spatial Transformer Networks: https://arxiv.org/abs/1506.02025
