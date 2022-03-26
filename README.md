# ImStack
> A quick and easy way to represent images as a stack of optimizable layers.


Optimizing the pixel values of an image to minimize some loss is common in some applications like style transfer. But because a change to any one pixel doesn't affect much of the image, results are often noisy and slow. By representing an image as a stack of layers at different resolutions, we get parameters that affect a large part of the image (low-res layers) as well as some that can encode fine detail (the high-res layers). There are better ways to do this, but I found myself using this approach enough that I decided to turn it into a proper library. 

TODO: link to demos for different tasks

## Install

This package is available on pypi so install should be as easy as:

`pip install imstack`

## How to use

We create a new image stack like so:

```python
ims = ImStack(n_layers=3)
```

By default, the first layer is 32x32 pixels and each subsequent layer is 2x larger. We can visualize the layers with:

```python
ims.plot_layers()
```


    
![png](docs/images/output_7_0.png)
    


The parameters (pixels) of the layers are set to requires_grad=True, so you can pass the layers to an optimizer with something like `optimizer = optim.Adam(ims.layers, lr=0.1, weight_decay=1e-4)` to modify them based on some loss. Calling the forward pass (`image = ims()`) returns a tensor representation of the combined image, suitable for various pytorch operations. 

For convenience, you can also get a PIL Image for easy viewing with:

```python
ims.to_pil()
```




    
![png](docs/images/output_9_0.png)
    



### Loading images into an ImStack

You don't need to start from scratch - pass in a PIL image and the ImStack will be initialized such that the layers combine to re-create the input image as closely as possible.

```python
from PIL import Image

# Load the input image
input_image = Image.open('demo_image.png')
input_image
```




    
![png](docs/images/output_12_0.png)
    



Note how the lower layers capture broad shapes while the final layer is mostly fine detail.

```python
# Create an image stack with init_image=input_image and plot the layers
ims_w_init = ImStack(n_layers=3, base_size=16, scale=4, out_size=256, init_image=input_image)
ims_w_init.plot_layers()
```


    
![png](docs/images/output_14_0.png)
    


# Examples

Coming soon, examples of this in practice.
