# Multilayer Authenticity Identifier (MAI)

MAI is a research project that attempts to train a CNN model to identify synthetic AI images.

## Why?

i am bored.

## Architecture

nothing is set in stone, but at the moment, MAI is a simple CNN model that looks like this:

1. 16-channel, 3x3 convolution layer -> 2x2 max pooling -> relu activation
2. 32-channel, 3x3 convolution layer -> 2x2 max pooling -> relu activation
3. 64-channel, 3x3 convolution layer -> 2x2 max pooling -> relu activation
4. 40,000-neuron layer -> relu -> 120-neuron layer -> relu -> 30 -> 1

the model expects a 200x200 image as an input and outputs a score, with 1 being that the input image is absolutely synthetic, and 0 being that it is absolutely authentic.

[BCEWithLogitLoss](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html) is used as the loss fn, and [RMSprop](https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html) as the optimizer.

## Datasets

MAI has been trained on the following datatsets:

- [poloclub/diffusiondb](https://huggingface.co/datasets/poloclub/diffusiondb)
- [nlphuji/flickr30k](https://huggingface.co/datasets/nlphuji/flickr30k)
- [keremberke/painting-style-classification](https://huggingface.co/datasets/keremberke/painting-style-classification)
- [animelover/scenery-images](https://huggingface.co/datasets/animelover/scenery-images)
- [nanxstats/movie-poster-5k](https://huggingface.co/datasets/nanxstats/movie-poster-5k)
- [Alphonsce/metal_album_covers](https://huggingface.co/datasets/Alphonsce/metal_album_covers)

## How to train?

make sure to have [poetry](https://python-poetry.org) installed.

clone the project, and run:

```
poetry install
```

open a shell in the venv created by poetry:

```
poetry shell
```

run `train.py` to train the model. make sure cuda is available as a cuda-enabled gpu is used to accelerate training. for each epoch, if the validation loss is less than the last epoch, the model is saved locally. you can customize the location easily in `train.py`.

## How to run inference?

run `inference.py` instead. place your test images in `test_images/` directory, and don't forget to reference the images in `inference.py`.

## More on modal.com

i am using (https://modal.com) to run the training and inference, but u can get rid of the modal.com glue pretty easily. you should first remove the decorators above the functions, then at where the functions are invoked, remove `.remote()` and instead invoke the function directly. remove `app` and `vol` variables as well.

