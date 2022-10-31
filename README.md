# gan-camouflage

## This is my 3rd Year University project code, as it was too large to submit via the online submission portal

To understand the work the went into the implementations (as well as the underlying science) the literature review and final dissertation are available in `./papers`.

The aim of this project was to research and investigate the use of Generative Adversarial Networks when applied to the generation of camouflage.

## Running the GANs

The dataset is already pre-formatted and as such no changes need to be made to it. Furthermore, running each training script should work when set to `num_epochs` = 1000. Training for 1000 epochs is the same as done in the report, however can take a long time. Hence, the video demonstration [here](https://www.youtube.com/watch?v=g7fM5RAPuqw) does not show training in progress.

The number of epochs and number of critic iterations can be changed, but make sure to create a new folder in the same structure as the already created folders to accomodate for the outputs.

This code leverages the [CUDA framework](https://developer.nvidia.com/cuda-zone) for use on my own graphics card for best performance and as such, you may need to set that up for yourself.


