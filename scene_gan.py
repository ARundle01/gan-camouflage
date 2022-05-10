import torch.nn as nn


class Discriminator(nn.Module):
    """
    Original GAN Discriminator Network
    """
    def __init__(self, img_dim):
        super(Discriminator, self).__init__()  # Initialise nn.Module methods and attribs
        # Simple sequential model for discriminator
        self.disc = nn.Sequential(
            nn.Linear(img_dim, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        Function to handle a forward step in training
        :param x: image data to supply to the network
        :return: returns measure of realness i.e. is the image real or generated
        """
        return self.disc(x)


class Generator(nn.Module):
    """
    Original GAN Generator Network
    """
    def __init__(self, z_dim, img_dim):
        super(Generator, self).__init__()  # Initialise nn.Module methods and attribs
        # Simple sequential model for generator
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, img_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        """
        Function to handle a forward step in training
        :param x: image data to supply to the network
        :return: returns a generated image
        """
        return self.gen(x)
