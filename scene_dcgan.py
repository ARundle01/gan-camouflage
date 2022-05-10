import torch
import torch.nn as nn
from torchsummary import summary


class Discriminator(nn.Module):
    """
    Class for DCGAN discriminator, is a subclass of nn.Module.

    Constructed of one input layer, three block layers and one output layer
    """
    def __init__(self, channels_img, features_d):
        super(Discriminator, self).__init__()  # Initialise nn.Module methods and vars
        # Use a sequential container to run layer after layer easily
        self.disc = nn.Sequential(
            # Input: n x channels_img x 64 x 64
            nn.Conv2d(
                channels_img,
                features_d,
                kernel_size=4,
                stride=2,
                padding=1,
            ),  # new shape: 32x32
            nn.LeakyReLU(0.2),  # no batchnorm in input layer
            self._block(features_d, features_d*2, 4, 2, 1),  # new shape: 16x16
            self._block(features_d*2, features_d*4, 4, 2, 1),  # new shape: 8x8
            self._block(features_d*4, features_d*8, 4, 2, 1),  # new shape: 4x4
            nn.Conv2d(features_d*8, 1, kernel_size=4, stride=2, padding=0),  # new shape: 1x1, no batchnorm or leakrelu
            nn.Sigmoid(),  # flatten to single value in range [0, 1]
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        """
        A single layer block, containing all relevant layer functions for easy repetition
        :param in_channels: number of input features
        :param out_channels: number of output features
        :param kernel_size: kernel size (4x4)
        :param stride: stride to use (2)
        :param padding: padding (1)
        :return: sequential container containing functions for this layer block
        """
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        """
        Defines forward computation at every call
        :param x: input to discriminator
        :return: product of computation by disc
        """
        return self.disc(x)


class Generator(nn.Module):
    """
    Class for DCGAN generator, is a subclass of nn.Module.

    Constructed of four block layers and one output layer
    """
    def __init__(self, z_dim, channels_img, features_g):
        super(Generator, self).__init__()  # Initialise nn.Module methods and vars
        # Use a sequential container to run layer after layer easily
        self.gen = nn.Sequential(
            # Input: n x z_dim x 1 x 1
            self._block(z_dim, features_g*16, 4, 1, 0),  # 4x4
            self._block(features_g*16, features_g*8, 4, 2, 1),  # 8x8
            self._block(features_g*8, features_g*4, 4, 2, 1),  # 16x16
            self._block(features_g*4, features_g*2, 4, 2, 1),  # 32x32
            nn.ConvTranspose2d(
                features_g*2,
                channels_img,
                kernel_size=4,
                stride=2,
                padding=1,
            ),  # 64x64
            nn.Tanh(),  # [-1, 1]
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        """
        A single layer block, containing all relevant layer functions for easy repetition
        :param in_channels: number of input features
        :param out_channels: number of output features
        :param kernel_size: kernel size (4x4)
        :param stride: stride to use (2)
        :param padding: padding (1)
        :return: sequential container containing functions for this layer block
        """
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        """
        Defines forward computation at every call
        :param x: input to generator
        :return: product of computation by gen
        """
        return self.gen(x)


def initialise_weights(model):
    """
    Initialises weights for CNN in Disc and Gen
    :param model: disc or gen
    """
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


def test():
    """
    Simple test function - shows shape summary of both networks
    """
    batch_size, in_channels, height, width = 8, 3, 64, 64
    z_dim = 100
    x = torch.randn((batch_size, in_channels, height, width))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    disc = Discriminator(in_channels, width).to(device)
    initialise_weights(disc)

    summary(disc, (3, 64, 64))

    gen = Generator(100, in_channels, width).to(device)
    initialise_weights(gen)

    summary(gen, (100, 1, 1))


if __name__ == '__main__':
    test()
