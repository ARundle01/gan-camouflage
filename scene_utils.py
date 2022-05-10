import torch


def gradient_penalty(critic, real, fake, device):
    """
    Calculates gradient penalty for discriminator
    :param critic: discriminator
    :param real: real image
    :param fake: generated image
    :param device: where model is running i.e. CUDA (GPU) or CPU
    :return: gradient penalty value
    """
    # Get shape details of real images
    BATCH_SIZE, CHANNELS, HEIGHT, WIDTH = real.shape

    # Create some random image, eps, and send to computing device
    eps = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, CHANNELS, HEIGHT, WIDTH).to(device)

    # Calculate interpolated image between real, eps and fake image
    interpolated_images = real * eps + fake * (1 - eps)

    # Calculate disc score of interpolated images
    mixed_scores = critic(interpolated_images)

    # Calculate gradient
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]

    # restructure gradient, calculate norm and penalty to norm
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_pen = torch.mean((gradient_norm - 1) ** 2)

    return gradient_pen
