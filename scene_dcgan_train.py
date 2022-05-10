import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from scene_dcgan import Discriminator, Generator, initialise_weights

# Hyperparams
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARN_RATE = 2e-4
BATCH_SIZE = 64
IMAGE_SIZE = 64
CHANNELS_IMG = 3
Z_DIM = 100
NUM_EPOCHS = 1000
FEATURES_DISC = 64
FEATURES_GEN = 64
SCENE_ROOT = "./scene-dataset"

# Resize, transform to tensor and mean-normalize
transforms = transforms.Compose(
    [
        transforms.Resize([IMAGE_SIZE, IMAGE_SIZE]),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(CHANNELS_IMG)],
            [0.5 for _ in range(CHANNELS_IMG)]
        )
    ]
)

# Get dataset and transform
dataset = datasets.ImageFolder(
    root=SCENE_ROOT,
    transform=transforms
)

# Split indexes by category they belong to
forest_idx = [i for i in range(len(dataset)) if dataset.imgs[i][1] == dataset.class_to_idx["forest"]]
building_idx = [i for i in range(len(dataset)) if dataset.imgs[i][1] == dataset.class_to_idx["building"]]
glacier_idx = [i for i in range(len(dataset)) if dataset.imgs[i][1] == dataset.class_to_idx["glacier"]]
mountain_idx = [i for i in range(len(dataset)) if dataset.imgs[i][1] == dataset.class_to_idx["mountain"]]
sea_idx = [i for i in range(len(dataset)) if dataset.imgs[i][1] == dataset.class_to_idx["sea"]]
street_idx = [i for i in range(len(dataset)) if dataset.imgs[i][1] == dataset.class_to_idx["street"]]

# Use indexes found before to create dataloader subsets
forest = torch.utils.data.Subset(dataset, forest_idx)
building = torch.utils.data.Subset(dataset, building_idx)
glacier = torch.utils.data.Subset(dataset, glacier_idx)
mountain = torch.utils.data.Subset(dataset, mountain_idx)
sea = torch.utils.data.Subset(dataset, sea_idx)
street = torch.utils.data.Subset(dataset, street_idx)

# Create dataloader
loader = DataLoader(forest, batch_size=BATCH_SIZE, shuffle=True)

# Create gen and disc, initialising weights for both
gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
disc = Discriminator(CHANNELS_IMG, FEATURES_DISC).to(device)
initialise_weights(gen)
initialise_weights(disc)

# Create Adam optimizers for gen and disc
opt_gen = optim.Adam(gen.parameters(), lr=LEARN_RATE, betas=(0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=LEARN_RATE, betas=(0.5, 0.999))

# Set optimisation criterion to BCE Loss
criterion = nn.BCELoss()

# Create some fixed noise
fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)
step = 0

# Create new loss output file
newfile = open(f"./dcgan/losses/epochs_{NUM_EPOCHS}.txt", "w")
newfile.close()

# Set gen and disc to train
gen.train()
disc.train()

# Start main training loop
for epoch in range(NUM_EPOCHS):
    # For each batch of images in dataloader
    for batch_idx, (real, _) in enumerate(loader):
        # Send real images to GPU
        real = real.to(device)

        # Send random noise to GPU
        noise = torch.randn((BATCH_SIZE, Z_DIM, 1, 1)).to(device)

        # Get generated images from generator, using noise
        fake = gen(noise)

        # Train Disc:
        # Get discriminator value for real images
        disc_real = disc(real).reshape(-1)
        # Calculate loss for real images
        loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))

        # Get discriminator value for generated images
        disc_fake = disc(fake).reshape(-1)
        # Calculate loss for fake images
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))

        # Calculate combined disc loss
        loss_disc = (loss_disc_real + loss_disc_fake) / 2

        # Network training process:
        # Gradients of network reset, backwards pass over weights is done and new weights calculated
        disc.zero_grad()
        loss_disc.backward(retain_graph=True)
        opt_disc.step()

        # Train Gen:
        # Get discriminator value for fake image
        output = disc(fake).reshape(-1)
        # Calculate generator loss
        loss_gen = criterion(output, torch.ones_like(output))

        # Gradients of network reset, backwards pass over weights is done and new weights calculated
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        # At start of each batch:
        if batch_idx % 100 == 0:
            # Print epoch, disc loss and gen loss to console
            print(
                f"Epoch [{epoch+1}/{NUM_EPOCHS}]; Batch {batch_idx}/{len(loader)}"
                f"        Loss D: {loss_disc:.4f}, Loss G: {loss_gen:.4f}"
            )

            # For current generator iteration
            with torch.no_grad():
                # Generate some images from fixed noise
                fake = gen(fixed_noise)

                # Take 32 samples of real and fake images
                # Make into image grid (8x4)
                img_grid_real = torchvision.utils.make_grid(
                    real[:32], normalize=True
                )
                img_grid_fake = torchvision.utils.make_grid(
                    fake[:32], normalize=True
                )

                # Open loss text file and write epoch, disc loss and gen loss
                with open(f"./dcgan/losses/epochs_{NUM_EPOCHS}.txt", "a") as output:
                    output.write(f"{epoch+1}, {loss_disc:.4f}, {loss_gen:.4f}\n")

                # Save real and fake image grids
                torchvision.utils.save_image(real[:32], fp=f"./dcgan/real/epochs {NUM_EPOCHS}/dcgan_real_e{epoch+1}.png", normalize=True)
                torchvision.utils.save_image(fake[:32], fp=f"./dcgan/fake/epochs {NUM_EPOCHS}/dcgan_fake_e{epoch+1}.png", normalize=True)

            # Increment global counter
            step += 1
