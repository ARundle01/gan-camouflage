import torch
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from scene_wgan import Discriminator, Generator, initialise_weights

# Hyperparams
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARN_RATE = 5e-5
BATCH_SIZE = 64
IMAGE_SIZE = 64
CHANNELS_IMG = 3
Z_DIM = 100
NUM_EPOCHS = 1000
FEATURES_DISC = 64
FEATURES_GEN = 64
SCENE_ROOT = "./scene-dataset"
CRITIC_ITER = 5
WEIGHT_CLIP = 0.01


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

# Create gen and disc, initiliasing weights for both
gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
critic = Discriminator(CHANNELS_IMG, FEATURES_DISC).to(device)
initialise_weights(gen)
initialise_weights(critic)

# Create RMSprop optimizers for gen and disc
opt_gen = optim.RMSprop(gen.parameters(), lr=LEARN_RATE)
opt_critic = optim.RMSprop(critic.parameters(), lr=LEARN_RATE)

# Create some fixed noise
fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)
step = 0

# Create new loss output file
newfile = open(f"./wgan/losses/num_critics {CRITIC_ITER}/epochs_{NUM_EPOCHS}.txt", "w")
newfile.close()

# Set gen and disc to train
gen.train()
critic.train()

# Start main training loop
for epoch in range(NUM_EPOCHS):
    # For each batch of images in dataloader
    for batch_idx, (real, _) in enumerate(loader):
        # Send real images to GPU
        real = real.to(device)

        # Train Critic for CRITIC_ITER number of times
        for _ in range(CRITIC_ITER):
            # Send random noise to GPU
            noise = torch.randn((BATCH_SIZE, Z_DIM, 1, 1)).to(device)
            # Get generated images from generator, using noise
            fake = gen(noise)

            # Get discriminator value for real images
            critic_real = critic(real).reshape(-1)
            # Get discriminator value for generated images
            critic_fake = critic(fake).reshape(-1)

            # Calculate combined disc loss
            loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake))

            # Network training process:
            # Gradients of network reset, backwards pass over weights is done and new weights calculated
            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_critic.step()

            # Clamp weights/params to WEIGHT_CLIP value
            for p in critic.parameters():
                p.data.clamp_(-WEIGHT_CLIP, WEIGHT_CLIP)

        # Train Generator:
        # Get discriminator value for fake image
        output = critic(fake).reshape(-1)
        # Calculate generator loss
        loss_gen = -torch.mean(output)

        # Network training process:
        # Gradients of network reset, backwards pass over weights is done and new weights calculated
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        # At start of each batch:
        if batch_idx % 100 == 0:
            # Print epoch, disc loss and gen loss to console
            print(
                f"Epoch [{epoch+1}/{NUM_EPOCHS}]; Batch {batch_idx}/{len(loader)}"
                f"        Loss D: {loss_critic:.4f}, Loss G: {loss_gen:.4f}"
            )

            # For current generator iteration
            with torch.no_grad():
                # Generate some images from fixed noise
                fake = gen(fixed_noise)
                # Take up to 32 examples
                # Make into image grid (8x4)
                img_grid_real = torchvision.utils.make_grid(
                    real[:32], normalize=True
                )
                img_grid_fake = torchvision.utils.make_grid(
                    fake[:32], normalize=True
                )

                # Open loss text file and write epoch, disc loss and gen loss
                with open(f"./wgan/losses/num_critics {CRITIC_ITER}/epochs_{NUM_EPOCHS}.txt", "a") as output:
                    output.write(f"{epoch+1}, {loss_critic:.4f}, {loss_gen:.4f}\n")

                # Save real and fake image grids
                torchvision.utils.save_image(real[:32], fp=f"./wgan/real/num_critics {CRITIC_ITER}/epochs {NUM_EPOCHS}/wgan_real_e{epoch+1}.png", normalize=True)
                torchvision.utils.save_image(fake[:32], fp=f"./wgan/fake/num_critics {CRITIC_ITER}/epochs {NUM_EPOCHS}/wgan_fake_e{epoch+1}.png", normalize=True)

            # Increment global counter
            step += 1
