import matplotlib
import pandas as pd
import skimage.io as io
import skimage.transform
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import cm
from matplotlib.colors import ListedColormap

from sewar.full_ref import uqi, mse

matplotlib.use('Qt5Agg')

_GAN_TYPES = ["gan", "dcgan", "wgan", "wgan-gp"]
_EPOCHS = [1, 10, 25, 50, 100, 250, 500, 1000]
_GAN_RESULTS = [2, 4, 3, 8, 5, 7, 6, 1]
_DCGAN_RESULTS = [9, 11, 10, 13, 12, 15, 14, 16]
_WGAN_RESULTS = [17, 19, 18, 21, 20, 23, 22, 24]
_WGAN_GP_RESULTS = [25, 27, 26, 29, 28, 31, 30, 32]
_REAL_RESULTS = [33, 35, 34, 37, 36, 39, 38, 40, 43, 41, 42, 45, 44, 46, 50, 47, 49, 48]
_SPOT = ["Immediately Spotted", "Somewhat Easy to Spot", "Somewhat Difficult to Spot", "Did not Spot Camouflage"]
_BLEND = ["Very Poorly", "Poorly", "Greatly", "Very Greatly", "There is no Camouflage"]


def create_overlay_grid(real_fname, fake_fname, result_fname, dpi=1200, hide_axes=False, test=False):
    """
    Takes slices from generated image grid and places it over real image grid to create camouflage.

    :param real_fname: file name for real image grid
    :param fake_fname: file name for generated image grid
    :param result_fname: file name for saving the result image grid
    :param dpi: dpi to save image as
    :param hide_axes: whether to hide plot axes
    :param test: if true, save in-process images as well
    """
    # Import real and fake images
    real_image = io.imread(fname=real_fname)
    fake_image = io.imread(fname=fake_fname)

    if test:
        plt.imshow(real_image)
        if hide_axes:
            plt.axis("off")

        plt.savefig("real-pre-squares.png", dpi=dpi, bbox_inches="tight", pad_inches=0)

    # extract all images from real grid
    for row in range(4):
        for col in range(8):
            x1 = 18 + (row * 66)
            y1 = 18 + (col * 66)
            x2 = 50 + (row * 66)
            y2 = 50 + (col * 66)

            real_image[x1:x2, y1:y2, :] = 255

    if test:
        plt.imshow(real_image)
        if hide_axes:
            plt.axis("off")

        plt.savefig("real-post-squares.png", dpi=dpi, bbox_inches="tight", pad_inches=0)

        plt.imshow(fake_image)
        if hide_axes:
            plt.axis("off")

        plt.savefig("fake-pre-slice.png", dpi=dpi, bbox_inches="tight", pad_inches=0)

    # Extract 32x32 section of each image from fake grid and place over the same area on real grid
    for row in range(4):
        for col in range(8):
            x1 = 18 + (row * 66)
            y1 = 18 + (col * 66)
            x2 = 50 + (row * 66)
            y2 = 50 + (col * 66)

            fake_slice = fake_image[x1:x2, y1:y2, :]
            real_image[x1:x2, y1:y2, :] = fake_slice

    # Plot image and save
    plt.imshow(real_image)
    if hide_axes:
        plt.axis("off")

    plt.savefig(result_fname, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close()


def extract_images(fname):
    """
    Extracts all 64x64 images from a given image grid

    :param fname: filename of image grid
    :return: array of images from grid
    """
    image = io.imread(fname=fname)
    images = []

    # Append every image from grid to an images array
    for row in range(4):
        for col in range(8):
            x1 = 2 + (row * 66)
            y1 = 2 + (col * 66)
            x2 = 66 + (row * 66)
            y2 = 66 + (col * 66)

            image_slice = image[x1:x2, y1:y2, :]
            images.append(image_slice)

    return images


def save_image(image_data, savename, dpi=1200, cmap="viridis", hide_axes=False):
    """
    Saves a single 64x64 image

    :param image_data: data of image to save
    :param savename: name under which to save
    :param dpi: dpi to save under
    :param cmap: colourmap to save with
    :param hide_axes: whether to hide axes
    """
    plt.imshow(image_data, cmap=cmap)

    if hide_axes:
        plt.axis("off")

    plt.savefig(f"{savename}.png", dpi=dpi, bbox_inches="tight", pad_inches=0)


def size_comparison(fname, hide_axes):
    """
    Creates and saves single image at 150x150 and 64x64
    :param fname: image to create comparison of
    :param hide_axes: whether to hide axes
    """
    image = io.imread(fname=fname)
    resized = skimage.transform.resize(image, (64, 64), anti_aliasing=False)

    save_image(image, "original_image", hide_axes=hide_axes)
    save_image(resized, "resized_image", hide_axes=hide_axes)


def save_all_overlays(gan_type="gan"):
    """
    Saves camouflage overlays at intervals in training for a given technique
    :param gan_type: technique to save images for
    """
    if gan_type == "gan":
        # Create camouflage for the original GAN results
        create_overlay_grid("./gan/real/epochs 10/gan_real_e10.png",
                            "./gan/fake/epochs 10/gan_fake_e10.png", "gan_e10_overlay.png",
                            dpi=1200, hide_axes=True)
        create_overlay_grid("./gan/real/epochs 10/gan_real_e10.png",
                            "./gan/fake/epochs 25/gan_fake_e25.png", "gan_e25_overlay.png",
                            dpi=1200, hide_axes=True)
        create_overlay_grid("./gan/real/epochs 10/gan_real_e10.png",
                            "./gan/fake/epochs 50/gan_fake_e50.png", "gan_e50_overlay.png",
                            dpi=1200, hide_axes=True)
        create_overlay_grid("./gan/real/epochs 10/gan_real_e10.png",
                            "./gan/fake/epochs 100/gan_fake_e100.png", "gan_e100_overlay.png",
                            dpi=1200, hide_axes=True)
        create_overlay_grid("./gan/real/epochs 10/gan_real_e10.png",
                            "./gan/fake/epochs 250/gan_fake_e250.png", "gan_e250_overlay.png",
                            dpi=1200, hide_axes=True)
        create_overlay_grid("./gan/real/epochs 10/gan_real_e10.png",
                            "./gan/fake/epochs 500/gan_fake_e500.png", "gan_e500_overlay.png",
                            dpi=1200, hide_axes=True)
        create_overlay_grid("./gan/real/epochs 10/gan_real_e10.png",
                            "./gan/fake/epochs 1000/gan_fake_e1000.png",
                            "gan_e1000_overlay.png", dpi=1200, hide_axes=True)
    elif gan_type == "dcgan":
        # Create camouflage for DCGAN results
        create_overlay_grid("./gan/real/epochs 10/gan_real_e10.png",
                            "./dcgan/fake/epochs 10/dcgan_fake_e10.png", "dcgan_e10_overlay.png",
                            dpi=1200, hide_axes=True)
        create_overlay_grid("./gan/real/epochs 10/gan_real_e10.png",
                            "./dcgan/fake/epochs 25/dcgan_fake_e25.png", "dcgan_e25_overlay.png",
                            dpi=1200, hide_axes=True)
        create_overlay_grid("./gan/real/epochs 10/gan_real_e10.png",
                            "./dcgan/fake/epochs 50/dcgan_fake_e50.png", "dcgan_e50_overlay.png",
                            dpi=1200, hide_axes=True)
        create_overlay_grid("./gan/real/epochs 10/gan_real_e10.png",
                            "./dcgan/fake/epochs 100/dcgan_fake_e100.png", "dcgan_e100_overlay.png",
                            dpi=1200, hide_axes=True)
        create_overlay_grid("./gan/real/epochs 10/gan_real_e10.png",
                            "./dcgan/fake/epochs 250/dcgan_fake_e250.png", "dcgan_e250_overlay.png",
                            dpi=1200, hide_axes=True)
        create_overlay_grid("./gan/real/epochs 10/gan_real_e10.png",
                            "./dcgan/fake/epochs 500/dcgan_fake_e500.png", "dcgan_e500_overlay.png",
                            dpi=1200, hide_axes=True)
        create_overlay_grid("./gan/real/epochs 10/gan_real_e10.png",
                            "./dcgan/fake/epochs 1000/dcgan_fake_e1000.png",
                            "dcgan_e1000_overlay.png", dpi=1200, hide_axes=True)
    elif gan_type == "wgan":
        # Create camouflage for the WGAN results
        create_overlay_grid("./gan/real/epochs 10/gan_real_e10.png",
                            "./wgan/fake/num_critics 5/epochs 10/wgan_fake_e10.png", "wgan_e10_overlay.png",
                            dpi=1200, hide_axes=True)
        create_overlay_grid("./gan/real/epochs 10/gan_real_e10.png",
                            "./wgan/fake/num_critics 5/epochs 25/wgan_fake_e25.png", "wgan_e25_overlay.png",
                            dpi=1200, hide_axes=True)
        create_overlay_grid("./gan/real/epochs 10/gan_real_e10.png",
                            "./wgan/fake/num_critics 5/epochs 50/wgan_fake_e50.png", "wgan_e50_overlay.png",
                            dpi=1200, hide_axes=True)
        create_overlay_grid("./gan/real/epochs 10/gan_real_e10.png",
                            "./wgan/fake/num_critics 5/epochs 100/wgan_fake_e100.png", "wgan_e100_overlay.png",
                            dpi=1200, hide_axes=True)
        create_overlay_grid("./gan/real/epochs 10/gan_real_e10.png",
                            "./wgan/fake/num_critics 5/epochs 250/wgan_fake_e250.png", "wgan_e250_overlay.png",
                            dpi=1200, hide_axes=True)
        create_overlay_grid("./gan/real/epochs 10/gan_real_e10.png",
                            "./wgan/fake/num_critics 5/epochs 500/wgan_fake_e500.png", "wgan_e500_overlay.png",
                            dpi=1200, hide_axes=True)
        create_overlay_grid("./gan/real/epochs 10/gan_real_e10.png",
                            "./wgan/fake/num_critics 5/epochs 1000/wgan_fake_e1000.png",
                            "wgan_e1000_overlay.png", dpi=1200, hide_axes=True)
    elif gan_type == "wgan-gp":
        # Create camouflage for the WGAN-GP results
        create_overlay_grid("./gan/real/epochs 10/gan_real_e10.png",
                            "./wgan-gp/fake/num_critics 5/epochs 10/wgan_gp_fake_e10.png", "wgan_gp_e10_overlay.png",
                            dpi=1200, hide_axes=True)
        create_overlay_grid("./gan/real/epochs 10/gan_real_e10.png",
                            "./wgan-gp/fake/num_critics 5/epochs 25/wgan_gp_fake_e25.png", "wgan_gp_e25_overlay.png",
                            dpi=1200, hide_axes=True)
        create_overlay_grid("./gan/real/epochs 10/gan_real_e10.png",
                            "./wgan-gp/fake/num_critics 5/epochs 50/wgan_gp_fake_e50.png", "wgan_gp_e50_overlay.png",
                            dpi=1200, hide_axes=True)
        create_overlay_grid("./gan/real/epochs 10/gan_real_e10.png",
                            "./wgan-gp/fake/num_critics 5/epochs 100/wgan_gp_fake_e100.png", "wgan_gp_e100_overlay.png",
                            dpi=1200, hide_axes=True)
        create_overlay_grid("./gan/real/epochs 10/gan_real_e10.png",
                            "./wgan-gp/fake/num_critics 5/epochs 250/wgan_gp_fake_e250.png", "wgan_gp_e250_overlay.png",
                            dpi=1200, hide_axes=True)
        create_overlay_grid("./gan/real/epochs 10/gan_real_e10.png",
                            "./wgan-gp/fake/num_critics 5/epochs 500/wgan_gp_fake_e500.png", "wgan_gp_e500_overlay.png",
                            dpi=1200, hide_axes=True)
        create_overlay_grid("./gan/real/epochs 10/gan_real_e10.png",
                            "./wgan-gp/fake/num_critics 5/epochs 1000/wgan_gp_fake_e1000.png",
                            "wgan_gp_e1000_overlay.png", dpi=1200, hide_axes=True)


def save_first_img(gan_type):
    """
    Saves the first image in the grid at specific intervals in training for a given technique
    :param gan_type: technique to save image for
    """
    if gan_type == "gan":
        # Extract images from GAN results
        e10 = extract_images("./gan/camo/gan_e10_overlay.png")
        e25 = extract_images("./gan/camo/gan_e25_overlay.png")
        e50 = extract_images("./gan/camo/gan_e50_overlay.png")
        e100 = extract_images("./gan/camo/gan_e100_overlay.png")
        e250 = extract_images("./gan/camo/gan_e250_overlay.png")
        e500 = extract_images("./gan/camo/gan_e500_overlay.png")
        e1000 = extract_images("./gan/camo/gan_e1000_overlay.png")
        # Save first image in grid
        save_image(e10[0], "gan_e10_first", hide_axes=True)
        save_image(e25[0], "gan_e25_first", hide_axes=True)
        save_image(e50[0], "gan_e50_first", hide_axes=True)
        save_image(e100[0], "gan_e100_first", hide_axes=True)
        save_image(e250[0], "gan_e250_first", hide_axes=True)
        save_image(e500[0], "gan_e500_first", hide_axes=True)
        save_image(e1000[0], "gan_e1000_first", hide_axes=True)
    elif gan_type == "dcgan":
        # Extract images from DCGAN results
        e10 = extract_images("./dcgan/camo/dcgan_e10_overlay.png")
        e25 = extract_images("./dcgan/camo/dcgan_e25_overlay.png")
        e50 = extract_images("./dcgan/camo/dcgan_e50_overlay.png")
        e100 = extract_images("./dcgan/camo/dcgan_e100_overlay.png")
        e250 = extract_images("./dcgan/camo/dcgan_e250_overlay.png")
        e500 = extract_images("./dcgan/camo/dcgan_e500_overlay.png")
        e1000 = extract_images("./dcgan/camo/dcgan_e1000_overlay.png")
        # Save first image in grid
        save_image(e10[0], "dcgan_e10_first", hide_axes=True)
        save_image(e25[0], "dcgan_e25_first", hide_axes=True)
        save_image(e50[0], "dcgan_e50_first", hide_axes=True)
        save_image(e100[0], "dcgan_e100_first", hide_axes=True)
        save_image(e250[0], "dcgan_e250_first", hide_axes=True)
        save_image(e500[0], "dcgan_e500_first", hide_axes=True)
        save_image(e1000[0], "dcgan_e1000_first", hide_axes=True)
    elif gan_type == "wgan":
        # Extract images from WGAN results
        e10 = extract_images("./wgan/camo/wgan_e10_overlay.png")
        e25 = extract_images("./wgan/camo/wgan_e25_overlay.png")
        e50 = extract_images("./wgan/camo/wgan_e50_overlay.png")
        e100 = extract_images("./wgan/camo/wgan_e100_overlay.png")
        e250 = extract_images("./wgan/camo/wgan_e250_overlay.png")
        e500 = extract_images("./wgan/camo/wgan_e500_overlay.png")
        e1000 = extract_images("./wgan/camo/wgan_e1000_overlay.png")
        # Save first image in grid
        save_image(e10[0], "wgan_e10_first", hide_axes=True)
        save_image(e25[0], "wgan_e25_first", hide_axes=True)
        save_image(e50[0], "wgan_e50_first", hide_axes=True)
        save_image(e100[0], "wgan_e100_first", hide_axes=True)
        save_image(e250[0], "wgan_e250_first", hide_axes=True)
        save_image(e500[0], "wgan_e500_first", hide_axes=True)
        save_image(e1000[0], "wgan_e1000_first", hide_axes=True)
    elif gan_type == "wgan-gp":
        # Extract images from WGAN-GP results
        e10 = extract_images("./wgan-gp/camo/wgan_gp_e10_overlay.png")
        e25 = extract_images("./wgan-gp/camo/wgan_gp_e25_overlay.png")
        e50 = extract_images("./wgan-gp/camo/wgan_gp_e50_overlay.png")
        e100 = extract_images("./wgan-gp/camo/wgan_gp_e100_overlay.png")
        e250 = extract_images("./wgan-gp/camo/wgan_gp_e250_overlay.png")
        e500 = extract_images("./wgan-gp/camo/wgan_gp_e500_overlay.png")
        e1000 = extract_images("./wgan-gp/camo/wgan_gp_e1000_overlay.png")
        # Save first image in grid
        save_image(e10[0], "wgan_gp_e10_first", hide_axes=True)
        save_image(e25[0], "wgan_gp_e25_first", hide_axes=True)
        save_image(e50[0], "wgan_gp_e50_first", hide_axes=True)
        save_image(e100[0], "wgan_gp_e100_first", hide_axes=True)
        save_image(e250[0], "wgan_gp_e250_first", hide_axes=True)
        save_image(e500[0], "wgan_gp_e500_first", hide_axes=True)
        save_image(e1000[0], "wgan_gp_e1000_first", hide_axes=True)


def get_losses(fname):
    """
    Extracts loss information from saved loss text files
    :param fname: filename for loss file
    :return:
    """
    # empty arrays for loss value, where index in array is epoch+1
    disc = []
    gen = []

    # open loss file
    with open(fname, 'r') as loss_input:
        # read every line
        lines = loss_input.readlines()

        # for each line, split by comma delimiter
        for line in lines:
            values = line.split(",")

            # for each loss value in list of values, strip of any whitespace
            for idx, value in enumerate(values):
                values[idx] = value.strip()

            # append discriminator loss to disc[] and generator loss to gen[]
            disc.append(float(values[1]))
            gen.append(float(values[2]))

    # return loss arrays
    return disc, gen


def plot_loss_graph(fname, disc, gen, max_epochs):
    """
    Plots gen and disc loss line graph
    :param fname: filename to save graph under
    :param disc: discriminator loss values
    :param gen: generator loss values
    :param max_epochs: maximum number of epochs to graph until
    """
    # Create list of epochs for x-axis scale
    epochs = list(range(1, max_epochs + 1))

    # Set axis labels
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    # Plot discriminator and generator losses as separate lines
    plt.plot(epochs, disc[:max_epochs], label="Discriminator Loss", color='red')
    plt.plot(epochs, gen[:max_epochs], label="Generator Loss", color='blue')

    # Create legend
    plt.legend()
    plt.grid(linestyle='--', color='black', linewidth=0.5)

    plt.locator_params(axis='x', nbins=20)

    # Save image
    plt.savefig(fname, dpi=1200, bbox_inches="tight", pad_inches=0)
    plt.close()


def calc_mse(camo_fname, real_fname):
    """
    Calculates Mean Square Error between camo and real image
    :param camo_fname: filename of camo image
    :param real_fname: filename of real image
    :return: MSE values
    """
    # Extract camo and real images
    camo_images = extract_images(camo_fname)
    real_images = extract_images(real_fname)
    mse_vals = []

    # For real image, calculate MSE between that and camo image
    for idx, real in enumerate(real_images):
        camo = camo_images[idx]

        MSE = mse(real, camo)
        mse_vals.append(MSE)

    return mse_vals


def calc_all_mse(gan_type):
    """
    Calculates MSE values for a given gan technique and outputs to text file
    :param gan_type: technique to calculate
    """
    real_fname = "./gan/real/epochs 10/gan_real_e10.png"
    epochs = _EPOCHS

    for epoch in epochs:
        # If WGAN-GP, separate formatting is needed
        if gan_type == "wgan-gp":
            newfile = open(f"./{gan_type}/camo/wgan_gp_e{epoch}_mse.txt", "w")
        # else, create a new mse file
        else:
            newfile = open(f"./{gan_type}/camo/{gan_type}_e{epoch}_mse.txt", "w")
        # close new file
        newfile.close()

        # calculate mse values for given technique
        if gan_type == "wgan-gp":
            mse_vals = calc_mse(
                f"./{gan_type}/camo/wgan_gp_e{epoch}_overlay.png", real_fname)
        else:
            mse_vals = calc_mse(f"./{gan_type}/camo/{gan_type}_e{epoch}_overlay.png", real_fname)

        # for value in mse array, write value to text file
        for idx, val in enumerate(mse_vals):
            if gan_type == "wgan-gp":
                with open(f"./{gan_type}/camo/wgan_gp_e{epoch}_mse.txt", "a") as output:
                    output.write(f"Image {idx + 1} MSE: {val:.4f}\n")
            else:
                with open(f"./{gan_type}/camo/{gan_type}_e{epoch}_mse.txt", "a") as output:
                    output.write(f"Image {idx + 1} MSE: {val:.4f}\n")


def calc_uiqi(camo_fname, real_fname):
    """
    Calculates the Universal Image Quality Index between camo and real image
    :param camo_fname: filename of camo image
    :param real_fname: filename of real image
    :return: array of uiqi values
    """
    # Extract camo and real images
    camo_images = extract_images(camo_fname)
    real_images = extract_images(real_fname)
    uiqi_vals = []

    # for real image, calculate UIQI between that and camo
    for idx, real in enumerate(real_images):
        camo = camo_images[idx]

        UIQI = uqi(real, camo)
        uiqi_vals.append(UIQI)

    return uiqi_vals


def calc_all_uiqi(gan_type):
    """
    Calculates UIQI values for a given gan technique and outputs to text file
    :param gan_type: technique to calculate
    """
    real_fname = "./gan/real/epochs 10/gan_real_e10.png"
    epochs = _EPOCHS

    for epoch in epochs:
        if gan_type == "wgan-gp":
            newfile = open(f"./{gan_type}/camo/wgan_gp_e{epoch}_uiqi.txt", "w")
        else:
            newfile = open(f"./{gan_type}/camo/{gan_type}_e{epoch}_uiqi.txt", "w")
        newfile.close()

        if gan_type == "wgan-gp":
            uiqi_vals = calc_uiqi(
                f"./{gan_type}/camo/wgan_gp_e{epoch}_overlay.png", real_fname)
        else:
            uiqi_vals = calc_uiqi(f"./{gan_type}/camo/{gan_type}_e{epoch}_overlay.png", real_fname)

        for idx, val in enumerate(uiqi_vals):
            if gan_type == "wgan-gp":
                with open(f"./{gan_type}/camo/wgan_gp_e{epoch}_uiqi.txt", "a") as output:
                    output.write(f"Image {idx + 1} UIQI: {val:.4f}\n")
            else:
                with open(f"./{gan_type}/camo/{gan_type}_e{epoch}_uiqi.txt", "a") as output:
                    output.write(f"Image {idx + 1} UIQI: {val:.4f}\n")


def calc_avg(fname):
    """
    Calculates average of MSE or UIQI values, given the text file
    :param fname: filename of MSE/UIQI text file
    :return: average value
    """
    vals = []

    # Open text file and read all lines
    with open(fname, "r") as input:
        lines = input.readlines()

        # for each line, split by colon delimiter and append stripped float value
        for line in lines:
            values = line.split(':')
            vals.append(float(values[1].strip()))

    # take mean of values
    avg = np.mean(vals)

    return float(f"{avg:.4f}")


def calc_all_avg(gan_type, avg_type):
    """
    Calculates average of MSE/UIQI at specific intervals in training, for a given gan_type
    :param gan_type: GAN technique to find average for
    :param avg_type: MSE or UIQI
    :return: list of averages
    """
    averages = []
    epochs = _EPOCHS

    for epoch in epochs:
        if gan_type == "wgan-gp":
            avg = calc_avg(f"./{gan_type}/camo/wgan_gp_e{epoch}_{avg_type}.txt")
        else:
            avg = calc_avg(f"./{gan_type}/camo/{gan_type}_e{epoch}_{avg_type}.txt")

        averages.append(avg)

    return averages


def plot_avg(gan_type, avg_type):
    """
    Plot one average similarity value for given GAN technique
    :param gan_type: GAN technique to plot
    :param avg_type: MSE or UIQI
    """
    epochs = _EPOCHS
    averages = calc_all_avg(gan_type, avg_type)

    # set filename of image to save
    if gan_type == "wgan-gp":
        fname = f"./{gan_type}/wgan_gp_avg_{avg_type}"
    else:
        fname = f"./{gan_type}/{gan_type}_avg_{avg_type}"

    # Plot average over training epochs
    plt.plot(epochs, averages, label=f"Average {avg_type.upper()}", color='red')
    plt.legend()
    plt.grid(linestyle='--', color='black', linewidth=0.5)

    plt.locator_params(axis='x', nbins=20)

    # Save plot
    plt.savefig(fname, dpi=1200, bbox_inches="tight", pad_inches=0)
    plt.close()


def plot_all_avg(avg_type):
    """
    Plots average MSE/UIQI for all GAN techniques on one graph
    :param avg_type: MSE or UIQI
    """
    epochs = _EPOCHS
    gan_types = _GAN_TYPES
    colours = ['red', 'blue', 'yellow', 'green']

    fname = f"./all_avg_{avg_type}"

    # For each GAN technique, plot average metric against training epoch
    for idx, gan_type in enumerate(gan_types):
        averages = calc_all_avg(gan_type, avg_type)
        plt.plot(epochs, averages, label=f"{gan_type.upper()}", color=colours[idx])

    plt.legend()
    plt.grid(linestyle='--', color='black', linewidth=0.5)
    plt.title(label=f"Average {avg_type.upper()} per epoch for all GANs")

    plt.locator_params(axis='x', nbins=20)

    plt.savefig(fname, dpi=1200, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_single_image(fname, img_fname, idx):
    """
    Saves any given image in a given grid
    :param fname: filename of image grid
    :param img_fname: filename to save image under
    :param idx: index of image in grid
    """
    imgs = extract_images(fname)

    img = imgs[idx]

    save_image(img, img_fname, hide_axes=True)


def save_batch_images(gan_type, img_idx):
    """
    Saves image at given position, over all epochs of training for specific GAN technique
    :param gan_type: GAN technique to save images for
    :param img_idx: index of image in grid
    """
    epochs = _EPOCHS

    # For each epoch to save for, get that overlay and save the image at img_idx
    for idx, epoch in enumerate(epochs):
        if gan_type == "wgan-gp":
            fname = f"./{gan_type}/camo/wgan_gp_e{epoch}_overlay.png"
        else:
            fname = f"./{gan_type}/camo/{gan_type}_e{epoch}_overlay.png"

        img_fname = f"./example-camo/{gan_type}-camo-{idx+1}"
        save_single_image(fname=fname, img_fname=img_fname, idx=img_idx[idx])


def load_observations(fname):
    """
    Loads CSV of observation test results
    :param fname:
    :return:
    """
    dataset = pd.read_csv(fname, sep=",")
    return dataset


def get_single_observe(dataset, idx, single=False):
    """
    Gets a single observation (answer to Q1 and Q2 for an image from one person)
    :param dataset: observation test dataset
    :param idx: index of observation
    :param single: whether column name is numbered or not
    :return: observation result data
    """
    # In this case, Spot is the answer to Question 1: How easy is it to spot this camouflage?
    # Blend is the answer to Question 2: How well does this camouflage blend in with the background?
    image_data = dataset[[f"Spot {idx}", f"Blend {idx}"]].copy()

    # If single, rename column to remove numbering from column name
    if single:
        image_data.rename(columns={f"Spot {idx}":"Spot", f"Blend {idx}":"Blend"}, inplace=True)

    return image_data


def get_batch_observe(dataset, observe_type):
    """
    Gets a batch of observations for a given GAN technique
    :param dataset: observation test dataset
    :param observe_type: GAN technique to get results for
    :return: batch of observations
    """
    # For each type supplied, get array of result order.
    # The questions were not asked in order of training i.e. epoch 1 might be question 2
    # Global variables exist to track order in which questions were asked
    if observe_type == "gan":
        results = _GAN_RESULTS
    elif observe_type == "dcgan":
        results = _DCGAN_RESULTS
    elif observe_type == "wgan":
        results = _WGAN_RESULTS
    elif observe_type == "wgan-gp":
        results = _WGAN_GP_RESULTS
    else:
        results = _REAL_RESULTS

    # Get and rename column heading
    temp = get_single_observe(dataset, results[0], single=True)
    temp.rename(columns={"Spot":f"Spot {1}", "Blend":f"Blend {1}"}, inplace=True)

    # For each index in results, get observation at that index
    for idx, result in enumerate(results):
        observe = get_single_observe(dataset, result, single=True)

        # If the index is not 0, i.e. not the header
        if idx != 0:
            # Get spot and blend values
            spot = observe["Spot"]
            blend = observe["Blend"]

            # Join spot and blend values to output dataframe
            temp = temp.join(spot)
            temp = temp.join(blend)
            temp.rename(columns={"Spot":f"Spot {idx+1}", "Blend":f"Blend {idx+1}"}, inplace=True)

    return temp


def get_observations(dataset, gan_type):
    """
    Gets observations for each observer (5 in total) for given GAN technique
    :param dataset: observations dataset
    :param gan_type: GAN technique
    :return: array of spot and blend responses, respectively, for all 5 people
    """
    gan_observes = get_batch_observe(dataset, gan_type)
    # Create empty arrays, containing 5 empty response arrays
    spot = [[], [], [], [], []]
    blend = [[], [], [], [], []]

    # For each user
    for user in range(0, 5):
        # For each index, get spot and blend values
        for idx in range(1, gan_observes.shape[1]//2 + 1):
            spot[user].append(gan_observes[f"Spot {idx}"][user])
            blend[user].append(gan_observes[f"Blend {idx}"][user])

    return spot, blend


def response_heatmap(dataset, gan_type):
    """
    Creates a heatmap showing distribution of responses
    :param dataset: observation dataset
    :param gan_type: GAN technique to create heatmap for
    """
    # Create numpy array of spot and blend value
    spot, blend = get_observations(dataset, gan_type)
    spot = np.array(spot)
    blend = np.array(blend)

    # Transpose the two arrays
    spot_t = np.transpose(spot)
    blend_t = np.transpose(blend)

    # Create array of same shape as spot/blend, filled with zeros
    data = np.zeros((5, 4), int)

    # For each response in transposed spot
    for response_idx in range(0, spot_t.shape[0]):
        # For each user in transposed spot
        for user_idx in range(0, spot_t.shape[1]):
            # Get spot and blend value
            spot_idx = spot_t[response_idx][user_idx]
            blend_idx = blend_t[response_idx][user_idx]

            # Add 1 to counter in same spot in non-transposed zero array
            data[blend_idx-1][spot_idx-1] += 1

    # Create grey colourmap for heatmap, that is cream background (easier to look at)
    gray_cmap = cm.get_cmap('gray_r')
    new_gray = ListedColormap(gray_cmap(np.linspace(0.05, 1.0, 256)))

    # Transpose data count
    data = np.transpose(data)

    # Create dataframe from count data, with each column being blend labels and each index in column being spot labels
    df = pd.DataFrame(data, columns=_BLEND, index=_SPOT)
    # Create sns heatmap using custom colourmap, with labels
    heatmap = sns.heatmap(df, cmap=new_gray, annot=True)
    fig = heatmap.get_figure()
    fig.savefig(f"./{gan_type}-heatmap.png", dpi=300, bbox_inches="tight", pad_inches=0)
    fig.clear()


if __name__ == '__main__':
    # real_fname = "./wgan/real/num_critics 5/epochs 500/wgan_real_e1.png"
    # fake_fname = "./wgan/fake/num_critics 5/epochs 500/wgan_fake_e500.png"

    # real_images = extract_images(real_fname)
    # fake_images = extract_images(fake_fname)
    #
    # uqi = uqi(real_images[0], fake_images[0])
    # mse = mse(real_images[0], fake_images[0])
    # print(f"MSE: {mse}")
    # print(f"UIQI: {uqi}")

    # size_comparison("./scene-dataset/forest/8.jpg")

    # gan_disc, gan_gen = get_losses("./gan/losses/epochs_1000.txt")
    # plot_loss_graph("./gan_loss", gan_disc, gan_gen, 1000)
    #
    # dcgan_disc, dcgan_gen = get_losses("./dcgan/losses/epochs_1000.txt")
    # plot_loss_graph("./dcgan_loss", dcgan_disc, dcgan_gen, 1000)
    #
    # wgan_disc, wgan_gen = get_losses("./wgan/losses/num_critics 5/epochs_1000.txt")
    # plot_loss_graph("./wgan_loss", wgan_disc, wgan_gen, 1000)
    #
    # wgan_gp_disc, wgan_gp_gen = get_losses("./wgan-gp/losses/num_critics 5/epochs_1000.txt")
    # plot_loss_graph("./wgan_gp_loss", wgan_gp_disc, wgan_gp_gen, 1000)

    # e1 = extract_images("./gan/camo/gan_e1_overlay.png")
    # save_image(e1[0], "gan_e1_first", hide_axes=True)
    #
    # e1 = extract_images("./dcgan/camo/dcgan_e1_overlay.png")
    # save_image(e1[0], "dcgan_e1_first", hide_axes=True)
    #
    # e1 = extract_images("./wgan/camo/wgan_e1_overlay.png")
    # save_image(e1[0], "wgan_e1_first", hide_axes=True)
    #
    # e1 = extract_images("./wgan-gp/camo/wgan_gp_e1_overlay.png")
    # save_image(e1[0], "wgan_gp_e1_first", hide_axes=True)

    # for gan_type in _GAN_TYPES:
    #     save_all_overlays(gan_type)
    #
    #     save_first_img(gan_type)
    #
    #     calc_all_mse(gan_type)
    #
    #     calc_all_uiqi(gan_type)
    #
    #     plot_avg(gan_type, "mse")
    #     plot_avg(gan_type, "uiqi")
    #
    #     calc_all_ssim(gan_type)
    #
    # plot_all_avg("mse")
    # plot_all_avg("uiqi")
    # plot_all_avg("ssim")

    # save_batch_images("gan", [3, 22, 6, 25, 9, 28, 7, 24])
    # save_batch_images("dcgan", [14, 2, 7, 23, 0, 6, 25, 24])
    # save_batch_images("wgan", [28, 0, 14, 3, 16, 23, 28, 26])
    # save_batch_images("wgan-gp", [6, 22, 18, 9, 16, 19, 30, 10])

    # save_single_image("./gan/real/epochs 10/gan_real_e10.png", img_fname="./example-camo/real-1", idx=0)
    # save_single_image("./gan/real/epochs 10/gan_real_e10.png", img_fname="./example-camo/real-2", idx=2)
    # save_single_image("./gan/real/epochs 10/gan_real_e10.png", img_fname="./example-camo/real-3", idx=3)
    # save_single_image("./gan/real/epochs 10/gan_real_e10.png", img_fname="./example-camo/real-4", idx=6)
    # save_single_image("./gan/real/epochs 10/gan_real_e10.png", img_fname="./example-camo/real-5", idx=7)
    # save_single_image("./gan/real/epochs 10/gan_real_e10.png", img_fname="./example-camo/real-6", idx=9)
    # save_single_image("./gan/real/epochs 10/gan_real_e10.png", img_fname="./example-camo/real-7", idx=10)
    # save_single_image("./gan/real/epochs 10/gan_real_e10.png", img_fname="./example-camo/real-8", idx=14)
    # save_single_image("./gan/real/epochs 10/gan_real_e10.png", img_fname="./example-camo/real-9", idx=16)
    # save_single_image("./gan/real/epochs 10/gan_real_e10.png", img_fname="./example-camo/real-10", idx=18)
    # save_single_image("./gan/real/epochs 10/gan_real_e10.png", img_fname="./example-camo/real-11", idx=19)
    # save_single_image("./gan/real/epochs 10/gan_real_e10.png", img_fname="./example-camo/real-12", idx=22)
    # save_single_image("./gan/real/epochs 10/gan_real_e10.png", img_fname="./example-camo/real-13", idx=23)
    # save_single_image("./gan/real/epochs 10/gan_real_e10.png", img_fname="./example-camo/real-14", idx=24)
    # save_single_image("./gan/real/epochs 10/gan_real_e10.png", img_fname="./example-camo/real-15", idx=25)
    # save_single_image("./gan/real/epochs 10/gan_real_e10.png", img_fname="./example-camo/real-16", idx=26)
    # save_single_image("./gan/real/epochs 10/gan_real_e10.png", img_fname="./example-camo/real-17", idx=28)
    # save_single_image("./gan/real/epochs 10/gan_real_e10.png", img_fname="./example-camo/real-18", idx=30)

    dataset = load_observations("./Observational Test (Numerical).csv")

    # img_1 = get_single_observe(dataset, 1, True)
    # print(img_1["Spot"])

    # gan = get_batch_observe(dataset, "gan")
    # dcgan = get_batch_observe(dataset, "dcgan")
    # wgan = get_batch_observe(dataset, "wgan")
    # wgan_gp = get_batch_observe(dataset, "wgan-gp")
    # real = get_batch_observe(dataset, "real")

    # gan_spot, gan_blend = get_observations(dataset, "gan")
    # print("\nGAN Results")
    # gan_easy, gan_hard = percent_spot(dataset, "gan")
    # gan_none, gan_poor, gan_great = percent_blend(dataset, "gan")
    # response_heatmap(dataset, "gan")

    # dcgan_spot, dcgan_blend = get_observations(dataset, "dcgan")
    # print("\nDCGAN Results")
    # dcgan_easy, dcgan_hard = percent_spot(dataset, "dcgan")
    # dcgan_none, dcgan_poor, dcgan_great = percent_blend(dataset, "dcgan")
    # response_heatmap(dataset, "dcgan")

    # wgan_spot, wgan_blend = get_observations(dataset, "wgan")
    # print("\nWGAN Results")
    # wgan_easy, wgan_hard = percent_spot(dataset, "wgan")
    # wgan_none, wgan_poor, wgan_great = percent_blend(dataset, "wgan")
    # response_heatmap(dataset, "wgan")

    # wgan_gp_spot, wgan_gp_blend = get_observations(dataset, "wgan-gp")
    # print("\nWGAN-GP Results")
    # wgan_gp_easy, wgan_gp_hard = percent_spot(dataset, "wgan-gp")
    # wgan_gp_none, wgan_gp_poor, wgan_gp_great = percent_blend(dataset, "wgan-gp")
    # response_heatmap(dataset, "wgan-gp")

    # real_spot, real_blend = get_observations(dataset, "real")
    # print("\nReal Results")
    # real_easy, real_hard = percent_spot(dataset, "real")
    # real_none, real_poor, real_great = percent_blend(dataset, "real")
    # response_heatmap(dataset, "real")

    for i in range(1, 1001):
        create_overlay_grid("./gan/real/epochs 10/gan_real_e10.png",
                            f"./gan/fake/epochs 1000/gan_fake_e{i}.png", f"./gan/all_camo/gan_overlay_e{i}.png",
                            dpi=1200, hide_axes=True)

