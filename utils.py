import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

def plot_latent_space(vae, n=30, figsize=15):
    # display an n*n 2D manifold of digits
    digit_size = 28
    scale = 1.0
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = vae.decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[
                i * digit_size : (i + 1) * digit_size,
                j * digit_size : (j + 1) * digit_size,
            ] = digit

    plt.figure(figsize=(figsize, figsize))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap="Greys_r")
    plt.show()

def visualize_dataset(dataset, title="Untitled", n_samples=9):
    plt.figure(figsize=(6, 6)).suptitle(title, fontsize=18)

    # decide subplot dimension
    d = np.sqrt(n_samples)
    d = np.ceil(d).astype("uint8")

    for i, samples in enumerate(iter(dataset.take(n_samples))):
        if type(samples) is tuple:
            images = samples[0]
        else:
            images = samples
        
        plt.subplot(d, d, i + 1)
        img = images[0].numpy()
        # print(images.shape)
        # print(img.shape)
        vmin = np.min(img)
        vmax = np.max(img)
        # print(vmin, vmax)
        plt.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
        plt.axis("off")
    plt.show()

def visualize_imgrid(X, title="Untitled", figpath=None):
    plt.figure(figsize=(6, 6))
    # plt.figure(figsize=(6, 6)).suptitle(title, fontsize=18)

    n_samples = X.shape[0]
    
    # decide subplot dimension
    d = np.sqrt(n_samples)
    d = np.ceil(d).astype("uint8")

    for i, samples in enumerate(X):
        plt.subplot(d, d, i + 1)
        img = samples
        # print(images.shape)
        # print(img.shape)
        vmin = np.min(img)
        vmax = np.max(img)
        # print(vmin, vmax)
        plt.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
        plt.axis("off")
    
    if figpath is not None:
        plt.savefig(figpath)


def visualize_grid(X, figpath=None):
    fig = plt.figure(figsize=(6, 6))

    # decide grid dimension
    n_samples = X.shape[0]
    d = np.sqrt(n_samples)
    d = np.ceil(d).astype("uint8")

    grid = ImageGrid(fig, 
        111, # similar to subplot(111)
        nrows_ncols=(d, d), # creates d x d grid
        axes_pad=0.05, # pad between axes in inch.
    )

    for ax, im in zip(grid, X):
        ax.imshow(im)
        ax.axis("off")
    
    if figpath is not None:
        plt.savefig(figpath)
    else:
        plt.show()