import matplotlib.pyplot as plt


def display_obs(obs):
    fig, ax = plt.subplots(1, len(obs), )
    for i, img in enumerate(obs):
        ax[i].imshow(img.squeeze(), cmap='binary')
    plt.show(fig)
