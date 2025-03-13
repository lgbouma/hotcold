import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from aesthetic.plot import set_style, savefig


def truncate_colormap(
    cmap: LinearSegmentedColormap, minval: float = 0.0, maxval: float = 1.0,
    n: int = 100
) -> LinearSegmentedColormap:
    """Truncate a colormap to limit its upper range.

    Args:
        cmap (LinearSegmentedColormap): Original colormap.
        minval (float, optional): Minimum value fraction. Defaults to 0.0.
        maxval (float, optional): Maximum value fraction. Defaults to 1.0.
        n (int, optional): Number of colors in new colormap.
            Defaults to 100.

    Returns:
        LinearSegmentedColormap: Truncated colormap.
    """
    new_colors = cmap(np.linspace(minval, maxval, n))
    new_cmap = LinearSegmentedColormap.from_list(
        f"trunc({cmap.name},{minval:.2f},{maxval:.2f})", new_colors
    )
    return new_cmap


def plot_li_vs_population() -> None:
    """Plot LiEW vs TEFF for stars with AGE < 100 binned by AGE.

    Reads data from '../data/EAGLES_table2.csv', filters for rows with
    AGE < 100, then loops over age bins (0–10, 10–20, etc.) to plot the data.
    Each bin is assigned a discrete color from a truncated plasma colormap.
    Stars in older bins are plotted with a higher zorder.
    The resulting plot is saved to 'results/li_ew/li_vs_population.png'.
    """

    # Read CSV file and filter by AGE.
    df = pd.read_csv("../data/EAGLES_table2.csv", skipinitialspace=True)
    df = df[df["AGE"] < 100]

    # Define age bins.
    bins = np.arange(0, 60 + 10, 10)  # [0, 10, 20, 30, 40, 50, 60]
    nbins = len(bins) - 1

    # Create truncated plasma colormap.
    plasma = plt.get_cmap("plasma")
    trunc_plasma = truncate_colormap(plasma, 0.0, 0.9)
    discrete_trunc_plasma = ListedColormap(
        trunc_plasma(np.linspace(0, 1, nbins)),
        name="trunc_plasma_discrete"
    )

    # Set plot style and create figure.
    set_style("science")
    rcParams['font.family'] = 'Arial'
    fig, ax = plt.subplots(figsize=(3,3))

    # Loop over age bins and plot each bin separately.
    for i in range(nbins):
        age_min = bins[i]
        age_max = bins[i + 1]
        if i < nbins - 1:
            sub_df = df[(df["AGE"] >= age_min) & (df["AGE"] < age_max)]
        else:
            # Include the right edge in the last bin.
            sub_df = df[(df["AGE"] >= age_min) & (df["AGE"] <= age_max)]
        if sub_df.empty:
            continue
        # Use the discrete colormap to select the color.
        color = discrete_trunc_plasma.colors[i]
        # Set zorder to i so that older bins (higher age) are plotted on top.
        ax.scatter(
            sub_df["TEFF"], sub_df["LiEW"], color=color,
            edgecolor="k", zorder=i, label=f"{age_min}-{age_max} Myr",
            linewidth=0.3, s=5
        )

    ax.scatter(
        2972, 107, marker='v', linewidth=0.6, color='greenyellow', s=50,
        zorder=99, edgecolor='k'
    )

    # Add legend to indicate the age bins.
    ax.legend(loc="upper left", fontsize='x-small', framealpha=1,
              handletextpad=-0.5, borderpad=0.1, borderaxespad=0.5,
              edgecolor='lightgray', ncols=2, columnspacing=0. )

    # Set axis labels and title.
    ax.set_xlabel("Effective Temperature [K]")
    ax.set_ylabel("Li$_{6708}$ EW [m$\mathrm{\AA}$]")
    ax.set_xlim([7100, 2800])
    ax.set_ylim([ -220, 860 ])
    #ax.set_title("LiEW vs TEFF for stars with AGE < 100")

    # Ensure output directory exists and save the figure.
    plot_dir = "results/li_ew"
    os.makedirs(plot_dir, exist_ok=True)
    savefig(fig, os.path.join(plot_dir, "li_vs_population.png"), writepdf=1)
    plt.close(fig)


if __name__ == "__main__":
    plot_li_vs_population()

