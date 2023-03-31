"""
step32-1.py: draw scatter plot with correlation
"""
import argparse
import itertools
import tarfile
import matplotlib
import matplotlib.colors
from matplotlib.patches import Ellipse
import matplotlib.pyplot
import numpy
import scipy.stats
import seaborn
import step00


def confidence_ellipse(x, y, ax, n_std=2.0, **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = numpy.cov(x, y)
    pearson = cov[0, 1] / numpy.sqrt(cov[0, 0] * cov[1, 1])

    ell_radius_x = numpy.sqrt(1 + pearson)
    ell_radius_y = numpy.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, **kwargs)

    scale_x = numpy.sqrt(cov[0, 0]) * n_std
    mean_x = numpy.mean(x)

    scale_y = numpy.sqrt(cov[1, 1]) * n_std
    mean_y = numpy.mean(y)

    transf = matplotlib.transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", type=str, help="Input TAR.gz file")
    parser.add_argument("output", type=str, help="Output TAR file")

    args = parser.parse_args()

    if not args.output.endswith(".tar"):
        raise ValueError("OUTPUT file must end with .TAR!!")

    matplotlib.use("Agg")
    matplotlib.rcParams.update(step00.matplotlib_parameters)
    seaborn.set(context="poster", style="whitegrid", rc=step00.matplotlib_parameters)

    input_data = step00.read_pickle(args.input)
    print(input_data)

    input_data.columns = list(map(lambda x: " ".join(x.split("; ")[-2:]), list(input_data.columns)[:-2])) + list(input_data.columns)[-2:]
    print(list(input_data.columns))

    taxa = sorted(list(input_data.columns)[:-2])
    for index in list(input_data.index):
        input_data.loc[index, taxa] = input_data.loc[index, taxa] / sum(input_data.loc[index, taxa])
    print(input_data)

    figures = list()
    for taxon1, taxon2 in itertools.combinations(taxa, r=2):
        stat, p = scipy.stats.pearsonr(input_data[taxon1], input_data[taxon2])

        if (abs(stat) < 0.5) or (p > 0.05):
            continue

        fig, ax = matplotlib.pyplot.subplots(figsize=(24, 24))

        seaborn.scatterplot(data=input_data, x=taxon1, y=taxon2, hue="LongStage", hue_order=step00.long_stage_order, palette=step00.color_stage_dict, legend="brief", s=800, edgecolor="none", ax=ax)

        matplotlib.pyplot.title(f"r={stat:.3f}, p={p:.3f}")
        matplotlib.pyplot.xlabel(f"{taxon1} proportion")
        matplotlib.pyplot.ylabel(f"{taxon2} proportion")
        matplotlib.pyplot.legend(title="", loc="upper right")
        matplotlib.pyplot.tight_layout()

        figures.append(f"{taxon1}+{taxon2}.pdf")
        fig.savefig(figures[-1])
        matplotlib.pyplot.close(fig)

    with tarfile.open(args.output, "w") as tar:
        for f in figures:
            tar.add(f, arcname=f)
