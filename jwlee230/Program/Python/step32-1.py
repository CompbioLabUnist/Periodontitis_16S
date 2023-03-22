"""
step32-1.py: draw scatter plot with correlation
"""
import argparse
import itertools
import tarfile
import matplotlib
import matplotlib.colors
import matplotlib.pyplot
import numpy
import scipy.stats
import seaborn
import step00

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

        seaborn.scatterplot(data=input_data, x=taxon1, y=taxon2, hue="LongStage", hue_order=step00.long_stage_order, palette=step00.color_stage_dict, legend="brief", s=400, ax=ax)
        matplotlib.pyplot.axline((numpy.mean(input_data[taxon1]), numpy.mean(input_data[taxon2])), slope=stat, color="k", linestyle="--", linewidth=4)
        matplotlib.pyplot.text(x=0.5, y=0.6, s=f"r={stat:.3f}\np={p:.3f}", color="k", ha="center", va="center")

        matplotlib.pyplot.xlim((-0.05, 1.05))
        matplotlib.pyplot.ylim((-0.05, 1.05))
        matplotlib.pyplot.xlabel(f"{taxon1} proportion")
        matplotlib.pyplot.ylabel(f"{taxon2} proportion")
        matplotlib.pyplot.legend(title="")
        matplotlib.pyplot.tight_layout()

        figures.append(f"{taxon1}+{taxon2}.pdf")
        fig.savefig(figures[-1])
        matplotlib.pyplot.close(fig)

    with tarfile.open(args.output, "w") as tar:
        for f in figures:
            tar.add(f, arcname=f)
