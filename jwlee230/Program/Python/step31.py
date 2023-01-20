"""
step31.py: Data composition
"""
import argparse
import itertools
import matplotlib
import matplotlib.colors
import matplotlib.pyplot
import numpy
import seaborn
import step00

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", type=str, help="Input TAR.gz file")
    parser.add_argument("output", type=str, help="Output PNG file")

    args = parser.parse_args()

    if not args.output.endswith(".png"):
        raise ValueError("OUTPUT file must end with .PNG!!")

    matplotlib.use("Agg")
    matplotlib.rcParams.update(step00.matplotlib_parameters)
    seaborn.set(context="poster", style="whitegrid", rc=step00.matplotlib_parameters)

    input_data = step00.read_pickle(args.input)
    print(input_data)

    taxa = list(input_data.columns)[:-2]
    for index in list(input_data.index):
        input_data.loc[index, taxa] = input_data.loc[index, taxa] / sum(input_data.loc[index, taxa])
    print(input_data)

    taxa = sorted(taxa, key=lambda x: sum(input_data.loc[:, x]), reverse=True)
    input_data = input_data.loc[:, taxa]
    print(input_data)

    fig, ax = matplotlib.pyplot.subplots(figsize=(36, 18))

    for i, (taxon, color) in list(enumerate(zip(taxa, itertools.cycle(matplotlib.colors.XKCD_COLORS)))):
        if i < 5:
            matplotlib.pyplot.bar(range(input_data.shape[0]), input_data.iloc[:, i], bottom=numpy.sum(input_data.iloc[:, :i], axis=1), color=color, linewidth=0, label="".join(taxon.split(";")[-2:]).replace("_", " "))
        else:
            matplotlib.pyplot.bar(range(input_data.shape[0]), input_data.iloc[:, i], bottom=numpy.sum(input_data.iloc[:, :i], axis=1), color=color, linewidth=0)

    matplotlib.pyplot.xticks([])
    matplotlib.pyplot.xlabel(f"{input_data.shape[0]} Samples")
    matplotlib.pyplot.ylabel(f"Proportion of {len(taxa)} Bacteria")
    matplotlib.pyplot.ylim(0, 1)
    matplotlib.pyplot.legend()
    matplotlib.pyplot.grid(True)

    matplotlib.pyplot.tight_layout()
    fig.savefig(args.output)
    fig.savefig(args.output.replace(".png", ".pdf"))
    matplotlib.pyplot.close(fig)
