"""
step40.py: Taxonomy barplot
"""
import argparse
import pandas
import matplotlib
import matplotlib.colors
import matplotlib.pyplot
import numpy
import tqdm
import step00


def get_name(taxonomy: str) -> str:
    taxonomy_list = taxonomy.split("; ")
    return " ".join(taxonomy_list[-2:]).replace("_", " ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", type=str, help="Input TAR.gz file")
    parser.add_argument("output", type=str, help="Output PNG file")

    args = parser.parse_args()

    if not args.output.endswith(".png"):
        raise ValueError("OUTPUT file must end with .PNG!!")

    data: pandas.DataFrame = step00.read_pickle(args.input)
    del data["ShortStage"], data["LongStage"]
    print(data)

    taxa_list = sorted(data.columns, key=lambda x: sum(data[x]), reverse=True)
    patient_list = sorted(data.index, key=lambda x: tuple(data.loc[x, taxa_list].to_numpy()), reverse=True)
    data = data.loc[patient_list, taxa_list]
    print(data)

    matplotlib.use("Agg")
    matplotlib.rcParams.update(step00.matplotlib_parameters)

    taxa_colors = dict(zip(taxa_list, matplotlib.colors.XKCD_COLORS.keys()))

    fig, ax = matplotlib.pyplot.subplots(figsize=(32, 18))

    for i, taxon in tqdm.tqdm(list(enumerate(taxa_list))):
        label = get_name(taxon) if (i < 20) else None
        matplotlib.pyplot.bar(range(len(patient_list)), data[taxon], bottom=numpy.sum(data.iloc[:, :i], axis=1), color=taxa_colors[taxon], label=label)

    matplotlib.pyplot.grid(True)
    matplotlib.pyplot.xticks([])
    matplotlib.pyplot.ylim((0, 1))
    matplotlib.pyplot.xlabel(f"{len(patient_list)} patients")
    matplotlib.pyplot.ylabel(f"{len(taxa_list)} taxa proportions")
    matplotlib.pyplot.legend(loc="upper right", ncol=4, fontsize="xx-small")

    matplotlib.pyplot.tight_layout()

    fig.savefig(args.output)
    fig.savefig(args.output.replace(".png", ".pdf"))
    fig.savefig(args.output.replace(".png", ".svg"))
    matplotlib.pyplot.close(fig)
