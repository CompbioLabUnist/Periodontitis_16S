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
    print(data)

    taxa_list = sorted(list(data.columns)[:-2], key=lambda x: sum(data[x]), reverse=True)
    patient_list = sorted(data.index, key=lambda x: sum(data.loc[x, taxa_list]), reverse=True)
    data = data.loc[patient_list, taxa_list + ["LongStage"]]
    print(data)

    matplotlib.use("Agg")
    matplotlib.rcParams.update(step00.matplotlib_parameters)

    taxa_colors = dict(zip(taxa_list, matplotlib.colors.XKCD_COLORS.keys()))

    fig, axs = matplotlib.pyplot.subplots(figsize=(32, 18), ncols=4, sharey=True)

    for i, stage in enumerate(step00.long_stage_order):
        drawing_data = data.loc[(data["LongStage"] == stage)]
        length = len(drawing_data)

        for j, taxon in tqdm.tqdm(list(enumerate(taxa_list))):
            label = get_name(taxon) if (i < 10) else None
            axs[i].bar(range(length), drawing_data[taxon], bottom=numpy.sum(drawing_data.iloc[:, :j], axis=1), color=taxa_colors[taxon], label=label)

        axs[i].grid(True)
        axs[i].set_xticks([])
        axs[i].set_xlabel(f"{length} {stage}")
        if i == 0:
            axs[i].set_ylabel(f"{len(taxa_list)} taxa abundances")
        elif i == 3:
            axs[i].legend(loc="upper right", ncol=2, fontsize="xx-small")

    matplotlib.pyplot.tight_layout()

    fig.savefig(args.output)
    fig.savefig(args.output.replace(".png", ".pdf"))
    fig.savefig(args.output.replace(".png", ".svg"))
    matplotlib.pyplot.close(fig)
