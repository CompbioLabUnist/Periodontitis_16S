"""
step40.py: Taxonomy barplot
"""
import argparse
import itertools
import typing
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

    output_data = pandas.DataFrame([(stage, taxon, numpy.mean(data.loc[(data["LongStage"] == stage), taxon]), numpy.std(data.loc[(data["LongStage"] == stage), taxon]) / 2) for stage, taxon in tqdm.tqdm(list(itertools.product(step00.long_stage_order, taxa_list)))], columns=["Stage", "Taxonomy", "mean", "std"])

    matplotlib.use("Agg")
    matplotlib.rcParams.update(step00.matplotlib_parameters)

    taxa_colors = dict(zip(taxa_list, matplotlib.colors.XKCD_COLORS.keys()))

    fig, ax = matplotlib.pyplot.subplots(figsize=(24, 24))

    used_taxa: typing.List[str] = list()
    for i, taxon in tqdm.tqdm(list(enumerate(taxa_list))):
        label = get_name(taxon) if (i < 20) else None
        matplotlib.pyplot.bar(range(len(step00.long_stage_order)), [output_data.loc[(output_data["Stage"] == stage) & (output_data["Taxonomy"] == taxon), "mean"].to_numpy()[0] for stage in step00.long_stage_order], bottom=[numpy.sum(output_data.loc[(output_data["Stage"] == stage) & (output_data["Taxonomy"].isin(used_taxa)), "mean"], axis=0) for stage in step00.long_stage_order], color=taxa_colors[taxon], label=label, error_kw={"elinewidth": 5})
        used_taxa.append(taxon)

    matplotlib.pyplot.grid(True)
    matplotlib.pyplot.xticks(range(len(step00.long_stage_order)), step00.long_stage_order, rotation="vertical")
    matplotlib.pyplot.ylim((0, 1))
    matplotlib.pyplot.ylabel(f"{len(taxa_list)} taxa proportions")
    matplotlib.pyplot.legend(loc="upper right", ncol=2, fontsize="xx-small")

    matplotlib.pyplot.tight_layout()

    fig.savefig(args.output)
    fig.savefig(args.output.replace(".png", ".pdf"))
    fig.savefig(args.output.replace(".png", ".svg"))
    matplotlib.pyplot.close(fig)
