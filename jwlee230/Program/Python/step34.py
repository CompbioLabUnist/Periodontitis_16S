"""
step34.py: Draw correlation with clinical data
"""
import argparse
import itertools
import tarfile
import matplotlib
import matplotlib.colors
import matplotlib.pyplot
import pandas
import scipy.stats
import seaborn
import statannot
import tqdm
import step00


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", type=str, help="Input TAR.gz file")
    parser.add_argument("output", type=str, help="Output TAR file")

    args = parser.parse_args()

    if not args.input.endswith(".tar.gz"):
        raise ValueError("Input file must be TAR.gz file")
    elif not args.output.endswith(".tar"):
        raise ValueError("Output file must be a TAR file")

    matplotlib.use("Agg")
    matplotlib.rcParams.update(step00.matplotlib_parameters)
    seaborn.set(context="poster", style="whitegrid", rc=step00.matplotlib_parameters)

    input_data = step00.read_pickle(args.input)
    print(input_data)

    input_data.columns = list(map(lambda x: " ".join(x.split("; ")[-2:]), list(input_data.columns)[:-2])) + list(input_data.columns)[-2:]
    print(list(input_data.columns))

    taxa = list(input_data.columns)[:-2]
    for index in tqdm.tqdm(list(input_data.index)):
        input_data.loc[index, taxa] = input_data.loc[index, taxa] / sum(input_data.loc[index, taxa])
    print(input_data)

    input_data["ShortStage"] = list(map(lambda x: {"H": 0, "Sli": 1, "M": 2, "S": 3}[x], input_data["ShortStage"]))
    input_data = input_data.loc[(input_data["LongStage"].isin(step00.long_stage_order[1:]))]
    print(input_data)

    raw_output_data = list()

    figures = list()
    for taxon in tqdm.tqdm(taxa):
        stat, p = scipy.stats.spearmanr(input_data["ShortStage"], input_data[taxon])
        raw_output_data.append((taxon, stat))

        fig, ax = matplotlib.pyplot.subplots(figsize=(18, 18))

        seaborn.violinplot(data=input_data, x="LongStage", y=taxon, order=step00.long_stage_order[1:], palette=step00.color_stage_dict, cut=1, linewidth=5, ax=ax)
        statannot.add_stat_annotation(ax, data=input_data, x="LongStage", y=taxon, order=step00.long_stage_order[1:], test="Mann-Whitney", box_pairs=list(itertools.combinations(step00.long_stage_order[1:], r=2)), text_format="star", loc="inside", verbose=0, comparisons_correction=None)

        # matplotlib.pyplot.title(f"r={stat:.3f}, p={p:.3f}")
        matplotlib.pyplot.xlabel("")
        matplotlib.pyplot.ylabel(f"{taxon}")
        matplotlib.pyplot.tight_layout()

        figures.append(f"{taxon}.pdf")
        fig.savefig(figures[-1])
        matplotlib.pyplot.close(fig)

    output_data = pandas.DataFrame(raw_output_data, columns=["Taxonomy", "coef"]).sort_values("coef")
    print(output_data)

    with tarfile.open(args.output, "w") as tar:
        for f in tqdm.tqdm(figures):
            tar.add(f, arcname=f)
