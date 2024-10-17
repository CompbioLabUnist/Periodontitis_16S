"""
step46.py: Draw violin plot by abundance
"""
import argparse
import itertools
import matplotlib
import matplotlib.pyplot
import numpy
import pandas
import scipy.stats
import seaborn
import statannotations.Annotator
import tqdm
import step00

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", help="Input TAR.gz file", type=str)
    parser.add_argument("output", type=str, help="Output PNG file")

    args = parser.parse_args()

    if not args.output.endswith(".png"):
        raise ValueError("OUTPUT must end with .png!!")

    data = step00.read_pickle(args.input)
    data.columns = list(map(lambda x: " ".join(x.split("; ")[-2:]).replace("_", " "), list(data.columns)[:-2])) + list(data.columns)[-2:]
    print(data)

    taxa = list(data.columns)[:-2]
    raw_output_data = list()
    for index, row in tqdm.tqdm(data.iterrows()):
        for taxon in taxa:
            raw_output_data.append((taxon, row["LongStage"], row[taxon]))
    output_data = pandas.DataFrame(raw_output_data, columns=["taxonomy", "LongStage", "Value"])
    print(output_data)

    box_pairs = list()
    for taxon, (s1, s2) in tqdm.tqdm(list(itertools.product(taxa, itertools.combinations(step00.long_stage_order[-2:], r=2)))):
        _, p = scipy.stats.mannwhitneyu(data.loc[(data["LongStage"] == s1), taxon], data.loc[(data["LongStage"] == s2), taxon])
        if (p < 0.05):
            box_pairs.append(((taxon, s1), (taxon, s2)))
    print(len(box_pairs))

    matplotlib.use("Agg")
    matplotlib.rcParams.update(step00.matplotlib_parameters)
    seaborn.set(context="poster", style="whitegrid", rc=step00.matplotlib_parameters)

    fig, ax = matplotlib.pyplot.subplots(figsize=(36, 24))

    seaborn.boxplot(data=output_data, x="taxonomy", y="Value", hue="LongStage", order=taxa, hue_order=step00.long_stage_order[1:], palette=step00.color_stage_dict, showfliers=False, ax=ax)
    statannotations.Annotator.Annotator(ax, box_pairs, data=output_data, x="taxonomy", y="Value", hue="LongStage", order=taxa, hue_order=step00.long_stage_order[1:]).configure(test="Mann-Whitney", text_format="star", loc="inside", comparisons_correction=None, verbose=0).apply_and_annotate()

    matplotlib.pyplot.xlabel("")
    matplotlib.pyplot.ylabel("Abundance")
    matplotlib.pyplot.xticks(rotation="vertical")
    matplotlib.pyplot.legend(loc="upper center")
    matplotlib.pyplot.tight_layout()

    fig.savefig(args.output)
    fig.savefig(args.output.replace(".png", ".pdf"))
    fig.savefig(args.output.replace(".png", ".svg"))
    matplotlib.pyplot.close(fig)
