"""
step37.py: Draw violin plot by alpha-diversity -- Extended
"""
import argparse
import itertools
import tarfile
import matplotlib
import matplotlib.pyplot
import pandas
import scipy.stats
import seaborn
import statannotations.Annotator
import tqdm
import step00

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", type=str, help="Input TSV file")
    parser.add_argument("output", type=str, help="Output TAR file")

    args = parser.parse_args()

    if not args.input.endswith(".tsv"):
        raise ValueError("INPUT must end with .tsv!!")
    elif not args.output.endswith(".tar"):
        raise ValueError("OUTPUT must end with .tar!!")

    input_data = pandas.read_csv(args.input, sep="\t", index_col=0)
    input_data["ShortStage"] = list(map(step00.change_ID_into_short_stage, list(input_data.index)))
    input_data["LongStage"] = list(map(step00.change_short_into_long, input_data["ShortStage"]))
    print(input_data)

    matplotlib.use("Agg")
    matplotlib.rcParams.update(step00.matplotlib_parameters)
    seaborn.set(context="poster", style="whitegrid", rc=step00.matplotlib_parameters)

    figures = list()
    for alpha in tqdm.tqdm(step00.alphas):
        box_pairs = list()
        for s1, s2 in itertools.combinations(step00.long_stage_order, r=2):
            _, p = scipy.stats.ttest_ind(input_data.loc[(input_data["LongStage"] == s1), alpha], input_data.loc[(input_data["LongStage"] == s2), alpha])
            if (p < 0.05):
                box_pairs.append((s1, s2))

        fig, ax = matplotlib.pyplot.subplots(figsize=(18, 18))

        seaborn.violinplot(data=input_data, x="LongStage", y=alpha, order=step00.long_stage_order, ax=ax, inner="box", palette=step00.color_stage_dict, cut=1, linewidth=10)
        statannotations.Annotator.Annotator(ax, box_pairs, data=input_data, x="LongStage", y=alpha, order=step00.long_stage_order).configure(test="t-test_ind", text_format="star", loc="inside", comparisons_correction=None, verbose=0).apply_and_annotate()

        stat, p = scipy.stats.kruskal(*[input_data.loc[(input_data["LongStage"] == stage), alpha] for stage in step00.long_stage_order])

        matplotlib.pyplot.xlabel("")
        matplotlib.pyplot.ylabel(alpha.replace("_", " "))
        matplotlib.pyplot.title(f"Kruskal-Wallis p={p:.2e}")
        matplotlib.pyplot.tight_layout()

        figures.append(f"{alpha}.pdf")
        fig.savefig(figures[-1])
        matplotlib.pyplot.close(fig)

    with tarfile.open(args.output, "w") as tar:
        for figure in tqdm.tqdm(figures):
            tar.add(figure, arcname=figure)
