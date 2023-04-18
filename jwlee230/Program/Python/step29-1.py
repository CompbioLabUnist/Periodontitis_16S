"""
step29-1.py: ANCOM violin plot with taxa
"""
import argparse
import itertools
import tarfile
import matplotlib
import matplotlib.pyplot
import scipy.stats
import seaborn
import statannotations.Annotator
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

    data = step00.read_pickle(args.input)
    for i, index in enumerate(data.index):
        data.iloc[i, :-2] = data.iloc[i, :-2] / sum(data.iloc[i, :-2])
    data.columns = list(map(lambda x: " ".join(x.split("; ")[-2:]).replace("_", " "), list(data.columns)[:-2])) + list(data.columns)[-2:]
    print(data)

    taxa = list(data.columns)[:-2]

    figures = list()
    for taxon in taxa:
        stat, p = scipy.stats.kruskal(*[data.loc[(data["LongStage"] == stage), taxon] for stage in step00.long_stage_order])

        fig, ax = matplotlib.pyplot.subplots(figsize=(24, 24))

        seaborn.violinplot(data=data, x="LongStage", y=taxon, order=step00.long_stage_order, palette=step00.color_stage_dict, inner="box", cut=1, linewidth=10, ax=ax)
        statannotations.Annotator.Annotator(ax, list(itertools.combinations(step00.long_stage_order, r=2)), data=data, x="LongStage", y=taxon, order=step00.long_stage_order).configure(test="Mann-Whitney", text_format="star", comparisons_correction=None, loc="inside", verbose=0).apply_and_annotate()

        matplotlib.pyplot.title(f"Kruskal-Wallis p={p:.2e}")
        matplotlib.pyplot.xlabel("")
        matplotlib.pyplot.tight_layout()

        figures.append(f"{taxon}.pdf")
        fig.savefig(figures[-1])
        matplotlib.pyplot.close(fig)

    with tarfile.open(args.output, "w") as tar:
        for f in figures:
            tar.add(f, arcname=f)
