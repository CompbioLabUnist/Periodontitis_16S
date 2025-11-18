"""
step56-1.py: Bar plot comparisons
"""
import argparse
import itertools
import pandas
import matplotlib
import matplotlib.pyplot
import seaborn
import statannotations.Annotator
import tqdm
import step00

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", type=str, nargs="+", help="Input STR file(s)")
    parser.add_argument("output", type=str, help="Output PNG file")
    parser.add_argument("--name", type=str, nargs="+", help="Name of databases")

    args = parser.parse_args()

    if len(args.input) != len(args.name):
        raise ValueError("Input and Name should be corresponding!")

    matplotlib.use("Agg")
    matplotlib.rcParams.update(step00.matplotlib_parameters)
    seaborn.set_theme(context="poster", style="whitegrid"
                      , rc=step00.matplotlib_parameters)

    tmp_data_list = list()
    for input_file, name in tqdm.tqdm(list(zip(args.input, args.name))):
        tmp_data = pandas.read_csv(input_file, sep="\t", index_col=0)
        tmp_data["DB"] = name
        tmp_data_list.append(tmp_data)

    input_data = pandas.concat(tmp_data_list, ignore_index=True)
    print(input_data)

    fig, ax = matplotlib.pyplot.subplots(figsize=(18, 18))

    metrics = sorted(step00.selected_derivations + ["AUC"])
    seaborn.barplot(data=input_data, x="Metrics", order=metrics, y="Value", hue="DB", hue_order=args.name, ax=ax)
    statannotations.Annotator.Annotator(ax, list(map(lambda x: ((x[0], x[1][0]), (x[0], x[1][1])), itertools.product(metrics, itertools.combinations(args.name, r=2)))), data=input_data, x="Metrics", order=metrics, y="Value", hue="DB", hue_order=args.name).configure(test="Mann-Whitney", text_format="star", loc="inside", comparisons_correction=None, verbose=True).apply_and_annotate()

    matplotlib.pyplot.xlabel("")
    matplotlib.pyplot.tight_layout()

    fig.savefig(args.output)
    fig.savefig(args.output.replace(".png", ".pdf"))
    matplotlib.pyplot.close(fig)
