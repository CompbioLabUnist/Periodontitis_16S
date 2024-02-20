"""
step39.py: Draw rarefaction curve
"""
import argparse
import itertools
import pandas
import matplotlib
import matplotlib.colors
import matplotlib.pyplot
import numpy
import seaborn
import tqdm
import step00


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", type=str, help="Input CSV file")
    parser.add_argument("clinical", help="Clinical TSV file", type=str)
    parser.add_argument("output", type=str, help="Output PNG file")
    parser.add_argument("--depth", help="Depth to plot", type=int, default=1)

    args = parser.parse_args()

    if not args.input.endswith(".csv"):
        raise ValueError("Input file must be a CSV file")
    elif not args.clinical.endswith(".tsv"):
        raise ValueError("Clinical file must end with .TSV!!")
    elif not args.output.endswith(".png"):
        raise ValueError("Output file must end with .PNG!!")

    matplotlib.use("Agg")
    matplotlib.rcParams.update(step00.matplotlib_parameters)
    seaborn.set_theme(context="poster", style="whitegrid", rc=step00.matplotlib_parameters)

    input_data = pandas.read_csv(args.input, index_col=0)
    print(input_data)

    clinical_data = pandas.read_csv(args.clinical, sep="\t", index_col=0, skiprows=[1])
    print(clinical_data)

    indices = sorted(input_data.index)
    columns = list(input_data.columns)
    column_set = list(set(map(lambda x: x[:x.find("_")], columns)))

    output_data = pandas.DataFrame([(int(column[6:]), numpy.mean(input_data.loc[index, list(filter(lambda x: x.startswith(column), columns))]), index) for index, column in tqdm.tqdm(list(itertools.product(indices, column_set)))], columns=["depth", "diversity", "Sample"]).dropna()
    print(output_data.loc[(output_data["depth"] == 1)])
    output_data["Stage"] = list(map(lambda x: clinical_data.loc[x, "LongStage"], output_data["Sample"]))
    print(output_data)

    fig, ax = matplotlib.pyplot.subplots(figsize=(32, 18))

    seaborn.lineplot(data=output_data, x="depth", y="diversity", hue="Stage", hue_order=["Healthy", "Stage I", "Stage II", "Stage III"], palette=step00.color_stage_dict, legend="full", units="Sample", estimator=None, ax=ax)

    matplotlib.pyplot.axvline(x=args.depth, color="k", linestyle="--")
    matplotlib.pyplot.text(x=args.depth, y=10, s=f"Sampling depth: {args.depth}", horizontalalignment="right", verticalalignment="bottom", rotation="vertical", fontsize="xx-small", color="k", bbox={"color": "white", "alpha": 0.4})
    matplotlib.pyplot.ylim(bottom=0)
    matplotlib.pyplot.tight_layout()

    fig.savefig(args.output)
    fig.savefig(args.output.replace(".png", ".pdf"))
    fig.savefig(args.output.replace(".png", ".svg"))
    matplotlib.pyplot.close(fig)
