"""
step39.py: Draw rarefaction curve
"""
import argparse
import itertools
import pandas
import matplotlib
import matplotlib.colors
import matplotlib.pyplot
import seaborn
import tqdm
import step00


def get_depth_number(string):
    return int(string[6:string.find("_")])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", type=str, help="Input CSV file")
    parser.add_argument("output", type=str, help="Output PNG file")

    args = parser.parse_args()

    if not args.input.endswith(".csv"):
        raise ValueError("Input file must be a CSV file")
    elif not args.output.endswith(".png"):
        raise ValueError("Output file must end with .PNG!!")

    matplotlib.use("Agg")
    matplotlib.rcParams.update(step00.matplotlib_parameters)
    seaborn.set_theme(context="poster", style="whitegrid", rc=step00.matplotlib_parameters)

    input_data = pandas.read_csv(args.input, index_col=0)
    print(input_data)

    indices = sorted(input_data.index)
    columns = list(input_data.columns)

    output_data = pandas.DataFrame([(get_depth_number(column), input_data.loc[index, column], index) for index, column in tqdm.tqdm(list(itertools.product(indices, columns)))], columns=["depth", "diversity", "Sample"]).dropna()
    print(output_data)

    fig, ax = matplotlib.pyplot.subplots(figsize=(32, 18))

    seaborn.lineplot(data=output_data, x="depth", y="diversity", hue="Sample", palette=list(matplotlib.colors.XKCD_COLORS.keys())[:len(indices)], legend=False, ax=ax)

    depth = 3786
    matplotlib.pyplot.axvline(x=depth, color="k", linestyle="--")
    matplotlib.pyplot.text(x=depth, y=0, s=f"Sampling depth: {depth}", horizontalalignment="right", verticalalignment="bottom", rotation="vertical", fontsize="xx-small", color="k", bbox={"color": "white", "alpha": 0.4})
    matplotlib.pyplot.tight_layout()

    fig.savefig(args.output)
    fig.savefig(args.output.replace(".png", ".pdf"))
    fig.savefig(args.output.replace(".png", ".svg"))
    matplotlib.pyplot.close(fig)
