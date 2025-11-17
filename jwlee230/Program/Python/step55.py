"""
step55.py: Venn diagram comparison
"""
import argparse
import matplotlib
import matplotlib.pyplot
import pandas
import seaborn
import venn
import step00

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", type=str, help="Input TAR.gz file", nargs=2)
    parser.add_argument("output", type=str, help="Output PNG file")

    args = parser.parse_args()

    matplotlib.use("Agg")
    matplotlib.rcParams.update(step00.matplotlib_parameters)
    seaborn.set_theme(context="poster", style="whitegrid", rc=step00.matplotlib_parameters)

    venn_data = dict()

    ancom_data = step00.read_pickle(args.input[0])
    venn_data["ANCOM"] = set(list(ancom_data.columns)[:-2])
    print(ancom_data)

    ancom_bc_data = step00.read_pickle(args.input[1])
    venn_data["ANCOM-BC2"] = set(list(ancom_bc_data.columns)[:-2])
    print(ancom_bc_data)

    print(sorted(venn_data["ANCOM"] & venn_data["ANCOM-BC2"]))

    fig, ax = matplotlib.pyplot.subplots(figsize=(6, 6))

    ax = venn.venn(venn_data, fmt="{size:d} ({percentage:.1f}%)", fontsize=12, ax=ax)
    matplotlib.pyplot.tight_layout()

    fig.savefig(args.output)
    fig.savefig(args.output.replace(".png", ".pdf"))
    fig.savefig(args.output.replace(".png", ".svg"))
    matplotlib.pyplot.close(fig)
