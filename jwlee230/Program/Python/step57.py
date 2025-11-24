"""
step57.py: Venn diagram for DB
"""
import argparse
import matplotlib
import matplotlib.pyplot
import pandas
import seaborn
import tqdm
import venn
import step00

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("korea", type=str, help="Input TSV file")
    parser.add_argument("spain", type=str, help="Input TSV file")
    parser.add_argument("portugal", type=str, help="Input TSV file")
    parser.add_argument("output", type=str, help="Output PNG file")

    args = parser.parse_args()

    matplotlib.use("Agg")
    matplotlib.rcParams.update(step00.matplotlib_parameters)
    seaborn.set_theme(context="poster", style="whitegrid", rc=step00.matplotlib_parameters)

    venn_data = dict()

    korea_data = pandas.read_csv(args.korea, sep="\t", index_col=0)
    korea_data = korea_data.loc[(korea_data["Reject null hypothesis"])]
    venn_data["Korea"] = set(korea_data.index)
    print(korea_data)

    spain_data = pandas.read_csv(args.spain, sep="\t", index_col=0)
    spain_data = spain_data.loc[(spain_data["Reject null hypothesis"])]
    venn_data["Spain"] = set(spain_data.index)
    print(spain_data)

    portugal_data = pandas.read_csv(args.portugal, sep="\t", index_col=0)
    portugal_data = portugal_data.loc[(portugal_data["Reject null hypothesis"])]
    venn_data["Portugal"] = set(portugal_data.index)
    print(portugal_data)

    fig, ax = matplotlib.pyplot.subplots(figsize=(6, 6))

    ax = venn.venn(venn_data, fmt="{size:d} ({percentage:.1f}%)", fontsize=12, ax=ax)
    matplotlib.pyplot.tight_layout()

    fig.savefig(args.output)
    fig.savefig(args.output.replace(".png", ".pdf"))
    fig.savefig(args.output.replace(".png", ".svg"))
    matplotlib.pyplot.close(fig)

    taxa_list = sorted(venn_data["Korea"] | venn_data["Spain"] | venn_data["Portugal"])
    output_data = pandas.DataFrame(index=taxa_list, columns=["Korea", "Spain", "Portugal"], dtype=str)
    for taxon in tqdm.tqdm(taxa_list):
        for column in ["Korea", "Spain", "Portugal"]:
            if taxon in venn_data[column]:
                output_data.loc[taxon, column] = "O"
            else:
                output_data.loc[taxon, column] = "X"
    # output_data.index = list(map(step00.simplified_taxonomy, taxa_list))
    print(output_data)
