"""
step34.py: Draw correlation with clinical data
"""
import argparse
import itertools
import tarfile
import matplotlib
import matplotlib.colors
import matplotlib.pyplot
import numpy
import pandas
import scipy.stats
import seaborn
import step00

id_column = "검체 (수진) 번호"
sample_column = "마크로젠 샘플번호"
stage_dict = {"Healthy": "Healthy", "CP_E": "Stage I", "CP_M": "Stage II", "CP_S": "Stage III"}
clinical_columns = ["나이", "Plaque Index", "Gingival Index", "PD", "AL", "RE", "No. of teeth"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", type=str, help="Input TAR.gz file")
    parser.add_argument("sample", type=str, help="Input TSV file")
    parser.add_argument("metadata", type=str, help="Metadata CSV file")
    parser.add_argument("output", type=str, help="Output TAR file")

    args = parser.parse_args()

    if not args.input.endswith(".tar.gz"):
        raise ValueError("Input file must be TAR.gz file")
    elif not args.sample.endswith(".xlsx"):
        raise ValueError("Sample file must be a XLSX file")
    elif not args.metadata.endswith(".csv"):
        raise ValueError("Metadata file must be a CSV file")
    elif not args.output.endswith(".tar"):
        raise ValueError("Output file must be a TAR file")

    matplotlib.use("Agg")
    matplotlib.rcParams.update(step00.matplotlib_parameters)
    seaborn.set(context="poster", style="whitegrid", rc=step00.matplotlib_parameters)

    input_data = step00.read_pickle(args.input)
    print(input_data)

    input_data.columns = list(map(lambda x: " ".join(x.split("; ")[-2:]), list(input_data.columns)[:-2])) + list(input_data.columns)[-2:]
    print(list(input_data.columns))

    taxa = sorted(list(input_data.columns)[:-2])
    for index in list(input_data.index):
        input_data.loc[index, taxa] = input_data.loc[index, taxa] / sum(input_data.loc[index, taxa])
    print(input_data)

    info_data = pandas.concat(pandas.read_excel(args.sample, engine="openpyxl", sheet_name=None), ignore_index=True)
    info_ids = list(info_data[id_column])
    print(info_data)
    print(sorted(info_data.columns))

    metadata = pandas.read_csv(args.metadata)
    metadata = metadata.loc[(metadata["Classification"].isin(stage_dict.keys()))]
    metadata["Stage"] = list(map(lambda x: stage_dict[x], list(metadata["Classification"])))
    meta_ids = list(metadata[id_column])
    print(metadata)
    print(sorted(metadata.columns))

    assert not (set(info_ids) - set(meta_ids))

    metadata = metadata.loc[(metadata[id_column].isin(info_ids))]
    metadata[sample_column] = list(map(lambda x: info_data.loc[(info_data[id_column] == x), sample_column].to_numpy()[0], metadata[id_column]))
    print(metadata)

    for clinical in clinical_columns:
        input_data[clinical] = list(map(lambda x: metadata.loc[(metadata[sample_column] == x), clinical].to_numpy()[0], input_data.index))
    print(input_data)

    figures = list()
    for clinical, taxon in itertools.product(clinical_columns, taxa):
        stat, p = scipy.stats.pearsonr(input_data[clinical], input_data[taxon])

        if (abs(stat) < 0.5) or (p > 0.05):
            continue

        fig, ax = matplotlib.pyplot.subplots(figsize=(24, 24))

        seaborn.scatterplot(data=input_data, x=clinical, y=taxon, hue="LongStage", hue_order=step00.long_stage_order, palette=step00.color_stage_dict, legend="brief", s=400, ax=ax)
        matplotlib.pyplot.axline((numpy.mean(input_data[clinical]), numpy.mean(input_data[taxon])), slope=stat, color="k", linestyle="--", linewidth=4)

        matplotlib.pyplot.xlabel(f"{clinical}")
        matplotlib.pyplot.ylabel(f"{taxon} proportion")
        matplotlib.pyplot.legend(title="")
        matplotlib.pyplot.title(f"r={stat:.3f}, p={p:.3f}")
        matplotlib.pyplot.tight_layout()

        figures.append(f"{clinical}+{taxon}.pdf")
        fig.savefig(figures[-1])
        matplotlib.pyplot.close(fig)

    with tarfile.open(args.output, "w") as tar:
        for f in figures:
            tar.add(f, arcname=f)
