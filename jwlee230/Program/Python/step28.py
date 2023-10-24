"""
step28.py: ANCOM cluster map
"""
import argparse
import matplotlib
import matplotlib.pyplot
import numpy
import pandas
import seaborn
import step00

id_column = "검체 (수진) 번호"
stage_dict = {"Healthy": "Healthy", "CP_E": "Stage I", "CP_M": "Stage II", "CP_S": "Stage III"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", type=str, help="Input TAR.gz file")
    parser.add_argument("info", type=str, help="Info data XLSX file")
    parser.add_argument("metadata", type=str, help="Metadata CSV file")
    parser.add_argument("output", type=str, help="Output PNG file")

    args = parser.parse_args()

    if not args.info.endswith(".xlsx"):
        raise ValueError("Info file must be a XLSX file")
    elif not args.metadata.endswith(".csv"):
        raise ValueError("Metadata file must be a CSV file")
    elif not args.output.endswith(".png"):
        raise ValueError("OUTPUT file must end with .PNG!!")

    matplotlib.use("Agg")
    matplotlib.rcParams.update(step00.matplotlib_parameters)
    seaborn.set(context="poster", style="whitegrid", rc=step00.matplotlib_parameters)

    data = step00.read_pickle(args.input)
    print(data)

    data = data.iloc[:, :-2]
    data.columns = list(map(lambda x: " ".join(x.split("; ")[-2:]).replace("_", " "), list(data.columns)))
    data = data.T
    data = numpy.log10(data + 1)
    print(data)

    info_data = pandas.concat(pandas.read_excel(args.info, engine="openpyxl", sheet_name=None), ignore_index=True)
    info_ids = list(info_data[id_column])
    print(info_data)

    metadata = pandas.read_csv(args.metadata)
    metadata = metadata.loc[(metadata["Classification"].isin(stage_dict.keys()))]
    metadata["Stage"] = list(map(lambda x: stage_dict[x], list(metadata["Classification"])))
    meta_ids = list(metadata[id_column])
    print(metadata)

    assert not (set(info_ids) - set(meta_ids))

    metadata = metadata.loc[(metadata[id_column].isin(info_ids))]
    metadata["Index"] = list(map(lambda x: info_data.loc[(info_data[id_column] == x), "마크로젠 샘플번호"].to_numpy()[0], metadata[id_column]))
    metadata = metadata.set_index("Index").sort_values(by=["Stage", "AL"])
    print(metadata["AL"])

    data = data.loc[:, metadata.index]

    g = seaborn.clustermap(data=data, xticklabels=False, yticklabels=True, row_cluster=True, col_cluster=False, col_colors=list(map(lambda x: step00.color_stage_dict[step00.change_ID_into_long_stage(x)], list(data.columns))), figsize=(32, 32), cmap="YlOrRd", cbar_pos=(0.90, 0.8, 0.05, 0.18), dendrogram_ratio=(0.2, 0.01))

    g.savefig(args.output)
    g.savefig(args.output.replace(".png", ".pdf"))
    g.savefig(args.output.replace(".png", ".svg"))
