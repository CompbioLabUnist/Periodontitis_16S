"""
step43.py: clinical data comparison
"""
import argparse
import itertools
import tarfile
import matplotlib
import matplotlib.pyplot
import pandas
import seaborn
import statannot
import tqdm
import step00


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("info", help="Information XLSX file", type=str)
    parser.add_argument("meta", help="Metadata CSV file", type=str)
    parser.add_argument("input", help="Input FASTQ.gz files", type=str, nargs="+")
    parser.add_argument("output", help="Output TAR files", type=str)

    args = parser.parse_args()

    if not args.info.endswith(".xlsx"):
        raise ValueError("INFO file must end with .xlsx!!")
    elif not args.meta.endswith(".csv"):
        raise ValueError("META file must end with .csv!!")
    elif list(filter(lambda x: not x.endswith(".fastq.gz"), args.input)):
        raise ValueError("INPUT file must end with .fastq.gz!!")
    elif not args.output.endswith(".tar"):
        raise ValueError("OUTPUT file must end with .tar!!")

    info_data = pandas.concat(pandas.read_excel(args.info, engine="openpyxl", sheet_name=None).values())
    print(info_data)

    metadata = pandas.read_csv(args.meta, index_col=3)
    print(metadata)

    data = pandas.DataFrame()
    data["#SampleID"] = list(map(step00.get_ID, list(filter(lambda x: x.endswith("_1.fastq.gz"), args.input))))
    data["Manage"] = list(map(lambda x: info_data.loc[(info_data["마크로젠 샘플번호"] == x), "검체 (수진) 번호"].to_numpy()[0], data["#SampleID"]))
    data["ShortStage"] = list(map(step00.change_ID_into_short_stage, data["#SampleID"]))
    data["LongStage"] = list(map(step00.change_short_into_long, data["ShortStage"]))
    print(data)

    clinical_columns = ["나이", "Plaque Index", "Gingival Index", "AL", "PD", "RE", "Count", "No. of teeth"]
    for column in tqdm.tqdm(clinical_columns):
        data[column] = list(map(lambda x: metadata.loc[x, column], data["Manage"]))
        data[column] = list(map(float, data[column]))
    print(data)

    matplotlib.use("Agg")
    matplotlib.rcParams.update(step00.matplotlib_parameters)
    seaborn.set_theme(context="poster", style="whitegrid", rc=step00.matplotlib_parameters)

    figures = list()
    order = ["Healthy", "Stage I", "Stage II", "Stage III"]

    for column in tqdm.tqdm(clinical_columns):
        fig, ax = matplotlib.pyplot.subplots(figsize=(18, 18))

        seaborn.violinplot(data=data, x="LongStage", y=column, order=order, palette=step00.color_stage_dict, ax=ax, inner="box", cut=1, linewidth=10)
        statannot.add_stat_annotation(ax, data=data, x="LongStage", y=column, order=order, test="t-test_ind", box_pairs=list(itertools.combinations(order, r=2)), text_format="star", loc="inside", verbose=0, comparisons_correction=None)

        matplotlib.pyplot.xlabel("")
        if column == clinical_columns[0]:
            matplotlib.pyplot.ylabel("Age")
        matplotlib.pyplot.tight_layout()

        figures.append(f"{column}.pdf")
        fig.savefig(figures[-1])
        matplotlib.pyplot.close(fig)

    with tarfile.open(args.output, "w") as tar:
        for file_name in tqdm.tqdm(figures):
            tar.add(file_name, arcname=file_name)
