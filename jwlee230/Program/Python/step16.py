"""
step16.py: Clearify raw TSV into pandas
"""
import argparse
import pandas
import step00


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", help="Input TSV file", type=str)
    parser.add_argument("clinical", help="Clinical TSV file", type=str)
    parser.add_argument("output", help="Output TAR.gz file", type=str)

    args = parser.parse_args()

    if not args.input.endswith(".tsv"):
        raise ValueError("Input file must end with .TSV!!")
    elif not args.clinical.endswith(".tsv"):
        raise ValueError("Clinical file must end with .TSV!!")

    raw_data = pandas.read_csv(args.input, sep="\t", skiprows=1)
    raw_data["taxonomy"] = list(map(step00.consistency_taxonomy, list(raw_data["taxonomy"])))
    print(raw_data)

    clinical_data = pandas.read_csv(args.clinical, sep="\t", index_col=0, skiprows=[1])
    print(clinical_data)

    data = raw_data.groupby(by="taxonomy").sum().T
    data = data.loc[sorted(list(data.index)), :]
    print(data)

    data["ShortStage"] = clinical_data.loc[data.index, "ShortStage"]
    data["LongStage"] = clinical_data.loc[data.index, "LongStage"]
    print(data)

    step00.make_pickle(args.output, data)
