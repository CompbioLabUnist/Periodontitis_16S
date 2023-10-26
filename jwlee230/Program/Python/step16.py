"""
step16.py: Clearify raw TSV into pandas
"""
import argparse
import pandas
import step00


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", help="Input TSV file", type=str)
    parser.add_argument("output", help="Output TAR.gz file", type=str)

    args = parser.parse_args()

    if not args.input.endswith(".tsv"):
        raise ValueError("Input file must end with .TSV!!")

    raw_data = pandas.read_csv(args.input, sep="\t", skiprows=1)
    raw_data["taxonomy"] = list(map(step00.consistency_taxonomy, list(raw_data["taxonomy"])))
    print(raw_data)

    data = raw_data.groupby(by="taxonomy").sum().T
    data = data.loc[sorted(list(data.index), key=step00.sorting), :]
    print(data)

    data["ShortStage"] = list(map(step00.change_ID_into_short_stage, list(data.index)))
    data["LongStage"] = list(map(step00.change_short_into_long, data["ShortStage"]))
    print(data)

    step00.make_pickle(args.output, data)
