"""
step41.py: Taxonomy TSV output
"""
import argparse
import pandas
import step00


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", type=str, help="Input TAR.gz file")
    parser.add_argument("output", type=str, help="Output TSV file")

    args = parser.parse_args()

    if not args.output.endswith(".tsv"):
        raise ValueError("OUTPUT file must end with .TSV!!")

    data: pandas.DataFrame = step00.read_pickle(args.input)
    del data["ShortStage"], data["LongStage"]
    print(data)

    data = data.sort_index(axis="columns")
    data["Periodontitis stage"] = list(map(step00.change_ID_into_long_stage, list(data.index)))
    data.index.name = "Sample ID"
    print(data)

    data.to_csv(args.output, sep="\t")
