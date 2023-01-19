"""
step30.py: Prepare TSV for LefSe
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
    data["!Subject"] = data["LongStage"]
    del data["ShortStage"]
    del data["LongStage"]
    data = data.T.sort_index()
    data.index.name = "taxonomy"
    print(data)

    data.to_csv(args.output, sep="\t")
