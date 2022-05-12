"""
step17.py: Select by ANCOM
"""
import argparse
import pandas
import step00


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", help="Input TAR.gz file", type=str)
    parser.add_argument("ancom", help="Input TSV file", type=str)
    parser.add_argument("output", help="Output TAR.gz file", type=str)

    args = parser.parse_args()

    if not args.ancom.endswith(".tsv"):
        raise ValueError("ANCOM file must end with .TSV!!")

    data = step00.read_pickle(args.input)
    print(list(data.columns))
    print(data)

    ancom_data = pandas.read_csv(args.ancom, sep="\t", names=["Bacteria", "W", "Reject"], header=0)
    ancom_data["Bacteria"] = list(map(step00.consistency_taxonomy, ancom_data["Bacteria"]))
    print(ancom_data)

    taxa = list(map(lambda x: x[0], list(filter(lambda x: x[1], zip(ancom_data["Bacteria"], ancom_data["Reject"])))))
    data = data[taxa + ["ShortStage", "LongStage"]]
    print(data)

    step00.make_pickle(args.output, data)
