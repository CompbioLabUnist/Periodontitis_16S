"""
step07.py: make metadata
"""
import argparse
import pandas

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("info", help="Information XLSX file", type=str)
    parser.add_argument("input", help="Input FASTQ.gz files", type=str, nargs="+")

    args = parser.parse_args()

    if not args.info.endswith(".xlsx"):
        raise ValueError("INFO file must end with .xlsx")
    elif list(filter(lambda x: x.endswith(".fastq.gz"), args.input)):
        raise ValueError("INPUT file must end with .fastq.gz!!")

    info_data = pandas.read_excel(args.info)

    data = pandas.DataFrame()
    data["#SampleID"] = sorted(list(set(list(map(lambda x: x.split("_")[0], list(map(lambda x: x.split("/")[-1], args.input)))))))
    data["BarcodeSequence"] = ""
    data["LinkPrimerSequence"] = ""

    data["ShortStage"] = list(map(lambda x: x[0] if x[0] == "H" else x[2], data["#SampleID"]))
    data["LongStage"] = list(map(lambda x: {"H": "Healthy", "E": "Early", "M": "Moderate", "S": "Severe"}[x], data["ShortStage"]))

    data["Description"] = ""

    print("\t".join(data.columns))
    print("#q2:types", "\t".join(["categorical"] * (len(data.columns) - 1)), sep="\t")
    for index, row in data.iterrows():
        print("\t".join(row))
