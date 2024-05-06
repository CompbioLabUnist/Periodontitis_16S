"""
step07.py: make metadata
"""
import argparse
import pandas
import tqdm
import step00

portugal_ID_set = set()
spain_ID_set = set()


def change_ID_into_short_stage(ID: str) -> str:
    if not ID.startswith("SRR"):
        return step00.change_ID_into_short_stage(ID)
    elif ID in portugal_ID_set:
        d = {"not applicable": "NA", "Perio_Grade_0": "H", "Perio_Grade_1": "Sli", "Perio_Grade_2": "M", "Perio_Grade_3": "S"}
        return d[portugal_info_data.loc[ID, "Grades_Periodontal_Health"]]
    elif ID in spain_ID_set:
        d = {"gingivitis": "NA", "health": "H", "stage_I_periodontitis": "Sli", "stage_II_periodontitis": "M", "stage_III_periodontitis": "S", "stage_IV_periodontitis": "NA"}
        return d[spain_info_data.loc[ID, "periodontal_health_status"]]

    raise ValueError("ID not found!!")


def check_DB(ID: str) -> str:
    if not ID.startswith("SRR"):
        return "Korea"
    elif ID in portugal_ID_set:
        return "Portugal"
    elif ID in spain_ID_set:
        return "Spain"

    raise ValueError("ID not found!!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("info", help="Information XLSX file", type=str)
    parser.add_argument("portugal", help="Portugal Information XLSX file", type=str)
    parser.add_argument("spain", help="Spain Information XLSX file", type=str)
    parser.add_argument("input", help="Input FASTQ.gz files", type=str, nargs="+")
    parser.add_argument("output", help="Output TSV files", type=str)

    args = parser.parse_args()

    if not args.info.endswith(".xlsx"):
        raise ValueError("INFO file must end with .xlsx!!")
    elif not args.portugal.endswith(".xlsx"):
        raise ValueError("Portugal Information XLSX file must end with .xlsx!!")
    elif not args.spain.endswith(".xlsx"):
        raise ValueError("Spain Information XLSX file must end with .xlsx!!")
    elif list(filter(lambda x: not x.endswith(".fastq.gz"), args.input)):
        raise ValueError("INPUT file must end with .fastq.gz!!")
    elif not args.output.endswith(".tsv"):
        raise ValueError("OUTPUT file must end with .tsv!!")

    info_data = pandas.concat(pandas.read_excel(args.info, engine="openpyxl", sheet_name=None).values())
    portugal_info_data = pandas.read_excel(args.portugal, engine="openpyxl", index_col="Run")
    spain_info_data = pandas.read_excel(args.spain, engine="openpyxl", index_col="Run")

    portugal_ID_set = set(portugal_info_data.index)
    spain_ID_set = set(spain_info_data.index)

    print(info_data)
    print(portugal_info_data)
    print(spain_info_data)

    data = pandas.DataFrame()
    data["#SampleID"] = list(map(step00.get_ID, list(filter(lambda x: x.endswith("_1.fastq.gz"), args.input))))
    data["BarcodeSequence"] = ""
    data["LinkPrimerSequence"] = ""

    data["ShortStage"] = list(map(change_ID_into_short_stage, data["#SampleID"]))
    data["LongStage"] = list(map(step00.change_short_into_long, data["ShortStage"]))
    data["DB"] = list(map(check_DB, data["#SampleID"]))

    data["Description"] = ""
    print(data)

    with open(args.output, "w") as f:
        f.write("\t".join(list(data.columns)))
        f.write("\n")
        f.write("#q2:types\t")
        f.write("\t".join(["categorical"] * (len(data.columns) - 1)))
        f.write("\n")

        for index, row in tqdm.tqdm(data.iterrows(), total=len(data)):
            f.write("\t".join(row))
            f.write("\n")
