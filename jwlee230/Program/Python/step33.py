"""
step33.py: clinical statistics
"""
import argparse
import numpy
import pandas
import scipy.stats
import step00

id_column = "검체 (수진) 번호"
stage_dict = {"Healthy": "Healthy", "CP_E": "Stage I", "CP_M": "Stage II", "CP_S": "Stage III"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", type=str, help="Input XLSX file")
    parser.add_argument("metadata", type=str, help="Metadata CSV file")
    parser.add_argument("output", type=str, help="Output CSV file")

    args = parser.parse_args()

    if not args.input.endswith(".xlsx"):
        raise ValueError("Input file must be a XLSX file")
    elif not args.metadata.endswith(".csv"):
        raise ValueError("Metadata file must be a CSV file")
    elif not args.output.endswith(".csv"):
        raise ValueError("Output file must be a CSV file")

    info_data = pandas.concat(pandas.read_excel(args.input, engine="openpyxl", sheet_name=None), ignore_index=True)
    info_ids = list(info_data[id_column])
    print(info_data)

    metadata = pandas.read_csv(args.metadata)
    metadata = metadata.loc[(metadata["Classification"].isin(stage_dict.keys()))]
    metadata["Stage"] = list(map(lambda x: stage_dict[x], list(metadata["Classification"])))
    meta_ids = list(metadata[id_column])
    print(metadata)
    print(sorted(metadata.columns))

    assert not (set(info_ids) - set(meta_ids))

    metadata = metadata.loc[(metadata[id_column].isin(info_ids))]
    print(metadata)

    raw_output_data = list()
    for clinical_column in ["나이", "Plaque Index", "Gingival Index", "PD", "AL", "RE", "No. of teeth"]:
        d = [clinical_column]
        for stage in step00.long_stage_order:
            selected_data = metadata.loc[(metadata["Stage"] == stage), clinical_column]
            d.append(f"{numpy.mean(selected_data):.2f}±{numpy.std(selected_data):.2f}")
        p = scipy.stats.kruskal(*[metadata.loc[(metadata["Stage"] == stage), clinical_column] for stage in step00.long_stage_order])[1]
        d.append(f"{p:.2e}")
        raw_output_data.append(d)

    true_values = {"성별": "남", "항생제": "예", "치과치료 유무": "예", "스케일링 유무": "예"}
    for clinical_column in ["성별", "항생제", "치과치료 유무", "스케일링 유무"]:
        d = [clinical_column]
        for stage in step00.long_stage_order:
            true_count = len(metadata.loc[(metadata["Stage"] == stage) & (metadata[clinical_column] == true_values[clinical_column])])
            false_count = len(metadata.loc[(metadata["Stage"] == stage) & (metadata[clinical_column] != true_values[clinical_column])])
            proportion = true_count / (true_count + false_count) * 100
            d.append(f"{true_count} ({proportion:.1f}%)")
        d.append("NA")
        raw_output_data.append(d)

    output_data = pandas.DataFrame(raw_output_data, columns=["Clinical", "Healthy", "Stage I", "Stage II", "Stage III", "p-value"])
    print(output_data)

    output_data.to_csv(args.output, index=False, encoding="utf-8-sig")
    output_data.to_latex(args.output.replace(".csv", ".tex"), index=False, column_format="llrr")
