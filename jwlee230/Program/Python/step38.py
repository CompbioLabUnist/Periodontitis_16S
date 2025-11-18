"""
step38.py: Calculate extended beta-diversity
"""
import argparse
import itertools
import tarfile
import typing
import matplotlib
import matplotlib.pyplot
import pandas
import seaborn
import statannotations.Annotator
import tqdm
import step00

input_data = pandas.DataFrame()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", type=str, help="Input TSV file")
    parser.add_argument("output", type=str, help="Output TAR file")

    args = parser.parse_args()

    if not args.input.endswith(".tsv"):
        raise ValueError("INPUT must end with .tsv!!")
    elif not args.output.endswith(".tar"):
        raise ValueError("OUTPUT must end with .tar!!")

    input_data = pandas.read_csv(args.input, sep="\t", index_col=0)
    input_data = input_data.loc[list(filter(lambda x: not x.startswith("SRR"), list(input_data.index))), list(filter(lambda x: not x.startswith("SRR"), list(input_data.columns)))]
    print(input_data)

    sample_by_stage_dict: typing.Dict[str, typing.List[str]] = {x: list() for x in step00.long_stage_order[1:]}
    for sample in tqdm.tqdm(list(input_data.index)):
        sample_by_stage_dict[step00.change_ID_into_long_stage(sample)].append(sample)

    matplotlib.use("Agg")
    matplotlib.rcParams.update(step00.matplotlib_parameters)
    seaborn.set(context="poster", style="whitegrid", rc=step00.matplotlib_parameters)

    figures = list()
    for base_stage in tqdm.tqdm(step00.long_stage_order[1:]):
        raw_output = list()
        for stage in step00.long_stage_order[1:]:
            raw_output += [(stage, input_data.loc[a, b]) for a, b in itertools.product(sample_by_stage_dict[base_stage], sample_by_stage_dict[stage])]

        output_data = pandas.DataFrame(raw_output, columns=["stage", "distance"])

        fig, ax = matplotlib.pyplot.subplots(figsize=(18, 18))
        seaborn.violinplot(data=output_data, x="stage", y="distance", order=step00.long_stage_order[1:], ax=ax, inner="box", palette=step00.color_stage_dict, cut=1, linewidth=10)
        # statannotations.Annotator.Annotator(ax, list(itertools.combinations(step00.long_stage_order, r=2)), data=output_data, x="stage", y="distance", order=step00.long_stage_order).configure(test="t-test_ind", text_format="star", loc="inside", comparisons_correction=None, verbose=0).apply_and_annotate()
        compare_list = list(filter(lambda x: x[0] == base_stage, list(itertools.combinations(step00.long_stage_order[1:], r=2))))
        statannotations.Annotator.Annotator(ax, compare_list, data=output_data, x="stage", y="distance", order=step00.long_stage_order[1:]).configure(test="Mann-Whitney", text_format="star", loc="inside", comparisons_correction=None, verbose=True).apply_and_annotate()

        matplotlib.pyplot.xlabel("")
        matplotlib.pyplot.title(f"Distance to {base_stage}")
        matplotlib.pyplot.tight_layout()

        figures.append(f"{base_stage}.pdf")
        fig.savefig(figures[-1])
        matplotlib.pyplot.close(fig)

    with tarfile.open(args.output, "w") as tar:
        for figure in tqdm.tqdm(figures):
            tar.add(figure, arcname=figure)
