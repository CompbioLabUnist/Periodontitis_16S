"""
step24.py: Make 2D plot of Pg. Act.
"""
import argparse
import matplotlib
import matplotlib.pyplot
import seaborn
import step00

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", type=str, help="Input TAR.gz file")
    parser.add_argument("output", type=str, help="Output PNG file")

    args = parser.parse_args()

    if not args.output.endswith(".png"):
        raise ValueError("OUTPUT file must end with .PNG!!")

    matplotlib.use("Agg")
    matplotlib.rcParams.update({"font.size": 30})

    data = step00.read_pickle(args.input)
    act_column = "k__Bacteria; p__Actinobacteria; c__Actinobacteria; o__Actinomycetales; f__Actinomycetaceae; g__Actinomyces"
    pg_column = "k__Bacteria; p__Bacteroidetes; c__Bacteroidia; o__Bacteroidales; f__Porphyromonadaceae; g__Porphyromonas; s__gingivalis"
    data = data[[act_column, pg_column, "LongStage"]]

    print(data)

    seaborn.set(context="poster", style="whitegrid")
    fig, ax = matplotlib.pyplot.subplots(figsize=(24, 24))

    seaborn.scatterplot(data=data, x=act_column, y=pg_column, hue="LongStage", style="LongStage", ax=ax, legend="full", hue_order=step00.long_stage_order, style_order=step00.long_stage_order, palette=step00.color_stage_order)

    matplotlib.pyplot.xlabel("Actinomyces")
    matplotlib.pyplot.ylabel("Porphyromonas gingivalis")

    fig.savefig(args.output)
    matplotlib.pyplot.close(fig)
