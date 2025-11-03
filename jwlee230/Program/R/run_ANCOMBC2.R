rm(list = ls())

args = commandArgs(trailingOnly=TRUE)

main <- function(input_file, metadata_file, output_file)
{
    library(ANCOMBC)
    input_data <- data.frame(read.csv(input_file, sep="\t", row.names=1))
    print(head(input_data))

    metadata_content <- readLines(metadata_file)
    metadata <- read.csv(textConnection(metadata_content[-2]), header=TRUE, stringsAsFactors=FALSE, sep="\t", row.names=1)
    print(head(metadata))

    output_data <- ancombc2(data=input_data, meta_data=metadata, group="LongStage", fix_formula="LongStage")
    write.table(output_data$res, output_file, sep="\t")
}

if (length(args) == 3)
{
    main(args[1], args[2], args[3])
} else { print("Check arguments!!") }
