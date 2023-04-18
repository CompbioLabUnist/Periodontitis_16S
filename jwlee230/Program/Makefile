# Variables
DATE := $(shell date "+%y%m%d")
RANDOM := $(shell bash -c 'echo $$RANDOM')
IMAGE_NAME = periodontitis:latest
CPUS = 20
MEMS = 100G
SHELL := /bin/bash
PWD := $(shell pwd)
TOOLS =
DOCKER = docker run --rm $(VOLUME_OPTS) $(RUN_OPTS) $(IMAGE_NAME)

# Options
VOLUME_OPTS = --volume $(abspath Output):/Output --volume $(abspath Data):/Data --volume $(abspath Python):/Python
RUN_OPTS = --tty --cpus="$(CPUS)" --memory="$(MEMS)"

# General
all:
.PHONY: all

log Output Docker/Tools:
	mkdir $@

# Docker
TOOLS += $(wildcard Docker/*)
build.log: Docker/Dockerfile $(TOOLS) | log Output
	rm -fv $@
	docker images | grep $(IMAGE_NAME) && docker rmi $(IMAGE_NAME) || true
	docker build --rm --tag $(IMAGE_NAME) $(<D) | tee $@

build: build.log
.PHONY: build

interactive: build.log
	docker run --rm $(VOLUME_OPTS) $(RUN_OPTS) --interactive $(IMAGE_NAME) /bin/bash || true
.PHONY: interactive

delete: build.log
	docker rmi $(IMAGE_NAME)
	rm -fv build.log
.PHONY: delete

stop:
	docker rm $(CONTAINER_NAME)

# SGE
tmp.sh: build.log
	echo "make -C $(PWD) latest" > tmp.sh

run: tmp.sh | log Output
	echo -e "#!$(SHELL)\nmake -C $(PWD) latest" > tmp.sh
	sbatch --chdir=$(abspath .) --cpus-per-task=$(CPUS) --error="$(abspath log)/%x.e%A" --job-name="Periodontitis_$(DATE)" --mail-type=ALL --mail-user="230@fumire.moe" --mem=$(MEMS) --output="$(abspath log)/%x.o%A" --export=ALL tmp.sh
.PHONY: run

# Actual
latest: step05 step15 step16 step17 step18 step19 step20 step22 step24 step26 step27 step28 step29 step30 step31
.PHONY: latest

# Step 01 (Make manifest files)
Output/Step01:
	mkdir -p $@

Output/Step01/whole.tsv: Python/step01.py $(wildcard Data/fastq/*.fastq.gz) | Output/Step01 build.log
	$(DOCKER) python3 $(addprefix /,$^) > $@

step01: Output/Step01/whole.tsv
.PHONY: step01

# Step 02 (De-multiplexing)
Output/Step02:
	mkdir -p $@

Output/Step02/%.qza: Output/Step01/%.tsv | Output/Step02 build.log
	$(DOCKER) qiime tools import --type "SampleData[PairedEndSequencesWithQuality]" --input-format "PairedEndFastqManifestPhred33V2" --input-path $(addprefix /,$<) --output-path $(addprefix /,$@) 1> $@.stdout 2> $@.stderr

Output/Step02/%.qzv: Output/Step02/%.qza | build.log
	$(DOCKER) qiime demux summarize --i-data $(addprefix /,$<) --o-visualization $(addprefix /,$@) 1> $@.stdout 2> $@.stderr

step02: Output/Step02/whole.qza Output/Step02/whole.qzv
.PHONY: step02

# Step 03 (Quality Filter)
Output/Step03:
	mkdir -p $@

Output/Step03/%.seq.qza Output/Step03/%.stat.qza: Output/Step02/%.qza | Output/Step03 build.log
	$(DOCKER) qiime quality-filter q-score --i-demux $(addprefix /,$<) --o-filtered-sequences $(addprefix /,$(basename $(basename $@))).seq.qza --o-filter-stats $(addprefix /,$(basename $(basename $@))).stat.qza 1> $@.stdout 2> $@.stderr

step03: Output/Step03/whole.seq.qza
.PHONY: step03

# Step 04 (DADA2 Denoising)
Output/Step04:
	mkdir -p $@

Output/Step04/%.table.qza Output/Step04/%.seq.qza Output/Step04/%.stat.qza: Output/Step02/%.qza | Output/Step04 build.log
	@echo "You have to check the length manually!!"
	$(DOCKER) qiime dada2 denoise-paired --i-demultiplexed-seqs $(addprefix /,$<) --p-n-threads $(CPUS) --p-trunc-len-f 300 --p-trunc-len-r 265 --o-table $(addprefix /,$(basename $(basename $@))).table.qza --o-representative-sequences $(addprefix /,$(basename $(basename $@))).seq.qza --o-denoising-stats $(addprefix /,$(basename $(basename $@))).stat.qza 1> $@.stdout 2> $@.stderr

step04: Output/Step04/whole.seq.qza
.PHONY: step04

# Step 05 (Deblur Denoising)
Output/Step05:
	mkdir -p $@

Output/Step05/%.table.qza Output/Step05/%.seq.qza Output/Step05/%.stat.qza: Output/Step03/%.seq.qza | Output/Step05 build.log
	@echo "You have to check the length manually!!"
	$(DOCKER) qiime deblur denoise-16S --i-demultiplexed-seqs $(addprefix /,$<) --p-trim-length 265 --p-jobs-to-start $(CPUS) --o-table $(addprefix /,$(basename $(basename $@))).table.qza --o-representative-sequences $(addprefix /,$(basename $(basename $@))).seq.qza --o-stats $(addprefix /,$(basename $(basename $@))).stat.qza 1> $@.stdout 2> $@.stderr

step05: Output/Step05/whole.table.qza
.PHONY: step05

# Step 06 (Visulaize Denoising)
Output/Step06:
	mkdir -p $@

Output/Step06/%.Demux.stat.qzv: Output/Step03/%.stat.qza | Output/Step06 build.log
	$(DOCKER) qiime metadata tabulate --m-input-file $(addprefix /,$<) --o-visualization $(addprefix /,$@) 1> $@.stdout 2> $@.stderr

Output/Step06/%.Deblur.stat.qzv: Output/Step05/%.stat.qza | Output/Step06 build.log
	$(DOCKER) qiime deblur visualize-stats --i-deblur-stats $(addprefix /,$<) --o-visualization $(addprefix /,$@) 1> $@.stdout 2> $@.stderr

step06: Output/Step06/whole.Demux.stat.qzv Output/Step06/whole.Deblur.stat.qzv
.PHONY: step06

# Step 07 (Make metadata)
Output/Step07:
	mkdir -p $@

Output/Step07/whole.tsv: Python/step07.py Data/치주염선정샘플_250샘플.xlsx $(wildcard Data/fastq/*.fastq.gz) | Output/Step07 build.log
	@$(DOCKER) python3 $(addprefix /,$^) > $@

step07: Output/Step07/whole.tsv
.PHONY: step07

# Step 08 (Summary FeatureTable and FeatureData)
Output/Step08:
	mkdir -p $@

Output/Step08/%.DADA.table.qzv: Output/Step04/%.table.qza Output/Step07/%.tsv | Output/Step08 build.log
	$(DOCKER) qiime feature-table summarize --i-table $(addprefix /,$(word 1,$^)) --m-sample-metadata-file $(addprefix /,$(word 2,$^)) --o-visualization $(addprefix /,$@) 1> $@.stdout 2> $@.stderr

Output/Step08/%.DADA.seq.qzv: Output/Step04/%.seq.qza | Output/Step08 build.log
	$(DOCKER) qiime feature-table tabulate-seqs --i-data $(addprefix /,$<) --o-visualization $(addprefix /,$@) 1> $@.stdout 2> $@.stderr

Output/Step08/%.Deblur.table.qzv: Output/Step05/%.table.qza Output/Step07/%.tsv | Output/Step08 build.log
	$(DOCKER) qiime feature-table summarize --i-table $(addprefix /,$(word 1,$^)) --m-sample-metadata-file $(addprefix /,$(word 2,$^)) --o-visualization $(addprefix /,$@) 1> $@.stdout 2> $@.stderr

Output/Step08/%.Deblur.seq.qzv: Output/Step05/%.seq.qza | Output/Step08 build.log
	$(DOCKER) qiime feature-table tabulate-seqs --i-data $(addprefix /,$<) --o-visualization $(addprefix /,$@) 1> $@.stdout 2> $@.stderr

step08: Output/Step08/whole.DADA.table.qzv Output/Step08/whole.DADA.seq.qzv
.PHONY: step08

# Step 09 (Phylogenetic diversity analysis)
Output/Step09:
	mkdir -p $@

Output/Step09/%.DADA.aligned-rep-seqs.qza Output/Step09/%.DADA.masked-aligned-rep-seqs.qza Output/Step09/%.DADA.unrooted-tree.qza Output/Step09/%.DADA.rooted-tree.qza: Output/Step04/%.seq.qza | Output/Step09 build.log
	$(DOCKER) qiime phylogeny align-to-tree-mafft-fasttree --i-sequences $(addprefix /,$<) --o-alignment $(addprefix /,$(basename $(basename $@))).aligned-rep-seqs.qza --o-masked-alignment $(addprefix /,$(basename $(basename $@))).masked-aligned-rep-seqs.qza --o-tree $(addprefix /,$(basename $(basename $@))).unrooted-tree.qza --o-rooted-tree $(addprefix /,$(basename $(basename $@))).rooted-tree.qza --p-n-threads $(CPUS) 1> $@.stdout 2> $@.stderr

Output/Step09/%.Deblur.aligned-rep-seqs.qza Output/Step09/%.Deblur.masked-aligned-rep-seqs.qza Output/Step09/%.Deblur.unrooted-tree.qza Output/Step09/%.Deblur.rooted-tree.qza: Output/Step05/%.seq.qza | Output/Step09 build.log
	$(DOCKER) qiime phylogeny align-to-tree-mafft-fasttree --i-sequences $(addprefix /,$<) --o-alignment $(addprefix /,$(basename $(basename $@))).aligned-rep-seqs.qza --o-masked-alignment $(addprefix /,$(basename $(basename $@))).masked-aligned-rep-seqs.qza --o-tree $(addprefix /,$(basename $(basename $@))).unrooted-tree.qza --o-rooted-tree $(addprefix /,$(basename $(basename $@))).rooted-tree.qza --p-n-threads $(CPUS) 1> $@.stdout 2> $@.stderr

step09: Output/Step09/whole.DADA.rooted-tree.qza
.PHONY: step09

# Step 10 (Core metrics phylogenetic)
Output/Step10/DADA2 Output/Step10/Deblur:
	mkdir -p $@

Output/Step10/DADA2/%.faith_pd_vector.qza Output/Step10/DADA2/%.unweighted_unifrac_distance_matrix.qza Output/Step10/DADA2/%.bray_curtis_pcoa_results.qza Output/Step10/DADA2/%.shannon_vector.qza Output/Step10/DADA2/%.rarefied_table.qza Output/Step10/DADA2/%.weighted_unifrac_distance_matrix.qza Output/Step10/DADA2/%.jaccard_pcoa_results.qza Output/Step10/DADA2/%.weighted_unifrac_pcoa_results.qza Output/Step10/DADA2/%.observed_features_vector.qza Output/Step10/DADA2/%.jaccard_distance_matrix.qza Output/Step10/DADA2/%.evenness_vector.qza Output/Step10/DADA2/%.bray_curtis_distance_matrix.qza Output/Step10/DADA2/%.unweighted_unifrac_pcoa_results.qza Output/Step10/DADA2/%.unweighted_unifrac_emperor.qzv Output/Step10/DADA2/%.jaccard_emperor.qzv Output/Step10/DADA2/%.bray_curtis_emperor.qzv Output/Step10/DADA2/%.weighted_unifrac_emperor.qzv: Output/Step09/%.DADA.rooted-tree.qza Output/Step04/%.table.qza Output/Step07/%.tsv | Output/Step10/DADA2 build.log
	@echo "You have to check the sequencing depth manually!!"
	$(DOCKER) qiime diversity core-metrics-phylogenetic --i-phylogeny $(addprefix /,$(word 1,$^)) --i-table $(addprefix /,$(word 2,$^)) --m-metadata-file $(addprefix /,$(word 3,$^)) --p-sampling-depth 3786 --o-rarefied-table $(addprefix /,$(basename $(basename $@))).rarefied_table.qza --o-faith-pd-vector $(addprefix /,$(basename $(basename $@))).faith_pd_vector.qza --o-observed-features-vector $(addprefix /,$(basename $(basename $@))).observed_features_vector.qza --o-shannon-vector $(addprefix /,$(basename $(basename $@))).shannon_vector.qza --o-evenness-vector $(addprefix /,$(basename $(basename $@))).evenness_vector.qza --o-unweighted-unifrac-distance-matrix $(addprefix /,$(basename $(basename $@))).unweighted_unifrac_distance_matrix.qza --o-weighted-unifrac-distance-matrix $(addprefix /,$(basename $(basename $@))).weighted_unifrac_distance_matrix.qza --o-jaccard-distance-matrix $(addprefix /,$(basename $(basename $@))).jaccard_distance_matrix.qza --o-bray-curtis-distance-matrix $(addprefix /,$(basename $(basename $@))).bray_curtis_distance_matrix.qza --o-unweighted-unifrac-pcoa-results $(addprefix /,$(basename $(basename $@))).unweighted_unifrac_pcoa_results.qza --o-weighted-unifrac-pcoa-results $(addprefix /,$(basename $(basename $@))).weighted_unifrac_pcoa_results.qza --o-unweighted-unifrac-emperor $(addprefix /,$(basename $(basename $@))).unweighted_unifrac_emperor.qzv --o-weighted-unifrac-emperor $(addprefix /,$(basename $(basename $@))).weighted_unifrac_emperor.qzv --o-jaccard-emperor $(addprefix /,$(basename $(basename $@))).jaccard_emperor.qzv --o-bray-curtis-emperor $(addprefix /,$(basename $(basename $@))).bray_curtis_emperor.qzv --o-jaccard-pcoa-results $(addprefix /,$(basename $(basename $@))).jaccard_pcoa_results.qza --o-bray-curtis-pcoa-results $(addprefix /,$(basename $(basename $@))).bray_curtis_pcoa_results.qza --p-n-jobs-or-threads $(CPUS) 1> $@.stdout 2> $@.stderr

Output/Step10/Deblur/%.faith_pd_vector.qza Output/Step10/Deblur/%.unweighted_unifrac_distance_matrix.qza Output/Step10/Deblur/%.bray_curtis_pcoa_results.qza Output/Step10/Deblur/%.shannon_vector.qza Output/Step10/Deblur/%.rarefied_table.qza Output/Step10/Deblur/%.weighted_unifrac_distance_matrix.qza Output/Step10/Deblur/%.jaccard_pcoa_results.qza Output/Step10/Deblur/%.weighted_unifrac_pcoa_results.qza Output/Step10/Deblur/%.observed_features_vector.qza Output/Step10/Deblur/%.jaccard_distance_matrix.qza Output/Step10/Deblur/%.evenness_vector.qza Output/Step10/Deblur/%.bray_curtis_distance_matrix.qza Output/Step10/Deblur/%.unweighted_unifrac_pcoa_results.qza Output/Step10/Deblur/%.unweighted_unifrac_emperor.qzv Output/Step10/DADA2/%.jaccard_emperor.qzv Output/Step10/Deblur/%.bray_curtis_emperor.qzv Output/Step10/Deblur/%.weighted_unifrac_emperor.qzv: Output/Step09/%.Deblur.rooted-tree.qza Output/Step05/%.table.qza Output/Step07/%.tsv | Output/Step10/Deblur build.log
	@echo "You have to check the sequencing depth manually!!"
	$(DOCKER) qiime diversity core-metrics-phylogenetic --i-phylogeny $(addprefix /,$(word 1,$^)) --i-table $(addprefix /,$(word 2,$^)) --m-metadata-file $(addprefix /,$(word 3,$^)) --p-sampling-depth 7253 --o-rarefied-table $(addprefix /,$(basename $(basename $@))).rarefied_table.qza --o-faith-pd-vector $(addprefix /,$(basename $(basename $@))).faith_pd_vector.qza --o-observed-features-vector $(addprefix /,$(basename $(basename $@))).observed_features_vector.qza --o-shannon-vector $(addprefix /,$(basename $(basename $@))).shannon_vector.qza --o-evenness-vector $(addprefix /,$(basename $(basename $@))).evenness_vector.qza --o-unweighted-unifrac-distance-matrix $(addprefix /,$(basename $(basename $@))).unweighted_unifrac_distance_matrix.qza --o-weighted-unifrac-distance-matrix $(addprefix /,$(basename $(basename $@))).weighted_unifrac_distance_matrix.qza --o-jaccard-distance-matrix $(addprefix /,$(basename $(basename $@))).jaccard_distance_matrix.qza --o-bray-curtis-distance-matrix $(addprefix /,$(basename $(basename $@))).bray_curtis_distance_matrix.qza --o-unweighted-unifrac-pcoa-results $(addprefix /,$(basename $(basename $@))).unweighted_unifrac_pcoa_results.qza --o-weighted-unifrac-pcoa-results $(addprefix /,$(basename $(basename $@))).weighted_unifrac_pcoa_results.qza --o-unweighted-unifrac-emperor $(addprefix /,$(basename $(basename $@))).unweighted_unifrac_emperor.qzv --o-weighted-unifrac-emperor $(addprefix /,$(basename $(basename $@))).weighted_unifrac_emperor.qzv --o-jaccard-emperor $(addprefix /,$(basename $(basename $@))).jaccard_emperor.qzv --o-bray-curtis-emperor $(addprefix /,$(basename $(basename $@))).bray_curtis_emperor.qzv --o-jaccard-pcoa-results $(addprefix /,$(basename $(basename $@))).jaccard_pcoa_results.qza --o-bray-curtis-pcoa-results $(addprefix /,$(basename $(basename $@))).bray_curtis_pcoa_results.qza --p-n-jobs-or-threads $(CPUS) 1> $@.stdout 2> $@.stderr

step10: Output/Step10/DADA2/whole.faith_pd_vector.qza
.PHONY: step10

# Step 11 (Alpha-diversity)
Output/Step11/DADA2 Output/Step11/Deblur:
	mkdir -p $@

Output/Step11/DADA2/%.qzv: Output/Step10/DADA2/%.qza Output/Step07/whole.tsv | Output/Step11/DADA2 build.log
	$(DOCKER) qiime diversity alpha-group-significance --i-alpha-diversity $(addprefix /,$(word 1,$^)) --m-metadata-file $(addprefix /,$(word 2,$^)) --o-visualization $(addprefix /,$@) 1> $@.stdout 2> $@.stderr

Output/Step11/Deblur/%.qzv: Output/Step10/Deblur/%.qza Output/Step07/whole.tsv | Output/Step11/Deblur build.log
	$(DOCKER) qiime diversity alpha-group-significance --i-alpha-diversity $(addprefix /,$(word 1,$^)) --m-metadata-file $(addprefix /,$(word 2,$^)) --o-visualization $(addprefix /,$@) 1> $@.stdout 2> $@.stderr

step11: Output/Step11/DADA2/whole.faith_pd_vector.qzv Output/Step11/DADA2/whole.observed_features_vector.qzv Output/Step11/DADA2/whole.shannon_vector.qzv Output/Step11/DADA2/whole.evenness_vector.qzv
.PHONY: step11

# Step 12 (Beta-diversity)
Output/Step12/DADA2 Output/Step12/Deblur:
	mkdir -p $@

Output/Step12/DADA2/%.qzv: Output/Step10/DADA2/%.qza Output/Step07/whole.tsv | Output/Step12/DADA2 build.log
	$(DOCKER) qiime diversity beta-group-significance --i-distance-matrix $(addprefix /,$(word 1,$^)) --m-metadata-file $(addprefix /,$(word 2,$^)) --m-metadata-column "LongStage" --o-visualization $(addprefix /,$@) --p-pairwise 1> $@.stdout 2> $@.stderr

Output/Step12/Deblur/%.qzv: Output/Step10/Deblur/%.qza Output/Step07/whole.tsv | Output/Step12/Deblur build.log
	$(DOCKER) qiime diversity beta-group-significance --i-distance-matrix $(addprefix /,$(word 1,$^)) --m-metadata-file $(addprefix /,$(word 2,$^)) --m-metadata-column "LongStage" --o-visualization $(addprefix /,$@) --p-pairwise 1> $@.stdout 2> $@.stderr

step12: Output/Step12/DADA2/whole.unweighted_unifrac_distance_matrix.qzv Output/Step12/DADA2/whole.weighted_unifrac_distance_matrix.qzv Output/Step12/DADA2/whole.jaccard_distance_matrix.qzv Output/Step12/DADA2/whole.bray_curtis_distance_matrix.qzv Output/Step12/Deblur/whole.unweighted_unifrac_distance_matrix.qzv Output/Step12/Deblur/whole.weighted_unifrac_distance_matrix.qzv Output/Step12/Deblur/whole.jaccard_distance_matrix.qzv Output/Step12/Deblur/whole.bray_curtis_distance_matrix.qzv
.PHONY: step12

# Step 13 (Taxonomy Classification)
Output/Step13:
	mkdir -p $@

Output/Step13/silva.qza: | Output/Step13 build.log
	wget "https://data.qiime2.org/2020.8/common/silva-138-99-nb-classifier.qza" -O $@

Output/Step13/gg.qza: | Output/Step13 build.log
	wget "https://data.qiime2.org/2020.8/common/gg-13-8-99-nb-classifier.qza" -O $@

Output/Step13/%.DADA2.silva.qza: Output/Step13/silva.qza Output/Step04/%.seq.qza | build.log
	$(DOCKER) qiime feature-classifier classify-sklearn --i-classifier $(addprefix /,$(word 1,$^)) --i-reads $(addprefix /,$(word 2,$^)) --p-n-jobs $(CPUS) --o-classification $(addprefix /,$@) 1> $@.stdout 2> $@.stderr

Output/Step13/%.DADA2.gg.qza: Output/Step13/gg.qza Output/Step04/%.seq.qza | build.log
	$(DOCKER) qiime feature-classifier classify-sklearn --i-classifier $(addprefix /,$(word 1,$^)) --i-reads $(addprefix /,$(word 2,$^)) --p-n-jobs $(CPUS) --o-classification $(addprefix /,$@) 1> $@.stdout 2> $@.stderr

Output/Step13/%.Deblur.silva.qza: Output/Step13/silva.qza Output/Step05/%.seq.qza | build.log
	$(DOCKER) qiime feature-classifier classify-sklearn --i-classifier $(addprefix /,$(word 1,$^)) --i-reads $(addprefix /,$(word 2,$^)) --p-n-jobs $(CPUS) --o-classification $(addprefix /,$@) 1> $@.stdout 2> $@.stderr

Output/Step13/%.Deblur.gg.qza: Output/Step13/gg.qza Output/Step05/%.seq.qza | build.log
	$(DOCKER) qiime feature-classifier classify-sklearn --i-classifier $(addprefix /,$(word 1,$^)) --i-reads $(addprefix /,$(word 2,$^)) --p-n-jobs $(CPUS) --o-classification $(addprefix /,$@) 1> $@.stdout 2> $@.stderr

Output/Step13/homd.fasta: | Output/Step13 build.log
	wget "http://www.homd.org/ftp/16S_rRNA_refseq/HOMD_16S_rRNA_RefSeq/V15.21/HOMD_16S_rRNA_RefSeq_V15.21.fasta" -O $@

Output/Step13/homd.txt: | Output/Step13 build.log
	wget "http://www.homd.org/ftp/16S_rRNA_refseq/HOMD_16S_rRNA_RefSeq/V15.21/HOMD_16S_rRNA_RefSeq_V15.21.qiime.taxonomy" -O $@

Output/Step13/homd_fasta.qza: Output/Step13/homd.fasta | build.log
	$(DOCKER) qiime tools import --type "FeatureData[Sequence]" --input-path $(addprefix /,$<) --output-path $(addprefix /,$@) 1> $@.stdout 2> $@.stderr

Output/Step13/homd_taxo.qza: Output/Step13/homd.txt | build.log
	$(DOCKER) qiime tools import --type "FeatureData[Taxonomy]" --input-format "HeaderlessTSVTaxonomyFormat" --input-path $(addprefix /,$<) --output-path $(addprefix /,$@) 1> $@.stdout 2> $@.stderr

Output/Step13/homd_seqs.qza: Output/Step13/homd_fasta.qza | build.log
	$(DOCKER) qiime feature-classifier extract-reads --i-sequences $(addprefix /,$<) --p-f-primer "GTGCCAGCMGCCGCGGTAA" --p-r-primer "GGACTACHVGGGTWTCTAAT" --o-reads $(addprefix /,$@) --p-n-jobs $(CPUS) 1> $@.stdout 2> $@.stderr

Output/Step13/homd.qza: Output/Step13/homd_seqs.qza Output/Step13/homd_taxo.qza | build.log
	$(DOCKER) qiime feature-classifier fit-classifier-naive-bayes --i-reference-reads $(addprefix /,$(word 1,$^)) --i-reference-taxonomy $(addprefix /,$(word 2,$^)) --o-classifier $(addprefix /,$@) 1> $@.stdout 2> $@.stderr

Output/Step13/%.DADA2.homd.qza: Output/Step13/homd.qza Output/Step04/%.seq.qza | build.log
	$(DOCKER) qiime feature-classifier classify-sklearn --i-classifier $(addprefix /,$(word 1,$^)) --i-reads $(addprefix /,$(word 2,$^)) --p-n-jobs $(CPUS) --o-classification $(addprefix /,$@) 1> $@.stdout 2> $@.stderr

Output/Step13/%.Deblur.homd.qza: Output/Step13/homd.qza Output/Step05/%.seq.qza | build.log
	$(DOCKER) qiime feature-classifier classify-sklearn --i-classifier $(addprefix /,$(word 1,$^)) --i-reads $(addprefix /,$(word 2,$^)) --p-n-jobs $(CPUS) --o-classification $(addprefix /,$@) 1> $@.stdout 2> $@.stderr

step13: Output/Step13/homd_fasta.qza Output/Step13/homd_seqs.qza Output/Step13/homd.qza Output/Step13/whole.DADA2.homd.qza
.PHONY: step13

# Step 14 (Run ANCOM)
Output/Step14:
	mkdir -p $@

Output/Step14/whole.DADA2.%.qza: Output/Step04/whole.table.qza Output/Step13/whole.DADA2.%.qza | Output/Step14 build.log
	$(DOCKER) qiime taxa collapse --i-table $(addprefix /,$(word 1,$^)) --i-taxonomy $(addprefix /,$(word 2,$^)) --p-level 7 --o-collapsed-table $(addprefix /,$@) 1> $@.stdout 2> $@.stderr

Output/Step14/whole.Deblur.%.qza: Output/Step05/whole.table.qza Output/Step13/whole.Deblur.%.qza | Output/Step14 build.log
	$(DOCKER) qiime taxa collapse --i-table $(addprefix /,$(word 1,$^)) --i-taxonomy $(addprefix /,$(word 2,$^)) --p-level 7 --o-collapsed-table $(addprefix /,$@) 1> $@.stdout 2> $@.stderr

Output/Step14/%.pseudocount.qza: Output/Step14/%.qza | build.log
	$(DOCKER) qiime composition add-pseudocount --i-table $(addprefix /,$(word 1,$^)) --o-composition-table $(addprefix /,$@) 1> $@.stdout 2> $@.stderr

Output/Step14/%.site.qzv: Output/Step14/%.qza Output/Step07/whole.tsv | build.log
	$(DOCKER) qiime composition ancom --i-table $(addprefix /,$(word 1,$^)) --m-metadata-file $(addprefix /,$(word 2,$^)) --m-metadata-column "LongStage" --o-visualization $(addprefix /,$@) 1> $@.stdout 2> $@.stderr

Output/Step14/%/ancom.tsv: Output/Step14/%.qzv | build.log
	$(DOCKER) rm -rfv $(addprefix /,$(@D))
	$(DOCKER) qiime tools export --input-path $(addprefix /,$(word 1,$^)) --output-path $(addprefix /,$(@D)) 1> $(@D).stdout 2> $(@D).stderr

step14: Output/Step14/whole.DADA2.homd.qza Output/Step14/whole.DADA2.homd.pseudocount.qza Output/Step14/whole.DADA2.homd.pseudocount.site.qzv Output/Step14/whole.DADA2.homd.pseudocount.site/ancom.tsv
.PHONY: step14

# Step 15 (Get TSV files)
Output/Step15:
	mkdir -p $@

Output/Step15/whole.DADA2.%.tsv: Output/Step04/whole.table.qza Output/Step13/whole.DADA2.%.qza | Output/Step15 build.log
	$(DOCKER) qiime tools export --input-path $(addprefix /,$(word 1,$^)) --output-path /Output/Step15/tmp
	$(DOCKER) qiime tools export --input-path $(addprefix /,$(word 2,$^)) --output-path /Output/Step15/tmp
	$(DOCKER) sed --in-place "1c#OTU ID\ttaxonomy\tconfidence" /Output/Step15/tmp/taxonomy.tsv
	$(DOCKER) biom add-metadata --input-fp /Output/Step15/tmp/feature-table.biom --observation-metadata-fp /Output/Step15/tmp/taxonomy.tsv --output-fp /Output/Step15/tmp/tmp.biom --sc-separated "taxonomy"
	$(DOCKER) biom convert --input-fp /Output/Step15/tmp/tmp.biom --output-fp $(addprefix /,$@) --to-tsv --process-obs-metadata "taxonomy" --tsv-metadata-formatter "sc_separated" --header-key "taxonomy"
	$(DOCKER) rm -rfv /Output/Step15/tmp

Output/Step15/whole.Deblur.%.tsv: Output/Step05/whole.table.qza Output/Step13/whole.Deblur.%.qza | Output/Step15 build.log
	$(DOCKER) qiime tools export --input-path $(addprefix /,$(word 1,$^)) --output-path /Output/Step15/tmp
	$(DOCKER) qiime tools export --input-path $(addprefix /,$(word 2,$^)) --output-path /Output/Step15/tmp
	$(DOCKER) sed --in-place "1c#OTU ID\ttaxonomy\tconfidence" /Output/Step15/tmp/taxonomy.tsv
	$(DOCKER) biom add-metadata --input-fp /Output/Step15/tmp/feature-table.biom --observation-metadata-fp /Output/Step15/tmp/taxonomy.tsv --output-fp /Output/Step15/tmp/tmp.biom --sc-separated "taxonomy"
	$(DOCKER) biom convert --input-fp /Output/Step15/tmp/tmp.biom --output-fp $(addprefix /,$@) --to-tsv --process-obs-metadata "taxonomy" --tsv-metadata-formatter "sc_separated" --header-key "taxonomy"
	$(DOCKER) rm -rfv /Output/Step15/tmp

Output/Step15/%.Proportion.tsv: Python/step15.py Output/Step15/%.tsv | build.log
	$(DOCKER) python3 $(addprefix /,$^ $@) 1> $@.stdout 2> $@.stderr

step15: Output/Step15/whole.DADA2.homd.tsv Output/Step15/whole.DADA2.homd.Proportion.tsv
.PHONY: step15

# Step 16 (Clearify raw TSV into pandas)
Output/Step16:
	mkdir -p $@

Output/Step16/%.tar.gz: Python/step16.py Output/Step15/%.tsv | Output/Step16 build.log
	$(DOCKER) python3 $(addprefix /,$^ $@) 1> $@.stdout 2> $@.stderr

step16: Output/Step16/whole.DADA2.homd.tar.gz Output/Step16/whole.DADA2.homd.Proportion.tar.gz
.PHONY: step16

# Step 17 (Select by ANCOM)
Output/Step17:
	mkdir -p $@

Output/Step17/%.tar.gz: Python/step17.py Output/Step16/%.tar.gz Output/Step14/%.pseudocount.site/ancom.tsv | Output/Step17 build.log
	$(DOCKER) python3 $(addprefix /,$^ $@) 1> $@.stdout 2> $@.stderr

Output/Step17/%.Proportion.tar.gz: Python/step17.py Output/Step16/%.Proportion.tar.gz Output/Step14/%.pseudocount.site/ancom.tsv | Output/Step17 build.log
	$(DOCKER) python3 $(addprefix /,$^ $@) 1> $@.stdout 2> $@.stderr

step17: Output/Step17/whole.DADA2.homd.tar.gz Output/Step17/whole.DADA2.homd.Proportion.tar.gz
.PHONY: step17

# Step 18 (Make t-SNE)
Output/Step18:
	mkdir -p $@

Output/Step18/whole.%.tar.gz: Python/step18.py Output/Step16/whole.%.tar.gz | Output/Step18 build.log
	$(DOCKER) python3 $(addprefix /,$^ $@) --cpu $(CPUS) 1> $@.stdout 2> $@.stderr

Output/Step18/ANCOM.%.tar.gz: Python/step18.py Output/Step17/whole.%.tar.gz | Output/Step18 build.log
	$(DOCKER) python3 $(addprefix /,$^ $@) --cpu $(CPUS) 1> $@.stdout 2> $@.stderr

step18: Output/Step18/whole.DADA2.homd.tar.gz Output/Step18/ANCOM.DADA2.homd.tar.gz Output/Step18/whole.DADA2.homd.Proportion.tar.gz Output/Step18/ANCOM.DADA2.homd.Proportion.tar.gz
.PHONY: step18

# Step 19 (Draw default t-SNE)
Output/Step19:
	mkdir -p $@

Output/Step19/%.png: Python/step19.py Output/Step18/%.tar.gz | Output/Step19 build.log
	$(DOCKER) python3 $(addprefix /,$^ $@) 1> $@.stdout 2> $@.stderr

step19: Output/Step19/whole.DADA2.homd.png Output/Step19/ANCOM.DADA2.homd.png Output/Step19/whole.DADA2.homd.Proportion.png Output/Step19/ANCOM.DADA2.homd.Proportion.png
.PHONY: step19

# Step 20 (Random Forest Classifier)
Output/Step20:
	mkdir -p $@

Output/Step20/%.tar: Python/step20-1.py Output/Step16/whole.%.tar.gz Output/Step18/whole.%.tar.gz | Output/Step20 build.log
	$(DOCKER) python3 $(addprefix /,$^ $@) --cpu $(CPUS) 1> $@.stdout 2> $@.stderr

Output/Step20/one.%.tar: Python/step20-1.py Output/Step16/whole.%.tar.gz Output/Step18/whole.%.tar.gz | Output/Step20 build.log
	$(DOCKER) python3 $(addprefix /,$^ $@) --cpu $(CPUS) --one 1> $@.stdout 2> $@.stderr

Output/Step20/two.%.tar: Python/step20-1.py Output/Step16/whole.%.tar.gz Output/Step18/whole.%.tar.gz | Output/Step20 build.log
	$(DOCKER) python3 $(addprefix /,$^ $@) --cpu $(CPUS) --two 1> $@.stdout 2> $@.stderr

Output/Step20/three.%.tar: Python/step20-1.py Output/Step16/whole.%.tar.gz Output/Step18/whole.%.tar.gz | Output/Step20 build.log
	$(DOCKER) python3 $(addprefix /,$^ $@) --cpu $(CPUS) --three 1> $@.stdout 2> $@.stderr

Output/Step20/ANCOM.%.tar: Python/step20.py Output/Step17/whole.%.tar.gz Output/Step18/ANCOM.%.tar.gz | Output/Step20 build.log
	$(DOCKER) python3 $(addprefix /,$^ $@) --cpu $(CPUS) 1> $@.stdout 2> $@.stderr

Output/Step20/ANCOM.one.%.tar: Python/step20.py Output/Step17/whole.%.tar.gz Output/Step18/ANCOM.%.tar.gz | Output/Step20 build.log
	$(DOCKER) python3 $(addprefix /,$^ $@) --cpu $(CPUS) --one 1> $@.stdout 2> $@.stderr

Output/Step20/ANCOM.two.%.tar: Python/step20.py Output/Step17/whole.%.tar.gz Output/Step18/ANCOM.%.tar.gz | Output/Step20 build.log
	$(DOCKER) python3 $(addprefix /,$^ $@) --cpu $(CPUS) --two 1> $@.stdout 2> $@.stderr

Output/Step20/ANCOM.three.%.tar: Python/step20.py Output/Step17/whole.%.tar.gz Output/Step18/ANCOM.%.tar.gz | Output/Step20 build.log
	$(DOCKER) python3 $(addprefix /,$^ $@) --cpu $(CPUS) --three 1> $@.stdout 2> $@.stderr

step20: Output/Step20/DADA2.homd.Proportion.tar Output/Step20/one.DADA2.homd.Proportion.tar Output/Step20/two.DADA2.homd.Proportion.tar Output/Step20/three.DADA2.homd.Proportion.tar Output/Step20/ANCOM.DADA2.homd.Proportion.tar Output/Step20/ANCOM.one.DADA2.homd.Proportion.tar Output/Step20/ANCOM.two.DADA2.homd.Proportion.tar Output/Step20/ANCOM.three.DADA2.homd.Proportion.tar
.PHONY: step20

# Step 21 (Extract TSV from Beta-diversity)
Output/Step21:
	mkdir -p $@

Output/Step21/DADA2.%.tsv: Output/Step10/DADA2/whole.%_distance_matrix.qza | Output/Step21 build.log
	$(DOCKER) qiime tools export --input-path $(addprefix /,$(word 1,$^)) --output-path $(addprefix /,$(@D)) 1> $@.stdout 2> $@.stderr
	$(DOCKER) mv $(addprefix /,$(@D))/distance-matrix.tsv $(addprefix /,$@)

Output/Step21/Deblur.%.tsv: Output/Step10/Deblur/whole.%_distance_matrix.qza | Output/Step21 build.log
	$(DOCKER) qiime tools export --input-path $(addprefix /,$(word 1,$^)) --output-path $(addprefix /,$(@D)) 1> $@.stdout 2> $@.stderr
	$(DOCKER) mv $(addprefix /,$(@D))/distance-matrix.tsv $(addprefix /,$@)

step21: Output/Step21/DADA2.unweighted_unifrac.tsv Output/Step21/DADA2.weighted_unifrac.tsv Output/Step21/DADA2.jaccard.tsv Output/Step21/DADA2.bray_curtis.tsv Output/Step21/Deblur.unweighted_unifrac.tsv Output/Step21/Deblur.weighted_unifrac.tsv Output/Step21/Deblur.jaccard.tsv Output/Step21/Deblur.bray_curtis.tsv
.PHONY: step21

# Step 22 (Draw t-SNE from Beta-diversity TSV)
Output/Step22:
	mkdir -p $@

Output/Step22/%.png: Python/step22.py Output/Step21/%.tsv | Output/Step22 build.log
	$(DOCKER) python3 $(addprefix /,$^ $@) --cpu $(CPUS) 1> $@.stdout 2> $@.stderr

step22: Output/Step22/DADA2.unweighted_unifrac.png Output/Step22/DADA2.weighted_unifrac.png Output/Step22/DADA2.jaccard.png Output/Step22/DADA2.bray_curtis.png Output/Step22/Deblur.unweighted_unifrac.png Output/Step22/Deblur.weighted_unifrac.png Output/Step22/Deblur.jaccard.png Output/Step22/Deblur.bray_curtis.png
.PHONY: step22

# Step 23 (Divide Healty vs. Early Only)
Output/Step23:
	mkdir -p $@

Output/Step23/HE.%.tar: Python/step23.py Output/Step17/whole.%.tar.gz Output/Step18/ANCOM.%.tar.gz | Output/Step23 build.log
	$(DOCKER) python3 $(addprefix /,$^ $@) --cpu $(CPUS) 1> $@.stdout 2> $@.stderr

step23: Output/Step23/HE.DADA2.homd.tar
.PHONY: step23

# Step 24 (Make 2D plot of Pg. & Act.)
Output/Step24:
	mkdir -p $@

Output/Step24/2D.%.png: Python/step24.py Output/Step17/whole.%.tar.gz | Output/Step24 build.log
	$(DOCKER) python3 $(addprefix /,$^ $@) 1> $@.stdout 2> $@.sterr

step24: Output/Step24/2D.DADA2.homd.png Output/Step24/2D.DADA2.homd.Proportion.png
.PHONY: step24

# Step 25 (Extract TSV from alpha-diversity)
Output/Step25:
	mkdir -p $@

Output/Step25/DADA2.%.tsv: Output/Step10/DADA2/whole.%_vector.qza | Output/Step25 build.log
	$(DOCKER) qiime tools export --input-path $(addprefix /,$(word 1,$^)) --output-path $(addprefix /,$(@D)) 1> $@.stdout 2> $@.stderr
	$(DOCKER) mv $(addprefix /,$(@D))/alpha-diversity.tsv $(addprefix /,$@)

Output/Step25/Deblur.%.tsv: Output/Step10/Deblur/whole.%_vector.qza | Output/Step25 build.log
	$(DOCKER) qiime tools export --input-path $(addprefix /,$(word 1,$^)) --output-path $(addprefix /,$(@D)) 1> $@.stdout 2> $@.stderr
	$(DOCKER) mv $(addprefix /,$(@D))/alpha-diversity.tsv $(addprefix /,$@)

step25: Output/Step25/DADA2.evenness.tsv Output/Step25/DADA2.faith_pd.tsv Output/Step25/DADA2.observed_features.tsv Output/Step25/DADA2.shannon.tsv
.PHONY: step25

# Step 26 (Draw violin plot by alpha-diversity)
Output/Step26:
	mkdir -p $@

Output/Step26/%.png: Python/step26.py Output/Step25/%.tsv | Output/Step26 build.log
	$(DOCKER) python3 $(addprefix /,$^ $@) 1> $@.stdout 2> $@.stderr

step26: Output/Step26/DADA2.evenness.png Output/Step26/DADA2.faith_pd.png Output/Step26/DADA2.observed_features.png Output/Step26/DADA2.shannon.png
.PHONY: step26

# Step 27 (Beta-diveristy pairwise permanova test)
Output/Step27/DADA2 Output/Step27/Deblur:
	mkdir -p $@

Output/Step27/DADA2/whole.%.qzv: Output/Step10/DADA2/whole.%_distance_matrix.qza Output/Step07/whole.tsv | build.log Output/Step27/DADA2
	$(DOCKER) qiime diversity beta-group-significance --i-distance-matrix $(addprefix /,$(filter %.qza,$^)) --m-metadata-file $(addprefix /,$(filter %.tsv,$^)) --m-metadata-column "LongStage" --p-pairwise --o-visualization $(addprefix /,$@) 1> $@.stdout 2> $@.stderr

Output/Step27/Deblur/whole.%.qzv: Output/Step10/Deblur/whole.%_distance_matrix.qza Output/Step07/whole.tsv | build.log Output/Step27/Deblur
	$(DOCKER) qiime diversity beta-group-significance --i-distance-matrix $(addprefix /,$(filter %.qza,$^)) --m-metadata-file $(addprefix /,$(filter %.tsv,$^)) --m-metadata-column "LongStage" --p-pairwise --o-visualization $(addprefix /,$@) 1> $@.stdout 2> $@.stderr

step27: Output/Step27/DADA2/whole.unweighted_unifrac.qzv Output/Step27/DADA2/whole.weighted_unifrac.qzv Output/Step27/DADA2/whole.jaccard.qzv Output/Step27/DADA2/whole.bray_curtis.qzv Output/Step27/Deblur/whole.unweighted_unifrac.qzv Output/Step27/Deblur/whole.weighted_unifrac.qzv Output/Step27/Deblur/whole.jaccard.qzv Output/Step27/Deblur/whole.bray_curtis.qzv
.PHONY: step27

# Step 28 (ANCOM cluster map)
Output/Step28:
	mkdir -p $@

Output/Step28/%.png: Python/step28.py Output/Step17/%.tar.gz | Output/Step28 build.log
	$(DOCKER) python3 $(addprefix /,$^ $@) 1> $@.stdout 2> $@.stderr

step28: Output/Step28/whole.DADA2.homd.png
.PHONY: step28

# Step 29 (ANCOM box plot)
Output/Step29:
	mkdir -p $@

Output/Step29/%.png: Python/step29.py Output/Step17/%.tar.gz | Output/Step29 build.log
	$(DOCKER) python3 $(addprefix /,$^ $@) 1> $@.stdout 2> $@.stderr

Output/Step29/%.tar: Python/step29-1.py Output/Step17/%.tar.gz | Output/Step29 build.log
	$(DOCKER) python3 $(addprefix /,$^ $@) 1> $@.stdout 2> $@.stderr

step29: Output/Step29/whole.DADA2.homd.png Output/Step29/whole.DADA2.homd.tar
.PHONY: step29

# Step 30 (LefSe)
Output/Step30 Output/Step30/Default Output/Step30/Cladogram:
	mkdir -p $@
	chmod 777 $@

Output/Step30/%.tsv: Python/step30.py Output/Step16/whole.%.tar.gz | Output/Step30 build.log
	$(DOCKER) python3 $(addprefix /,$^ $@) 1> $@.stdout 2> $@.stderr

Output/Step30/%.in: Output/Step30/%.tsv | build.log
	docker run --rm $(VOLUME_OPTS) $(RUN_OPTS) biobakery/lefse format_input.py $(addprefix /,$^ $@) -c 2 -u 1 1> $@.stdout 2> $@.stderr

Output/Step30/%.res: Output/Step30/%.in | build.log
	docker run --rm $(VOLUME_OPTS) $(RUN_OPTS) biobakery/lefse run_lefse.py $(addprefix /,$^ $@) --verbose 1 -y 1 1> $@.stdout 2> $@.stderr

Output/Step30/Default/%.pdf: Output/Step30/%.res | build.log Output/Step30/Default
	docker run --rm $(VOLUME_OPTS) $(RUN_OPTS) biobakery/lefse plot_res.py $(addprefix /,$^ $@) --format pdf 1> $@.stdout 2> $@.stderr || $(DOCKER) touch $(addprefix /,$@)

Output/Step30/Cladogram/%.res: Output/Step30/%.res | build.log Output/Step30/Cladogram
	grep --perl-regexp "True|False" $^ > $@ || cp -f $^ $@

Output/Step30/Cladogram/%.pdf: Output/Step30/Cladogram/%.res | build.log
	docker run --rm $(VOLUME_OPTS) $(RUN_OPTS) biobakery/lefse plot_cladogram.py $(addprefix /,$^ $@) --format pdf 1> $@.stdout 2> $@.stderr || $(DOCKER) touch $(addprefix /,$@)

step30: Output/Step30/DADA2.homd.tsv Output/Step30/DADA2.homd.in Output/Step30/DADA2.homd.res Output/Step30/Default/DADA2.homd.pdf Output/Step30/Cladogram/DADA2.homd.res Output/Step30/Cladogram/DADA2.homd.pdf
.PHONY: step30

# Step 31 (Data composition)
Output/Step31:
	mkdir -p $@

Output/Step31/whole.png: Python/step31.py Output/Step16/whole.DADA2.homd.tar.gz | build.log Output/Step31
	$(DOCKER) python3 $(addprefix /,$^ $@) 1> $@.stdout 2> $@.stderr

Output/Step31/ANCOM.png: Python/step31.py Output/Step17/whole.DADA2.homd.tar.gz | build.log Output/Step31
	$(DOCKER) python3 $(addprefix /,$^ $@) 1> $@.stdout 2> $@.stderr

step31: Output/Step31/whole.png Output/Step31/ANCOM.png
.PHONY: step31

# Step 32 (Draw pairplot)
Output/Step32:
	mkdir -p $@

Output/Step32/whole.png: Python/step32.py Output/Step17/whole.DADA2.homd.tar.gz | build.log Output/Step32
	$(DOCKER) python3 $(addprefix /,$^ $@) 1> $@.stdout 2> $@.stderr

Output/Step32/whole.tar: Python/step32-1.py Output/Step17/whole.DADA2.homd.tar.gz | build.log Output/Step32
	$(DOCKER) python3 $(addprefix /,$^ $@) 1> $@.stdout 2> $@.stderr

step32: Output/Step32/whole.png Output/Step32/whole.tar
.PHONY: step32

# Step 33 (Clinical statistics)
Output/Step33:
	mkdir -p $@

Output/Step33/stat.csv: Python/step33.py Data/치주염선정샘플_250샘플.xlsx Data/metadata.csv | Output/Step33 build.log
	$(DOCKER) python3 $(addprefix /,$^ $@) 1> $@.stdout 2> $@.stderr

step33: Output/Step33/stat.csv
.PHONY: step33

# Step 34 (Draw correlation with clinical data)
Output/Step34:
	mkdir -p $@

Output/Step34/clinical.tar: Python/step34.py Output/Step17/whole.DADA2.homd.tar.gz | Output/Step34 build.log
	$(DOCKER) python3 $(addprefix /,$^ $@) 1> $@.stdout 2> $@.stderr

step34: Output/Step34/clinical.tar
.PHONY: step34

# Step 35 (Pathway analysis)
Output/Step35:
	mkdir -p $@

Output/Step35/pathway.tar: Python/step35.py Output/Step35/path_abun_unstrat_descrip.xlsx | Output/Step35 build.log
	$(DOCKER) python3 $(addprefix /,$^ $@) 1> $@.stdout 2> $@.stderr

step35: Output/Step35/pathway.tar
.PHONY: step35
