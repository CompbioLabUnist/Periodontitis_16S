# Variables
DATE := $(shell date "+%y%m%d")
RANDOM := $(shell bash -c 'echo $$RANDOM')
IMAGE_NAME = periodontitis:latest
CPUS = 10
MEMS = 100G
PWD := $(shell pwd)
TOOLS = 
DOCKER = docker run --rm $(VOLUME_OPTS) $(RUN_OPTS) $(IMAGE_NAME)

# Options
VOLUME_OPTS = --volume $(abspath Output):/Output --volume $(abspath Data):/Data --volume $(abspath Python):/Python
RUN_OPTS = --tty --cpus="$(CPUS)" --memory="$(MEMS)"

# General
all:
.PHONY += all

log Output Docker/Tools:
	mkdir $@

# Docker
TOOLS += $(wildcard Docker/*)
build.log: Docker/Dockerfile $(TOOLS) | log Output
	rm -fv $@
	docker images | grep $(IMAGE_NAME) && docker rmi $(IMAGE_NAME) || true
	docker build --rm --tag $(IMAGE_NAME) $(<D) | tee $@

build: build.log
.PHONY += build

interactive: build.log
	docker run --rm $(VOLUME_OPTS) $(RUN_OPTS) --interactive $(IMAGE_NAME) /bin/bash || true
.PHONY += interactive

delete: build.log
	docker rmi $(IMAGE_NAME)
	rm -fv build.log
.PHONY += delete

stop:
	docker rm $(CONTAINER_NAME)

# SGE
tmp.sh: build.log
	echo "make -C $(PWD) latest" > tmp.sh

run: tmp.sh | log Output
	qsub -cwd -l h_vmem=$(MEMS) -m abe -M "230@fumire.moe" -N Periodontitis_$(DATE) -pe smp $(CPUS) -o $(abspath log) -e $(abspath log) $<
.PHONY += run

# Actual
latest: step14
.PHONY += step01

# Step 01 (Make manifest files)
Output/Step01:
	mkdir -p $@

Output/Step01/whole.tsv: Python/step01.py $(wildcard Data/fastq/*.fastq.gz) | Output/Step01 build.log
	@$(DOCKER) python3 $(addprefix /,$^) > $@

step01: Output/Step01/whole.tsv
.PHONY += step01

# Step 02 (De-multiplexing)
Output/Step02:
	mkdir -p $@

Output/Step02/%.qza: Output/Step01/%.tsv | Output/Step02 build.log
	$(DOCKER) qiime tools import --type "SampleData[PairedEndSequencesWithQuality]" --input-format "PairedEndFastqManifestPhred33V2" --input-path $(addprefix /,$<) --output-path $(addprefix /,$@) 1> $@.stdout 2> $@.stderr

Output/Step02/%.qzv: Output/Step02/%.qza | build.log
	$(DOCKER) qiime demux summarize --i-data $(addprefix /,$<) --o-visualization $(addprefix /,$@) 1> $@.stdout 2> $@.stderr

step02: Output/Step02/whole.qza Output/Step02/whole.qzv
.PHONY += step02

# Step 03 (Quality Filter)
Output/Step03:
	mkdir -p $@

Output/Step03/%.seq.qza Output/Step03/%.stat.qza: Output/Step02/%.qza | Output/Step03 build.log
	$(DOCKER) qiime quality-filter q-score --i-demux $(addprefix /,$<) --o-filtered-sequences $(addprefix /,$(basename $(basename $@))).seq.qza --o-filter-stats $(addprefix /,$(basename $(basename $@))).stat.qza 1> $@.stdout 2> $@.stderr

step03: Output/Step03/whole.seq.qza
.PHONY += step03

# Step 04 (DADA2 Denoising)
Output/Step04:
	mkdir -p $@

Output/Step04/%.table.qza Output/Step04/%.seq.qza Output/Step04/%.stat.qza: Output/Step02/%.qza | Output/Step04 build.log
	@echo "You have to check the length manually!!"
	$(DOCKER) qiime dada2 denoise-paired --i-demultiplexed-seqs $(addprefix /,$<) --p-n-threads $(CPUS) --p-trunc-len-f 300 --p-trunc-len-r 265 --o-table $(addprefix /,$(basename $(basename $@))).table.qza --o-representative-sequences $(addprefix /,$(basename $(basename $@))).seq.qza --o-denoising-stats $(addprefix /,$(basename $(basename $@))).stat.qza 1> $@.stdout 2> $@.stderr

step04: Output/Step04/whole.seq.qza
.PHONY += step04

# Step 05 (Deblur Denoising)
Output/Step05:
	mkdir -p $@

Output/Step05/%.table.qza Output/Step05/%.seq.qza Output/Step05/%.stat.qza: Output/Step03/%.seq.qza | Output/Step05 build.log
	@echo "You have to check the length manually!!"
	$(DOCKER) qiime deblur denoise-16S --i-demultiplexed-seqs $(addprefix /,$<) --p-trim-length 265 --p-jobs-to-start $(CPUS) --o-table $(addprefix /,$(basename $(basename $@))).table.qza --o-representative-sequences $(addprefix /,$(basename $(basename $@))).seq.qza --o-stats $(addprefix /,$(basename $(basename $@))).stat.qza 1> $@.stdout 2> $@.stderr

step05: Output/Step05/whole.seq.qza
.PHONY += step05

# Step 06 (Visulaize Denoising)
Output/Step06:
	mkdir -p $@

Output/Step06/%.Demux.stat.qzv: Output/Step03/%.stat.qza | Output/Step06 build.log
	$(DOCKER) qiime metadata tabulate --m-input-file $(addprefix /,$<) --o-visualization $(addprefix /,$@) 1> $@.stdout 2> $@.stderr

Output/Step06/%.Deblur.stat.qzv: Output/Step05/%.stat.qza | Output/Step06 build.log
	$(DOCKER) qiime deblur visualize-stats --i-deblur-stats $(addprefix /,$<) --o-visualization $(addprefix /,$@) 1> $@.stdout 2> $@.stderr

step06: Output/Step06/whole.Demux.stat.qzv Output/Step06/whole.Deblur.stat.qzv
.PHONY += step06

# Step 07 (Make metadata)
Output/Step07:
	mkdir -p $@

Output/Step07/whole.tsv: Python/step07.py Data/치주염선정샘플_250샘플.xlsx $(wildcard Data/fastq/*.fastq.gz) | Output/Step07 build.log
	@$(DOCKER) python3 $(addprefix /,$^) > $@

step07: Output/Step07/whole.tsv
.PHONY += step07

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

step08: Output/Step08/whole.DADA.table.qzv Output/Step08/whole.DADA.seq.qzv Output/Step08/whole.Deblur.table.qzv Output/Step08/whole.Deblur.seq.qzv
.PHONY += step08

# Step 09 (Phylogenetic diversity analysis)
Output/Step09:
	mkdir -p $@

Output/Step09/%.DADA.aligned-rep-seqs.qza Output/Step09/%.DADA.masked-aligned-rep-seqs.qza Output/Step09/%.DADA.unrooted-tree.qza Output/Step09/%.DADA.rooted-tree.qza: Output/Step04/%.seq.qza | Output/Step09 build.log
	$(DOCKER) qiime phylogeny align-to-tree-mafft-fasttree --i-sequences $(addprefix /,$<) --o-alignment $(addprefix /,$(basename $(basename $@))).aligned-rep-seqs.qza --o-masked-alignment $(addprefix /,$(basename $(basename $@))).masked-aligned-rep-seqs.qza --o-tree $(addprefix /,$(basename $(basename $@))).unrooted-tree.qza --o-rooted-tree $(addprefix /,$(basename $(basename $@))).rooted-tree.qza --p-n-threads $(CPUS) 1> $@.stdout 2> $@.stderr

Output/Step09/%.Deblur.aligned-rep-seqs.qza Output/Step09/%.Deblur.masked-aligned-rep-seqs.qza Output/Step09/%.Deblur.unrooted-tree.qza Output/Step09/%.Deblur.rooted-tree.qza: Output/Step05/%.seq.qza | Output/Step09 build.log
	$(DOCKER) qiime phylogeny align-to-tree-mafft-fasttree --i-sequences $(addprefix /,$<) --o-alignment $(addprefix /,$(basename $(basename $@))).aligned-rep-seqs.qza --o-masked-alignment $(addprefix /,$(basename $(basename $@))).masked-aligned-rep-seqs.qza --o-tree $(addprefix /,$(basename $(basename $@))).unrooted-tree.qza --o-rooted-tree $(addprefix /,$(basename $(basename $@))).rooted-tree.qza --p-n-threads $(CPUS) 1> $@.stdout 2> $@.stderr

step09: Output/Step09/whole.DADA.rooted-tree.qza Output/Step09/whole.Deblur.rooted-tree.qza
.PHONY += step09

# Step 10 (Core metrics phylogenetic)
Output/Step10/DADA2 Output/Step10/Deblur:
	mkdir -p $@

Output/Step10/DADA2/%.faith_pd_vector.qza Output/Step10/DADA2/%.unweighted_unifrac_distance_matrix.qza Output/Step10/DADA2/%.bray_curtis_pcoa_results.qza Output/Step10/DADA2/%.shannon_vector.qza Output/Step10/DADA2/%.rarefied_table.qza Output/Step10/DADA2/%.weighted_unifrac_distance_matrix.qza Output/Step10/DADA2/%.jaccard_pcoa_results.qza Output/Step10/DADA2/%.weighted_unifrac_pcoa_results.qza Output/Step10/DADA2/%.observed_features_vector.qza Output/Step10/DADA2/%.jaccard_distance_matrix.qza Output/Step10/DADA2/%.evenness_vector.qza Output/Step10/DADA2/%.bray_curtis_distance_matrix.qza Output/Step10/DADA2/%.unweighted_unifrac_pcoa_results.qza Output/Step10/DADA2/%.unweighted_unifrac_emperor.qzv Output/Step10/DADA2/%.jaccard_emperor.qzv Output/Step10/DADA2/%.bray_curtis_emperor.qzv Output/Step10/DADA2/%.weighted_unifrac_emperor.qzv: Output/Step09/%.DADA.rooted-tree.qza Output/Step04/%.table.qza Output/Step07/%.tsv | Output/Step10/DADA2 build.log
	@echo "You have to check the sequencing depth manually!!"
	$(DOCKER) qiime diversity core-metrics-phylogenetic --i-phylogeny $(addprefix /,$(word 1,$^)) --i-table $(addprefix /,$(word 2,$^)) --m-metadata-file $(addprefix /,$(word 3,$^)) --p-sampling-depth 3786 --o-rarefied-table $(addprefix /,$(basename $(basename $@))).rarefied_table.qza --o-faith-pd-vector $(addprefix /,$(basename $(basename $@))).faith_pd_vector.qza --o-observed-features-vector $(addprefix /,$(basename $(basename $@))).observed_features_vector.qza --o-shannon-vector $(addprefix /,$(basename $(basename $@))).shannon_vector.qza --o-evenness-vector $(addprefix /,$(basename $(basename $@))).evenness_vector.qza --o-unweighted-unifrac-distance-matrix $(addprefix /,$(basename $(basename $@))).unweighted_unifrac_distance_matrix.qza --o-weighted-unifrac-distance-matrix $(addprefix /,$(basename $(basename $@))).weighted_unifrac_distance_matrix.qza --o-jaccard-distance-matrix $(addprefix /,$(basename $(basename $@))).jaccard_distance_matrix.qza --o-bray-curtis-distance-matrix $(addprefix /,$(basename $(basename $@))).bray_curtis_distance_matrix.qza --o-unweighted-unifrac-pcoa-results $(addprefix /,$(basename $(basename $@))).unweighted_unifrac_pcoa_results.qza --o-weighted-unifrac-pcoa-results $(addprefix /,$(basename $(basename $@))).weighted_unifrac_pcoa_results.qza --o-unweighted-unifrac-emperor $(addprefix /,$(basename $(basename $@))).unweighted_unifrac_emperor.qzv --o-weighted-unifrac-emperor $(addprefix /,$(basename $(basename $@))).weighted_unifrac_emperor.qzv --o-jaccard-emperor $(addprefix /,$(basename $(basename $@))).jaccard_emperor.qzv --o-bray-curtis-emperor $(addprefix /,$(basename $(basename $@))).bray_curtis_emperor.qzv --o-jaccard-pcoa-results $(addprefix /,$(basename $(basename $@))).jaccard_pcoa_results.qza --o-bray-curtis-pcoa-results $(addprefix /,$(basename $(basename $@))).bray_curtis_pcoa_results.qza --p-n-jobs-or-threads $(CPUS) 1> $@.stdout 2> $@.stderr

Output/Step10/Deblur/%.faith_pd_vector.qza Output/Step10/Deblur/%.unweighted_unifrac_distance_matrix.qza Output/Step10/Deblur/%.bray_curtis_pcoa_results.qza Output/Step10/Deblur/%.shannon_vector.qza Output/Step10/Deblur/%.rarefied_table.qza Output/Step10/Deblur/%.weighted_unifrac_distance_matrix.qza Output/Step10/Deblur/%.jaccard_pcoa_results.qza Output/Step10/Deblur/%.weighted_unifrac_pcoa_results.qza Output/Step10/Deblur/%.observed_features_vector.qza Output/Step10/Deblur/%.jaccard_distance_matrix.qza Output/Step10/Deblur/%.evenness_vector.qza Output/Step10/Deblur/%.bray_curtis_distance_matrix.qza Output/Step10/Deblur/%.unweighted_unifrac_pcoa_results.qza Output/Step10/Deblur/%.unweighted_unifrac_emperor.qzv Output/Step10/DADA2/%.jaccard_emperor.qzv Output/Step10/Deblur/%.bray_curtis_emperor.qzv Output/Step10/Deblur/%.weighted_unifrac_emperor.qzv: Output/Step09/%.Deblur.rooted-tree.qza Output/Step05/%.table.qza Output/Step07/%.tsv | Output/Step10/Deblur build.log
	@echo "You have to check the sequencing depth manually!!"
	$(DOCKER) qiime diversity core-metrics-phylogenetic --i-phylogeny $(addprefix /,$(word 1,$^)) --i-table $(addprefix /,$(word 2,$^)) --m-metadata-file $(addprefix /,$(word 3,$^)) --p-sampling-depth 7253 --o-rarefied-table $(addprefix /,$(basename $(basename $@))).rarefied_table.qza --o-faith-pd-vector $(addprefix /,$(basename $(basename $@))).faith_pd_vector.qza --o-observed-features-vector $(addprefix /,$(basename $(basename $@))).observed_features_vector.qza --o-shannon-vector $(addprefix /,$(basename $(basename $@))).shannon_vector.qza --o-evenness-vector $(addprefix /,$(basename $(basename $@))).evenness_vector.qza --o-unweighted-unifrac-distance-matrix $(addprefix /,$(basename $(basename $@))).unweighted_unifrac_distance_matrix.qza --o-weighted-unifrac-distance-matrix $(addprefix /,$(basename $(basename $@))).weighted_unifrac_distance_matrix.qza --o-jaccard-distance-matrix $(addprefix /,$(basename $(basename $@))).jaccard_distance_matrix.qza --o-bray-curtis-distance-matrix $(addprefix /,$(basename $(basename $@))).bray_curtis_distance_matrix.qza --o-unweighted-unifrac-pcoa-results $(addprefix /,$(basename $(basename $@))).unweighted_unifrac_pcoa_results.qza --o-weighted-unifrac-pcoa-results $(addprefix /,$(basename $(basename $@))).weighted_unifrac_pcoa_results.qza --o-unweighted-unifrac-emperor $(addprefix /,$(basename $(basename $@))).unweighted_unifrac_emperor.qzv --o-weighted-unifrac-emperor $(addprefix /,$(basename $(basename $@))).weighted_unifrac_emperor.qzv --o-jaccard-emperor $(addprefix /,$(basename $(basename $@))).jaccard_emperor.qzv --o-bray-curtis-emperor $(addprefix /,$(basename $(basename $@))).bray_curtis_emperor.qzv --o-jaccard-pcoa-results $(addprefix /,$(basename $(basename $@))).jaccard_pcoa_results.qza --o-bray-curtis-pcoa-results $(addprefix /,$(basename $(basename $@))).bray_curtis_pcoa_results.qza --p-n-jobs-or-threads $(CPUS) 1> $@.stdout 2> $@.stderr

step10: Output/Step10/DADA2/whole.faith_pd_vector.qza Output/Step10/Deblur/whole.faith_pd_vector.qza
.PHONY += step10

# Step 11 (Alpha-diversity)
Output/Step11/DADA2 Output/Step11/Deblur:
	mkdir -p $@

Output/Step11/DADA2/%.qzv: Output/Step10/DADA2/%.qza Output/Step07/whole.tsv | Output/Step11/DADA2 build.log
	$(DOCKER) qiime diversity alpha-group-significance --i-alpha-diversity $(addprefix /,$(word 1,$^)) --m-metadata-file $(addprefix /,$(word 2,$^)) --o-visualization $(addprefix /,$@) 1> $@.stdout 2> $@.stderr

Output/Step11/Deblur/%.qzv: Output/Step10/Deblur/%.qza Output/Step07/whole.tsv | Output/Step11/Deblur build.log
	$(DOCKER) qiime diversity alpha-group-significance --i-alpha-diversity $(addprefix /,$(word 1,$^)) --m-metadata-file $(addprefix /,$(word 2,$^)) --o-visualization $(addprefix /,$@) 1> $@.stdout 2> $@.stderr

step11: Output/Step11/DADA2/whole.faith_pd_vector.qzv Output/Step11/DADA2/whole.observed_features_vector.qzv Output/Step11/DADA2/whole.shannon_vector.qzv Output/Step11/DADA2/whole.evenness_vector.qzv Output/Step11/Deblur/whole.faith_pd_vector.qzv Output/Step11/Deblur/whole.observed_features_vector.qzv Output/Step11/Deblur/whole.shannon_vector.qzv Output/Step11/Deblur/whole.evenness_vector.qzv
.PHONY += step11

# Step 12 (Beta-diversity)
Output/Step12/DADA2 Output/Step12/Deblur:
	mkdir -p $@

Output/Step12/DADA2/%.qzv: Output/Step10/DADA2/%.qza Output/Step07/whole.tsv | Output/Step12/DADA2 build.log
	$(DOCKER) qiime diversity beta-group-significance --i-distance-matrix $(addprefix /,$(word 1,$^)) --m-metadata-file $(addprefix /,$(word 2,$^)) --m-metadata-column "LongStage" --o-visualization $(addprefix /,$@) --p-pairwise

Output/Step12/Deblur/%.qzv: Output/Step10/Deblur/%.qza Output/Step07/whole.tsv | Output/Step12/Deblur build.log
	$(DOCKER) qiime diversity beta-group-significance --i-distance-matrix $(addprefix /,$(word 1,$^)) --m-metadata-file $(addprefix /,$(word 2,$^)) --m-metadata-column "LongStage" --o-visualization $(addprefix /,$@) --p-pairwise

step12: Output/Step12/DADA2/whole.unweighted_unifrac_distance_matrix.qzv Output/Step12/DADA2/whole.weighted_unifrac_distance_matrix.qzv Output/Step12/DADA2/whole.jaccard_distance_matrix.qzv Output/Step12/DADA2/whole.bray_curtis_distance_matrix.qzv Output/Step12/Deblur/whole.unweighted_unifrac_distance_matrix.qzv Output/Step12/Deblur/whole.weighted_unifrac_distance_matrix.qzv Output/Step12/Deblur/whole.jaccard_distance_matrix.qzv Output/Step12/Deblur/whole.bray_curtis_distance_matrix.qzv
.PHONY += step12

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

step13: Output/Step13/whole.DADA2.silva.qza Output/Step13/whole.DADA2.gg.qza Output/Step13/whole.Deblur.silva.qza Output/Step13/whole.Deblur.gg.qza
.PHONY += step13

# Step 14 (Run ANCOM)
Output/Step14:
	mkdir -p $@

Output/Step14/%.DADA2.silva.qza: Output/Step04/%.table.qza Output/Step13/%.DADA2.silva.qza | Output/Step14 build.log
	$(DOCKER) qiime taxa collapse --i-table $(addprefix /,$(word 1,$^)) --i-taxonomy $(addprefix /,$(word 2,$^)) --p-level 7 --o-collapsed-table $(addprefix /,$@) 1> $@.stdout 2> $@.stderr

Output/Step14/%.DADA2.gg.qza: Output/Step04/%.table.qza Output/Step13/%.DADA2.gg.qza | Output/Step14 build.log
	$(DOCKER) qiime taxa collapse --i-table $(addprefix /,$(word 1,$^)) --i-taxonomy $(addprefix /,$(word 2,$^)) --p-level 7 --o-collapsed-table $(addprefix /,$@) 1> $@.stdout 2> $@.stderr

Output/Step14/%.Deblur.silva.qza: Output/Step05/%.table.qza Output/Step13/%.Deblur.silva.qza | Output/Step14 build.log
	$(DOCKER) qiime taxa collapse --i-table $(addprefix /,$(word 1,$^)) --i-taxonomy $(addprefix /,$(word 2,$^)) --p-level 7 --o-collapsed-table $(addprefix /,$@) 1> $@.stdout 2> $@.stderr

Output/Step14/%.Deblur.gg.qza: Output/Step05/%.table.qza Output/Step13/%.Deblur.gg.qza | Output/Step14 build.log
	$(DOCKER) qiime taxa collapse --i-table $(addprefix /,$(word 1,$^)) --i-taxonomy $(addprefix /,$(word 2,$^)) --p-level 7 --o-collapsed-table $(addprefix /,$@) 1> $@.stdout 2> $@.stderr

Output/Step14/%.pseudocount.qza: Output/Step14/%.qza | build.log
	$(DOCKER) qiime composition add-pseudocount --i-table $(addprefix /,$(word 1,$^)) --o-composition-table $(addprefix /,$@) 1> $@.stdout 2> $@.stderr

Output/Step14/%.site.qzv: Output/Step14/%.qza Output/Step07/whole.tsv | build.log
	$(DOCKER) qiime composition ancom --i-table $(addprefix /,$(word 1,$^)) --m-metadata-file $(addprefix /,$(word 2,$^)) --m-metadata-column "LongStage" --o-visualization $(addprefix /,$@) 1> $@.stdout 2> $@.stderr

Output/Step14/%/ancom.tsv: Output/Step14/%.qzv | build.log
	$(DOCKER) rm -rfv $(addprefix /,$(@D))
	$(DOCKER) qiime tools export --input-path $(addprefix /,$(word 1,$^)) --output-path $(addprefix /,$(@D))

step14: Output/Step14/whole.DADA2.gg.pseudocount.site/ancom.tsv Output/Step14/whole.DADA2.silva.pseudocount.site/ancom.tsv Output/Step14/whole.Deblur.gg.pseudocount.site/ancom.tsv Output/Step14/whole.Deblur.silva.pseudocount.site/ancom.tsv
.PHONY += step14
