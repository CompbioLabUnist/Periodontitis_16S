# Variables
DATE := $(shell date "+%y%m%d")
RANDOM := $(shell bash -c 'echo $$RANDOM')
IMAGE_NAME = periodontitis:latest
CPUS = 20
MEMS = 200G
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
	qsub -cwd -l h_vmem=$(MEMS) -m abe -M "230@fumire.moe" -N Premature_$(DATE) -pe smp $(CPUS) -o $(abspath log) -e $(abspath log) $<
.PHONY += run

# Actual
latest: step01
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
