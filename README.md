# baseLess: minION sequence detection, no basecalling required
BaseLess reduces your MinION sequencing device to a simple species detector. As a trade-off, it runs on inexpensive computational hardware like single-board computers. The neural network it uses is a composite of many smaller networks, each detecting a different k-mer, thus reconfiguration of baseLess to detect a different sequence amounts to combining different smaller networks. 

## Install
Install baseLess from conda:
```shell
conda install -c bioconda -c conda-forge -c cvdelannoy baseless
```
Before compiling your first model, you can choose to download pre-generated k-mer models into the default directory of the package (current size <1MB):
```shell
baseLess update_16s_db
```
We have generated models for 

## Running
### Abundance mode
BaseLess determines whether the reads originate from a given target genome or not, by comparing the abundance of k-mers it finds to the abundance of k-mers in the genomes.
- For pure samples
- Works with larger genomes
- Useful for differentiating visually similar species or strains

To generate a model:
```
baseLess compile_model \
  --model-type abundance \
  --target-fasta target.fasta \                                # fasta containing target genome
  --background-fastas background1.fasta backgroud2.fasta ... \ # genomes of other species
  --nn-directory directory/of/nns \                            # directory containing NNs
  --out-model your_model_name.h5
```

If you have reads of species close to your target species, you can use these as training reads and increase the probability of finding salient k-mers in your training data significantly. In that case, run:
```
baseLess compile_model \
  --model-type abundance \
  --target-fasta target.fasta \                                # fasta containing target genome
  --background-fastas background1.fasta backgroud2.fasta ... \ # genomes of other species
  --nn-directory directory/of/nns \                            # directory containing NNs
  --out-model your_model_name.h5  \
  --training-reads training/reads/directory \
  --cores 8 \                                                  # more CPUs == faster
  --train-on-the-fly
```
k-mer models will be generated as needed, so this will take longer to run (typically <1 hour). Reads will need to be corrected prior to usage for training, see [below](#Generating-k-mer-detection-networks).

### Read detection mode
BaseLess looks for all k-mers in each read. If a read contains a minimum fraction of total k-mers (>66%, by default), it is considered to originate from the target species. Though not necessarily restricted to any sequence, this has been tested for and attuned to 16S sequence at the moment.
- For amplified/metagenomic samples
- Useful for microbial species detection

To generate a model:
```
baseLess compile_model \
  --model-type read_detection \
  --target-fasta target.fasta \                                # fasta containing one or more 16S sequences
  --out-model your_model_name.h5
```

### Inference
To run inference simultaneously with the MinION, change configuration so that one read per fast5 is returned, and point baseLess to the fast5 output directory of your reads:
```
baseLess run_inference \
    --fast5-in directory/of/fast5s \ 
    --out-dir your/output/directory \
    --model your_model_name.h5 \
    --inference-mode once
```
- In read detection mode, baseLess will only retain fast5s that contain your target. In abundance mode, no reads will be retained.
- baseLess will run indefinitely, until stopped manually (`ctrl-c`).
- To run on a directory of reads and stop when all reads are analysed, provide `--inference-mode once`. If you do not want the reads to be removed, add `--copy-reads`.

### Generating k-mer detection networks
baseLess comes with a pre-generated set of k-mer detection networks, for k-mers that should be particularly salient in the context of 16S detection. We plan to further expand the number of available k-mer networks, however if you have a large read set of a known species, a good reference genome, and you want to generate k-mer networks yourself, you can do so as follows.

Basecall your reads with guppy if you haven't yet. If reads were output as multi-fast5, split them using from the [ont-fast5-api](https://pypi.org/project/ont-fast5-api/) package:
```
single_to_multi_fast5 -i multi/fast5/input/folder -s single/fast5/output/folder
```

Correct your reads with [tombo](https://anaconda.org/bioconda/ont-tombo):
```
tombo resquiggle single/fast5/output/folder reference_genome.fasta
```

Use these reads as input for baseLess:
```
baseLess run_production_pipeline \
  --training-reads path/to/training/reads \
  --test-reads path/to/test/reads \
  --out-dir path/to/networks/directory \
  --cores 8 \                                # more CPUs == faster
  --kmer-list                                # txt list, 1 line per k-mer
```

## Minimal hardware
### Running on Nvidia Jetson Nano (2GB)
The Nvidia Jetson Nano is a single-board computer with dedicated GPU. Designed to be baseLess is light enough to run on 

Before first usage open a terminal in the baseless directory and build the image:

```shell
docker build -f Dockerfile_jetson -t baseless .
```

open the docker container interactively:
```shell
docker run -it --rm --runtime nvidia --network host baseless
```
Add any directories you may want to read/write by adding `-v /local/dir:/home/some_dir_name `

You can now start running inference! As the Jetson Nano has little memory to spare (2GB or 4GB, depending on your model), you may need to limit memory using the `--mem` option. We found that the 2GB model runs well with 512mb (`--mem 512`).

Some tips to make the most of your limited memory:
- Switch off the GUI -- `sudo systemctl stop lxdm` or `sudo sysctl disable lxdm` to prevent the GUI starting at startup. You can also follow Zafer Arican's [advice](https://www.zaferarican.com/post/how-to-save-1gb-memory-on-jetson-nano-by-installing-lubuntu-desktop) and switch to lubuntu.
