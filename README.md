# Introduction
This project builds on the work of Taguchi et al. from [Universal Automatic Phonetic Transcription into the International Phonetic Alphabet](https://www.isca-archive.org/interspeech_2023/taguchi23_interspeech.pdf) with the goal of creating speech recognition models that produce high quality phonetic transcriptions of spontaneous speech for dialects of American English.

Note that we have made significant changes from the original fork, which was focused on universal transcription for multiple languages with the same model. In contrast, we are interested in creating extremely high quality transcripts for a single language and understanding how much data is necessary to achieve the desired quality.

Visit our HuggingFace Spaces for [demos of the trained models](https://huggingface.co/spaces/ginic/multipa-english-to-ipa) or [detailed explanations on evaluation metrics](https://huggingface.co/spaces/ginic/phone_errors).

# Acknowledgements
This started as a [Data Core Seed Funding](https://ds.cs.umass.edu/data-core-seed-funding) engagement between [UMass Amherst Center for Data Science](https://ds.cs.umass.edu/) and [UMass Amherst Department of Linguistics](https://www.umass.edu/linguistics/). We gratefully acknowledge the support of the Center for Data Science and Artificial Intelligence, the HFA/CICS Collaborative Seed Fund, and the Public Interest Technology Initiative, all at UMass Amherst.

The computational resources for this work are provided by the Unity Research Computing Platform, a multi-institutional cluster lead by University of Massachusetts Amherst, the University of Rhode Island, and University of Massachusetts Dartmouth.

# Installation
1. Create a conda environment using `conda env create --name <your_env_name> --file=<your_env_file>.yml`. A conda environment for cuda 12.4 has been provided in `multipa_cuda124.yml` for GPUs. A CPU compatible environment is in `multipa.yml`. For other installation options and GPU settings, see [pytorch.org](https://pytorch.org). Note  that the conda environment files also install the separate command line program [ffmpeg](https://github.com/FFmpeg/FFmpeg) to the environment, which is used in newer versions of the `transformers` library with `torchcodec` for manipulating audio more quickly. However, this is optional. If you choose to install the package without using conda or installing ffmpeg manually, the code will fall back to using `soundfile` library, which works well, but may be slower. Results obtained using `torchcodec` and `soundfile` should be similar, but may not be exactly the same, due to slight differences in audio decoding. 

2. Then install remaining requirements: `pip install .`. You can run `pip install -e .[dev,test]` for the developer set up, after which the best way to check that the installation worked is to run unit tests with `pytest`. You will also need to download the unidic dictionary with `python -m unidic download` (in the same python environment as you installation).

For convenience, scripts for setup and installation are provided in `scripts/setup.sh` (local or personal computer) and `scripts/setup_slurm.sh` (Slurm compute cluster).

# How to run
The pipeline consists of scripts for data preprocessing, model training and model evaluation.

## Data Preprocessing
First, the corpus must be converted to the appropriate HuggingFace format with IPA transcriptions of audio. This is done with the `src/multipa/preprocess.py` script, which can be run as `multipa-preprocess` after installation.
Three input corpora are supported each with various associated options. To see all options, run `multipa-preprocess --help` or `multipa-preprocess <corpus> --help`. For all corpora you will need to specify an `--output_dir` output folder and `--num_proc` for number of processes in HuggingFace transformers.

1) `librispeech`: Loads the "clean" portions of [LibriSpeech from HuggingFace](https://huggingface.co/datasets/openslr/librispeech_asr) with "Train.100" as the training data and "Valid" as the validation set. LibriSpeech is English read speech.
2) `commonvoice`: Loads data from [Common Voice Corpus 11.0 from HuggingFace](https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0) dataset for language codes specified using the `--languages/-l` option. Supported language codes are "ja", "pl", "mt", "hu", "fi", "el", "ta" (Japanese, Polish, Maltese, Hungarian, Finnish, Modern Greek and Tamil). Common Voice uses orthographic transcriptions which are converted to IPA using rules from the `multipa.converter` package. This corpus is a disk-space hog, so you will probably need to [clear the HuggingFace cache](https://huggingface.co/docs/datasets/v2.15.0/en/cache#cache-directory) using `--clear_cache` and `--cache_dir` options.
3) `buckeye`: Takes as input a folder containing predefined train/dev/test splits of the [Buckeye Corpus](https://buckeyecorpus.osu.edu). Your input folder should contain directories named "Train", "Dev" and "Test" and each of those folders must contain audio files, header-less TSV files for orthographic and IPA transcriptions, each with the following columns:
    - Column 1 is the utterance label. Add ".wav" to this to get the audio file for that utterance. The first two digits give the speaker #, the next two digits give the original file number from Buckeye.
    - Column 2 is the length of the utterance in seconds
    - Column 3 is the transcription of the utterance


⚠️ *Warning*: For the purposes of this work, `librispeech` and `commonvoice` options have not been thoroughly tested, but should retain backwards compatibility with the original Taguchi et al. work.


## Training
Model training is done using `src/multipa/main.py`, which can be run as `multipa-train` after installation. This script has many training options and subparsers for each supported corpus (`librispeech`, `commonvoice`, `buckeye`), viewable with `multipa-train --help` or `multipa-train <corpus> --help`. The shared options are for setting model hyper-parameters and the corpus-specific options are for selecting the number of data samples to use in training/validation and additional filters for sample quality or speaker demographics.

Many examples of model training on Buckeye are available in `scripts/buckeye_experiments`.

To train a model on Common Voice with 7 languages, 1000 training samples and 200 validation samples for each, where audio samples with bad quality are not filtered out, the suffix for the output model folder name is `japlmthufielta-nq-ns`, orthographic spaces are removed, and the number of epochs is set to 10:
```
multipa-train  --num_train_epochs 10 \
        commonvoice \
        -l ja pl mt hu fi el ta \
        -tr 1000 1000 1000 1000 1000 1000 1000 \
        -te 200 200 200 200 200 200 200 \
        -qf False False False False False False False \
        --suffix "japlmthufielta-nq-ns" \
        --no_space \

```



## Evaluation
Metrics used for model evaluation are described with examples at https://huggingface.co/spaces/ginic/phone_errors.

To evaluate an existing models' performance on the test split of the Buckeye corpus, you can run `scrc/multipa/evaluate.py` as `multipa-evaluate` and list models from Hugging Face with the `--hf_models` arg or models saved in local files with the `--local_models` flag. If you used the `--no_space` flag during training, you should use it during evaluation as well. Run `multipa-evaluate --help` for more info. This script currently only works with the Buckeye test corpus obtained during the data preprocessing step. For example:
```
multipa-evaluate --hf_models ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa-plus-2000 \
 --local_models data/buckeye_model/wav2vec2-large-xlsr-buckeye-ipa_test \
 --eval_out data/test_eval/buckeye_eval.csv \
 --verbose_results_dir data/test_eval/detailed_results --no_space --data_dir <path_to_preprocessed_buckeye_folder>
 ```

 Additional external models and tools, such as [Allosaurus](https://github.com/xinjli/allosaurus) and [Epitran](https://github.com/dmort27/epitran), have been evaluated in `notebooks/evaluation_extras.ipynb`. To reproduce these, first install the `dev` extras (`pip install .[dev]`).

## Model
You can run the original Taguchi et al. model (trained on 1k samples for each language, 9h in total) [here](https://huggingface.co/ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns) or play with multiple models at https://huggingface.co/spaces/ginic/multipa-english-to-ipa.

## Notes
- If you are using AFS, `preprocess.py` might cause `OS Error: File too large` due to reaching the limit of the number of files that a directory can accommodate.
- Additional data from Forvo themselves are not uploaded in this repository.
- The full list of IPA symbols was obtained from the [Panphon](https://github.com/dmort27/panphon) library.

## Known Issues
- `RuntimeError: Expected input_lengths to have value at least 0, but got value -1 (while checking arguments for ctc_loss_gpu)` appears sometimes just before training starts. This is non-deterministic and can be fixed by changing `--train_seed`.

## Citations
If you use this code, please cite the following papers:

Partridge, Virginia, Joe Pater, Parth Bhangla, Ali Nirheche and Brandon Prickett. 2025/to appear. AI-assisted analysis of phonological variation in English. Special session on Deep Phonology, AMP 2025, UC Berkeley. To appear in the Proceedings of AMP 2025.
```
@inproceedings{partridge2025_amp,
  author = {Partridge, Virginia, Joe Pater, Parth Bhangla, Ali Nirheche and Brandon Prickett},
  title = {{Automated phonetic transcription for varieties of English: wav2vec 2.0 fine-tuned on the Buckeye Corpus}},
  year=2025,
  booktitle={Proceedings of the Annual Meeting of Phonology 2025},
}
```


Chihiro Taguchi, Yusuke Sakai, Parisa Haghani, David Chiang. "Universal Automatic Phonetic Transcription into the International Phonetic Alphabet". INTERSPEECH 2023.
```
@inproceedings{taguchi23_interspeech,
  author={Chihiro Taguchi and Yusuke Sakai and Parisa Haghani and David Chiang},
  title={{Universal Automatic Phonetic Transcription into the International Phonetic Alphabet}},
  year=2023,
  booktitle={Proc. INTERSPEECH 2023},
  pages={2548--2552},
  doi={10.21437/Interspeech.2023-2584}
}
```

## Contact
Feel free to file a GitHub issue for bugs or questions. However, please understand that our team is small and may not be able to respond to all requests.

# CMU Dict License
The [Carnegie Mellon Pronouncing Dictionary](https://github.com/Alexir/CMUdict/tree/master) distributed with this code in `src/multipa/resources/cmudict-0.7b-ipa.txt` is covered by the following BSD 3-Clause License. This license does not apply to other code in this repository.
```
Copyright (c) 2015, Alexander Rudnicky
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of dictTools nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```