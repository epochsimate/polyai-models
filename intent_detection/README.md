[![PolyAI](polyai-logo.png)](https://poly-ai.com/)


# Intent detection models


This section shares the intent detection models used in the benchmarking 
presented in INTENT PAPER URL. These intent detectors are very lightweight 
and can be trained even in a laptop cpu, while still having superior 
performance than some popular larger models. They also show very good 
performance in low data settings.
These models are based on classifiers trained on top 2 popular sentence 
encoders: ConveRT and USE. We also use BERT_large as a baseline, both as a 
fixed sentence encoder and finetuning the whole model. We also share 
scripts to profile the intent detection models.

* [Requirements](#requirements)
* [Benchmarked datasets](#Benchmarked-datasets)
* [Models](#models)
* [Citations](#citations)


# Requirements

Using these models requires Tensorflow 1.14 and [Tensorflow Hub](https://www.tensorflow.org/hub):

* Convert: Tensorflow Text 0.6.0
* USE: tf_sentencepiece 0.1.83
* BERT: bert-tensorflow 1.0.1

Also, python 3.6 is required, because of `tf_sentencepiece` and `Tensorflow 1.14` compatibility issues.

# Benchmarked datasets

## HWU NLU dataset

Dataset released by the Heriot-Watt University composed of popular personal assistant queries.

[Dataset paper](https://arxiv.org/pdf/1903.05566.pdf)
[Dataset repository](https://github.com/xliuhw/NLU-Evaluation-Data)

## Clinc Intent Classification dataset

Dataset released by Clinc AI composed of popular personal assistant queries.

[Dataset paper](https://arxiv.org/pdf/1909.02027.pdf)
[Dataset repository](https://github.com/clinc/oos-eval)

## PolyAI Online Banking dataset

Dataset released by PolyAI composed of online banking queries.

[Dataset paper](TODO)
[Dataset repository](TODO)

## Dataset comparison

|         	        | Domain 	        | Example intents       | Train set size    | Test set size     | Number of intents |
| :---              | :---:	            | :---:	                | :---:	            | :---:	            | :---:	            |
| **Dataset**       | 	                | 	                    | 	                | 	                |                   |
| HWU               | Personal assistant| play_music            | 15000	            | 4500	            | 150               |
| Clinc             | Personal assistant| alarm_query           | 9960	            | 1076	            | 64                |
| Banking           | Online banking    | card_about_to_expire	| 10003	            | 3080	            | 77                |

## Download the benchmarking data

To download the benchmarked datasets run the following bash script:

```bash
cd <INTENT_REPO_ROOT>
export PYTHONPATH=.
export DATA_DIR=<PATH_TO_DATA_DIR>
sh data_utils/get_all_data.sh $DATA_DIR
```

This will create `<PATH_TO_DATA_DIR>` and 3 subdirectories with the train and test data of the datasets converted into csv format. It will also create the low data regime train splits.


# Models

## Fixed encoder based intent detectors

Classifiers trained on sentence representations obtained from pretrained sentence encoders. The sentence encoders are not finetuned while training the intent detection model.

run them with:

```bash
export OUTPUT_DIR=<PATH_TO_OUTPUT_DIR>
python run_classifier.py --data_dir $DATA_DIR \
  --params config.default \
  --output_dir=$OUTPUT_DIR \
  --params_overrides task=${DS},data_regime=${DR},encoder_type=${ENC}
```

* `$DS` is the dataset key, it can be `banking`, `clinc` or `hwu`
* `$DR` is the data regime, it can be `10`, `30` or `full`
* `$ENC` is the encoder type, it can be `convert`, `use`, `combined` or `bert`. See the following sections to know their meanings.

The default hyperparameters of the models can be changed in the `config.py` file, or can be overriden by the `--params_overrides` command line argument

### ConveRT-Intent

Classifier trained on top of [ConveRT](https://github.com/PolyAI-LDN/polyai-models/blob/master/README.md#convert) sentence encodings.

`$ENC` key: `convert`

### USE-Intent

Classifier trained on top of [USE Multilingual Large](https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/1) sentence encodings.

`$ENC` key: `use`

### Combined-Intent

Classifier trained on top of the concatenation of ConveRT and USE sentence encodings. This proved to outperform all the other intent detection models.

`$ENC` key: `combined`

### BERT-Intent fixed

Classifier trained on top of [BERT Large](https://tfhub.dev/google/bert_uncased_L-24_H-1024_A-16/1) sentence encodings. The encodings are obtained by averaging the sequence of token embeddings outputed by BERT

`$ENC` key: `bert`

## Finetuned encoder based intent detectors

Classifiers trained by finetuning the weigths of a pretrained sentence encoder.

### BERT-Intent finetuned

Classifier trained on top of the CLS token of [BERT Large](https://github.com/google-research/bert). This is included as a baseline and will require access to a TPU to be trained.

run it with:

```bash
export OUTPUT_DIR=<PATH_TO_OUTPUT_DIR>
python run_bert_finetuned_classifier.py --data_dir <INTENT_DATA_DIR> \
  --params config.bert_ft \
  --output_dir=$OUTPUT_DIR 
```



Note that to avoid unnecessary computations and catastrofic forgetting, each Task and Data Regime has its own hyperparameters for BERT finetuned. 
These hyperparameters are defined in `config.py` for each task and data regime. e.g. to run in banking with the "30" data regime, run:

```bash
python run_bert_finetuned_classifier.py --data_dir <INTENT_DATA_DIR> \
  --params config.bert_ft_tpu_banking_10 \
  --output_dir=$OUTPUT_DIR \
  --params_overrides use_tpu=false
```

These are the hyperparameters used for the benchmarks, and have been trained in a TPU on GCP. Note that some GPUs (<32gb memory) will run OOM when running this code. To set up the TPU trainig parameters on GCP, run:

```bash
export TPU_NAME=<TPU_NAME>
export TPU_ZONE=<TPU_ZONE>
export GCP_PROJECT=<GCP_PROJECT>
python run_bert_finetuned_classifier.py --data_dir <INTENT_DATA_DIR> \
  --params config.bert_ft \
  --output_dir=$OUTPUT_DIR \
  --params_overrides tpu_name=${TPU_NAME},tpu_zone=${TPU_ZONE},gcp_project=${GCP_PROJECT}
```
## Train your own intent detector

You can easily train your own intent detector and interact with the
 `run_classifier_interactive.py` script. This will train an intent detector 
 in any data you provide through a csv file and let you interact with it. 
 The csv must have the following format:

```csv
text, category
sentence1, intent1
sentence2, intent1
sentence3, intent2
sentence4, intent2
sentence5, intent2
sentence6, intent3
sentence7, intent3
sentence8, intent3
```

Then just run 

```bash
python run_classifier_interactive.py --train_file <INTENT_FILE_PATH> \
  --params config.default \
  --params_overrides encoder_type=${ENC}
```

And an intent detector will be trained and you will be able to interact with it.
You can change any hyperparameter either through the config.py file or through 
`--params_overrides`
## Citations

When using these models in your work, or the banking dataset please cite our paper, PAPER URL

```bibtex
@inproceedings{Casanueva2020,
    author      = {I{\~{n}}igo Casanueva and Tadas Temcinas and Matthew Henderson and Daniela Gerz and Ivan Vulic},
    title       = {TODO},
    year        = {2020},
    month       = {jul},
    note        = {Data available at TODO},
    url         = {TODO},
    booktitle   = {Arxiv},
}

```
