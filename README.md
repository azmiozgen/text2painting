# text2painting

## Convert keywords into painting

![definition](./assets/simple_task_definition.jpg "definition")

Sample results (input words - stage1 output - stage2 output - stage3 output - ground truth)

![results](./assets/results.jpg "results")

## Model

Model is built as sequential GANs with three stages

![basic](./assets/basic_model.jpg "basic")

## Stage-1 structure

![stage1](./assets/gan1.jpg "stage1")

## Stage-2 structure

![stage2](./assets/gan2.jpg "stage2")

## Stage-3 structure

![stage2](./assets/gan3.jpg "stage2")

## Installation

Set environment with

`conda env create -f environment.yml`

Install Spacy English language model

`python -m spacy download en`

## Data

A small sample data can be downloaded from

https://drive.google.com/open?id=1KXEIrRGDvASEEm-vT_jJcvFLncapJDEL

Place them under `data/deviantart_verified/images` and `data/wikiart_verified/images`
