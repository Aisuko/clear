# Dataset

> Note: This dataset is provided to facilitate code execution and result replication; it is not intended for data sharing. I have really bad experience on pre-processing the datset.

The pre-processed dataset:

|Dataset|Size|URL|Location|
|---|---|---|---|
|aisuko/in-hospital-motality-6-48-contrast-learning|173MB|https://huggingface.co/datasets/aisuko/in-hospital-motality-6-48-contrast-learning|root path of project|
|aisuko/mimic3-pre-training-datasets|https://huggingface.co/datasets/aisuko/mimic3-pre-training-datasets|root path of project|
|aisuko/mimic3_benchmark1-4|https://huggingface.co/datasets/aisuko/mimic3_benchmark1-4|mimic3-benchmark pipeline execution result of step 1-4|
|aisuko/multimodal-mimic3-pre-training-raw-data|https://huggingface.co/datasets/aisuko/multimodal-mimic3-pre-training-raw-data|multimodal-mimic/mimi3-benchmarks/|


## MIMIC-III

The dataset used for this paper is MIMIC-III. The data can be downloaded here [https://physionet.org/content/mimiciii/1.4/](https://physionet.org/content/mimiciii/1.4/). **NOTE**: To gain access to this dataset, you will need to complete the required training. 


### MIMIC-III Benchmark

Once you've downloaded the MIMIC-III dataset, you will need to build the MIMIC-III Benchmark from [https://github.com/YerevaNN/mimic3-benchmarks](https://github.com/YerevaNN/mimic3-benchmarks). We used a modified version of this code so that we can index patient IDs and read in and out times without opening files. Replace:

```
mimic3-benchmarks/mimic3benchmark/scripts/extract_episodes_from_subjects.py
mimic3-benchmarks/mimic3benchmark/scripts/create_decompensation.py
mimic3-benchmarks/mimic3benchmark/scripts/create_in_hospital_mortality.py
mimic3-benchmarks/mimic3benchmark/scripts/create_length_of_stay.py
mimic3-benchmarks/mimic3benchmark/scripts/create_phenotyping.py
mimic3-benchmarks/mimic3benchmark/scripts/create_multitask.py
```

with:

```
multimodal-medical-pretraining/mimic3benchmark/extract_episodes_from_subjects.py
multimodal-medical-pretraining/mimic3benchmark/create_decompensation.py
multimodal-medical-pretraining/mimic3benchmark/create_in_hospital_mortality.py
multimodal-medical-pretraining/mimic3benchmark/create_length_of_stay.py
multimodal-medical-pretraining/mimic3benchmark/create_phenotyping.py
multimodal-medical-pretraining/mimic3benchmark/create_multitask.py
```


Here is the repo [https://github.com/Aisuko/mimic3-benchmark](https://github.com/Aisuko/mimic3-benchmark) you can do it directly.


Check the project structure section at the [README](../README.md)