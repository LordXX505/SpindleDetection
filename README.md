# SpindleDetection
SpindleDetection

### Paper: SpindleU-Net: An Adaptive U-Net Framework for Sleep Spindle Detection in Single-Channel EEG

### Paper link: [https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9514837](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9514837)


# Content
* ### [Setup](#Setup)
* ### [Dataset Preparation](#Preparation)
* ### [Training]

## <div id='Setup'>Setup</div>

First, clone the repo and install required packages:
```
git clone https://github.com/microsoft/unilm.git
cd SpindleDetection

conda create -n SpindleDetection
conda activate SpindleDetection
pip install -r requirements.txt
```

## <div id='Preparation'>Dataset Preparation</div>
We use the Mass SS2 datasets. 

You can download from [https://doi.org/10.5683/SP3/Y889CS](https://doi.org/10.5683/SP3/Y889CS).This is PSG recordings.

And SS2 Sleep Annotations:[https://borealisdata.ca/dataset.xhtml?persistentId=doi:10.5683/SP3/Y889CS](https://borealisdata.ca/dataset.xhtml?persistentId=doi:10.5683/SP3/Y889CS).

```
git clone https://github.com/microsoft/unilm.git
cd unilm/vlmo
pip install -r requirements.txt
```
