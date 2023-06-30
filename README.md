# SpindleDetection
SpindleDetection

### Paper: SpindleU-Net: An Adaptive U-Net Framework for Sleep Spindle Detection in Single-Channel EEG

### Paper link: [https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9514837](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9514837)


# Content
* ### [Setup](#Setup)
* ### [Dataset Preparation](#Preparation)
* ### [Training](#Training)

## <div id='Setup'>Setup</div>

First, clone the repo and install required packages:
```
git clone https://github.com/LordXX505/SpindleDetection.git
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
conda activate SpindleDetection
cd SpindleDetection
mkdir ../data/SS2/SS2_bio
mkdir ../data/SS2/SS2_ana
Put PSG recordings to ../data/SS2/SS2_bio and annotations to ../data/SS2/SS2_ana.
sh data.sh
cd utils
python3 augment.py
```


## <div id='Training'>Training</div>
Using slurm:
```
sh start.sh
```
But carefully, you may manually change the Master addr in ./utils/misc the init_distributed_mode function.

And if you want the loss function is CrossEntropy:
```
sh start.sh
```

