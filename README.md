# Timeliness
The source code of the paper "Learning and Deducing Temporal Orders"

## Datasets

Please download the zip file from the following link and decompress it in the root directory.

https://drive.google.com/drive/folders/1B7z0FFOSFfqOvgP3r78exvYZey9G1RZW?usp=share_link


The preview of the project structure is as follow.

```
.
├── data
│   ├── career
│   ├── comm_
│   ├── nba
│   ├── person
└── GATE
    ├── baselines
    ├── creator
    ├── critic
    ├── discovery
    ├── gate.py
    ├── utility.py
    ├── gate.sh
    ├── main.py
    ├── metrics.py
    ├── pretrain
    ├── proc
    ├── requirements.txt
    ├── result
    ├── results
    └── shell
```
## Install packages
```
pip3 install -r requirements.txt
```


## Run the code
```
cd GATE
python main.py --creator Gate --data ${data_path} --lr ${lr} --batch_size ${batch_size} --high_conf_sample_ratio ${conf_sample_size} --conf_threshold ${conf_threshold} --variant gate --gpuOption ${gpu}
```
## Example
```
python3 main.py --creator Gate --data /home/rsltgy/Desktop/GATE/GATE/data/person/ --lr 1e-4 --batch_size 8 --high_conf_sample_ratio 0.52 --epoch 5
```

Here the arguments are described as follow

- **data_path** is the path of the original data (*.csv file)
- **lr** is the learning rate
- **batch_size** is the batch size
- **conf_sample_size** is the sample ratio of temporal orders to be predicted by the Creator
- **conf_threshold** is the threshold of confidence
- **variant** is the variant option: gate, creator, critic, creatornc, creatorne, creatorna, gatenc and creatoritr
- **gpu** is the gpu cuda option


## Run the settings
```
cd GATE
mkdir result
```


main.py is the entry and gate.py is the primary code of the Timeliness.

To run the code or evaluate the experiments in the submitted paper, go to "shell" folder that stores all scripts of Figure 6(a)-(t). 

E.g., for Figure 6(a), simply run the following script
```
./fig_a.sh ${gpu_id}
```
where gpu_id is the cuda gpu id. 

The experimental results are saved in the "result" folder with different filenames.
