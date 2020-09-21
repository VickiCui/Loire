## Installation

- Setup a conda environment and install some prerequisite packages

```bash
conda create -n vibert python=3.6  # Create a virtual environment
source activate vibert             # Activate virtual environment

pip install -r  requirements.txt

git clone https://github.com/cocodataset/cocoapi.git  # Install pycocotools
cd cocoapi/PythonAPI
python setup.py build_ext install

python -m nltk.downloader all           # Install NLTK data
```



## Scene Layout Generator 

- Download the COCO datasets to train layout generator

```bash
cd layout_generator
./experiments/scripts/fetch_data.sh
cd ..
```

- train layout generator

```shell
cd layout_generator
python -u ./tools/train.py 
--cuda --batch_size=8 --data_dir=./data/coco --log_dir=./coco
--exp_name=coco --attn_emb=True --output_cls_size=83
--finetune_lr=5e-5 --max_input_length=64 --max_output_length=20
--num_workers=1 --accumulation_steps=4 --n_epochs=15 --warmup=0.05 
```

the checkpoint of ViBERT will be saved at $log_dir

## Commonsense Reasoning

### Data

- Download the Commonsense & WinoGrande datasets

```bash
cd reasoning
./download_data.sh
cd ..
```

### Train

- do commonsense reasoning on CommonsenseQA with BERT+ViBERT

```Shell
cd reasoning
CUDA_VISIBLE_DEVICES=0,1 python -u train.py --cuda --num_workers=1 --parallel\
    --epochs=5 --model_name=Loire --do_train\
    --task=commonsenseqa\
    --data_dir=$PATH_TO_DATA\
    --model=bert-base --max_seq_length=64 --seed=1\
    --batch_size=16 --accumulation_steps=1 --log_per_steps=500\
    --warmup=0.1 --lr=1e-5 --adam_betas='(0.9, 0.98)' --adam_eps=1e-06\
    --dropout=0.3 --weight_decay=0.01\
    --feature --norm --proj=f2t prefix\
    --pretrained_bert=$PATH_TO_VIBERT\
    --save_ckpt
```

- do commonsense reasoning on WinoGrande with BERT+ViBERT

```Shell
cd reasoning
CUDA_VISIBLE_DEVICES=0,1 python -u train.py --cuda --num_workers=1 --parallel\
    --epochs=10 --model_name=Loire --do_train\
    --task=winogrande --train_size=xl\
    --data_dir=$PATH_TO_DATA\
    --model=bert-base --max_seq_length=64 --seed=1\
    --batch_size=16 --accumulation_steps=1 --log_per_steps=500\
    --warmup=0.1 --lr=1e-5 --adam_betas='(0.9, 0.98)' --adam_eps=1e-06\
    --dropout=0.3 --weight_decay=0.01\
    --feature --norm --proj=f2t\
    --pretrained_bert=$PATH_TO_VIBERT\
    --save_ckpt
```

For training on different sizes of training set, set `--size={xs,s,m,l,xl}'`accordingly.

For commonsense reasoning with RoBERTa+ViBERT, set `--model=roberta-base` or `--model=roberta-large`, and change other parameters, such as `--max_seq_length`, `--batch_size`, accordingly.

The model will be trained using single GPU by default. To use multiple GPUs please set the `--parallel` flag and modify the batch size using the `--batch_size` flag accordingly.

The model will add a LayerNorm after Linear layer by default. To not add a LayerNorm, please remove the `--norm` flag.

Checkpoint files can be found in `./logs/commonsenseqa` or `./logs/winogrande`

### Test

- test on CommonsenseQA with BERT+ViBERT

```shell
python -u train.py 
--cuda --model_name=vibert --do_pred --feature --norm
--max_seq_length=64 --model=bert --task=commonsense_qa
--data_dir=./data/commonsenseQA
--pretrained_bert=$YOUR_PATH_SAVED_VIBERT
--pred_name=$OUTPUT_FILENAME
--pretrained=$THE_CHECKPOINT_TO_BE_EVALUATED
```

For test on WinoGrande, set `--task=winogrande` and `--data_dir=./data/WinoGrande`.

For test RoBERTa+ViBERT, set `--model=roberta`.

Parameters `--feature, --norm, --max_seq_length, --model, pretrained_bert` should be consistent with the parameters set during training

