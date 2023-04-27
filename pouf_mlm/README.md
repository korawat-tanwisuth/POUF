* [Introduction](#introduction)
* [Instructions](#instruction)
* [Citation](#citation)


## Introduction

## Instruction

1. Install the requirements with the following command:

```
pip install -r requirements.txt
```

2. Prepare the data with the following commands:

```bash
cd data
bash download_dataset.sh
```

Then use the following command (in the root directory) to generate the data:

```bash
python tools/generate_k_shot_data.py
```
3. Run the demo on the SST-5 experiments with the following commands:

   - Unsupervised
     ```bash
     python run.py \
        --task_name sst-5 \
        --data_dir data/k-shot/sst-5/16-42 \
        --overwrite_output_dir \
        --do_train \
        --do_eval \
        --do_predict \
        --evaluate_during_training \
        --model_name_or_path roberta-large \
        --few_shot_type prompt \
        --num_k 16 \
        --max_steps 1000 \
        --eval_steps 100 \
        --per_device_train_batch_size 2 \
        --learning_rate 1e-5 \
        --num_train_epochs 0 \
        --output_dir result_unsup_kshot/tmp \
        --seed 42 \
        --template "*cls**sent_0*_It_was*mask*.*sep+*" \
        --mapping "{0:'terrible',1:'bad',2:'okay',3:'good',4:'great'}" \
        --first_sent_limit 110 \
        --other_sent_limit 20 \
        --unsup
     ```   
   - Few-shot
    ```bash
    python run.py \
        --task_name sst-5 \
        --data_dir data/k-shot/sst-5/16-42 \
        --overwrite_output_dir \
        --do_train \
        --do_eval \
        --do_predict \
        --evaluate_during_training \
        --model_name_or_path roberta-large \
        --few_shot_type prompt-demo \
        --num_k 16 \
        --max_steps 1000 \
        --eval_steps 100 \
        --per_device_train_batch_size 2 \
        --learning_rate 1e-5 \
        --num_train_epochs 0 \
        --output_dir result/tmp \
        --seed 42 \
        --template "*cls**sent_0*_It_was*mask*.*sep+*" \
        --mapping "{0:'terrible',1:'bad',2:'okay',3:'good',4:'great'}" \
        --first_sent_limit 110 \
        --other_sent_limit 20 \
        --double_demo
    ```

## Citation

We adapt our code using the following codebase:

**LM-BFF**
```bibtex
@inproceedings{gao2021making,
   title={Making Pre-trained Language Models Better Few-shot Learners},
   author={Gao, Tianyu and Fisch, Adam and Chen, Danqi},
   booktitle={Association for Computational Linguistics (ACL)},
   year={2021}
}
```
