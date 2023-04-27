# Unsup
# SST-5
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

# Few-shot
# SST-5
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


