#!/bin/bash

export DATA_LIST=NCBI-disease+BC4CHEMD+BC2GM+linnaeus+JNLPBA-dna+JNLPBA-rna+JNLPBA-ct+JNLPBA-cl
export EVAL_DATA_LIST=NCBI-disease+BC4CHEMD+BC2GM+linnaeus+JNLPBA-dna+JNLPBA-rna+JNLPBA-ct+JNLPBA-cl
export OUTPUT_DIR=./finetuned_model

export ENTITY=NCBI-disease # BC4CHEMD, BC2GM etc.
export NUM_EPOCHS=50
export BATCH_SIZE=32
export SAVE_STEPS=-1
export LOG_STEPS=1000
export SEED=1
export LR=3e-5
export WARMUP=0

python ./run_ner.py \
    --model_name_or_path RoBERTa-large-PM-M3-Voc-hf \
    --data_dir ./NERdata/ \
    --labels ./NERdata/$ENTITY/labels.txt \
    --output_dir $OUTPUT_DIR \
    --data_list $DATA_LIST \
    --eval_data_list $EVAL_DATA_LIST \
    --num_train_epochs $NUM_EPOCHS \
    --max_seq_length 128 \
    --warmup_steps $WARMUP \
    --learning_rate $LR \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --seed $SEED \
    --logging_steps $LOG_STEPS \
    --save_steps $SAVE_STEPS \
    --do_train \
    --do_eval \
    --do_predict \
    --overwrite_output_dir
    # --evaluate_during_training \
