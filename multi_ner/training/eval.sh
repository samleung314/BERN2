#!/bin/bash

export MODEL_NAME='./bern2-ner' # or 'dmis-lab/bern2-ner'
export OUTPUT_DIR='./output'    # Save an output file for evaluation results
export ENTITY="NCBI-disease"    # ("NCBI-disease" "BC2GM" "BC4CHEMD" "JNLPBA-cl" "JNLPBA-ct" "JNLPBA-dna" "JNLPBA-rna" "linnaeus")
export BATCH_SIZE=32
export SEED=1

python -m debugpy --listen localhost:5678 --wait-for-client run_eval.py \
  --model_name_or_path $MODEL_NAME \
  --data_dir NERdata/ \
  --labels NERdata/$ENTITY/labels.txt \
  --output_dir $OUTPUT_DIR \
  --eval_data_type $ENTITY \
  --eval_data_list $ENTITY \
  --max_seq_length 128 \
  --per_device_eval_batch_size $BATCH_SIZE \
  --seed $SEED \
  --do_eval \
  --do_predict
