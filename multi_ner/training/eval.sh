export MODEL_NAME=./finetuned_model # or 'dmis-lab/bern2-ner'
export OUTPUT_DIR=./output # Save an output file for evaluation results
export ENTITY=NCBI-disease
export BATCH_SIZE=32
export SEED=1

python run_eval.py
    --model_name_or_path $MODEL_NAME
    --data_dir NERdata/
    --labels NERdata/$ENTITY/labels.txt
    --output_dir $OUTPUT_DIR
    --eval_data_type $ENTITY
    --eval_data_list $ENTITY
    --max_seq_length 128
    --per_device_eval_batch_size $BATCH_SIZE
    --seed $SEED
    --do_eval
    --do_predict
