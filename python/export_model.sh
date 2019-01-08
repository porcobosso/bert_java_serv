python export_model.py \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=./test_output/ \
  --output_dir=./test_output/ \
  --export_dir=./test_output/export/ \
  --num_labels=2 \
  --max_seq_length=250
