# a demo of execute Tensorflow model fine-tuned based on bert—base-chinese using java
## export model
1. copy python/* to your bert directory
```
- bert
 - export_model.py
 - export_model.sh
 - ...
 - modeling.py
 - ...

```
2. modify export_model.sh
```shell
python export_model.py \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=./test_output/ \
  --output_dir=./test_output/ \
  --export_dir=./test_output/export/ \
  --num_labels=2 \
  --max_seq_length=250
```
3. the exported model can be find in $export_dir$
```
- $export_dir$
 - 1546929188
  - saved_model.pb
  - variables
   - variables.data-00000-of-00001
   - variables.index
```
## execute model in java
a demo can be find in `ModelTest`
```java
Preprocessor preprocessor = new Preprocessor();
preprocessor.initVocab();
Model model = new Model(
        "/Users/future/github/bert_java_serv/models/sample0",
        48,12, preprocessor
);
int[] labels = model.predictLabel(new String[]{"我丈夫是军人，我想离婚，怎么办？"});
assert labels[0] == 3;
```