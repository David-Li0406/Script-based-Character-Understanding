export CUDA_VISIBLE_DEVICES=1

LEARNING_RATE_FIRST=1e-5
LEARNING_RATE_SECOND=2e-5
EPOCH_FIRST=31
EPOCH_SECOND=101

COREF_WEIGHT=1.0
LINKING_WEIGHT=1.0
IN_BATCH_WEIGHT=1.0
RECAP_WEIGHT=1.0

MODEL_SIZE=base


python train.py \
    --train \
    --trn_data=./data/revised_dataset/trn_SpanBERTspanbert-${MODEL_SIZE}-cased_batches_recap.json \
    --dev_data=./data/revised_dataset/dev_SpanBERTspanbert-${MODEL_SIZE}-cased_batches_recap.json \
    --tst_data=./data/revised_dataset/tst_SpanBERTspanbert-${MODEL_SIZE}-cased_batches_recap.json \
    --model=SpanBERT/spanbert-${MODEL_SIZE}-cased \
    --dev_keys=../data/data_set_keys_scene_dev.json  \
    --trn_keys=../data/data_set_keys_scene_trn.json \
    --tst_keys=../data/data_set_keys_scene_tst.json \
    --num_epochs=$EPOCH_FIRST \
    --learning_rate=$LEARNING_RATE_FIRST \
    --coref_weight=$COREF_WEIGHT \
    --linking_weight=$LINKING_WEIGHT \
    --in_batch_weight=$IN_BATCH_WEIGHT \
    --recap_weight=$RECAP_WEIGHT \
    --output_dir=trained_model/stage1/coref-${COREF_WEIGHT}_linking-${LINKING_WEIGHT}_lr-${LEARNING_RATE_FIRST}_epoch-${EPOCH_FIRST}_in_batch-${IN_BATCH_WEIGHT}_recap-${RECAP_WEIGHT}_size-${MODEL_SIZE}/ \
    & pid1=$!

wait $pid1

python train.py \
    --train \
    --trn_data=./data/revised_dataset/trn_SpanBERTspanbert-${MODEL_SIZE}-cased_batches_recap.json \
    --dev_data=./data/revised_dataset/dev_SpanBERTspanbert-${MODEL_SIZE}-cased_batches_recap.json \
    --tst_data=./data/revised_dataset/tst_SpanBERTspanbert-${MODEL_SIZE}-cased_batches_recap.json \
    --model=SpanBERT/spanbert-${MODEL_SIZE}-cased \
    --dev_keys=../data/data_set_keys_scene_dev.json  \
    --trn_keys=../data/data_set_keys_scene_trn.json \
    --tst_keys=../data/data_set_keys_scene_tst.json \
    --num_epochs=$EPOCH_SECOND \
    --learning_rate=$LEARNING_RATE_SECOND \
    --coref_weight=$COREF_WEIGHT \
    --linking_weight=$LINKING_WEIGHT \
    --output_dir=trained_model/stage2/coref-${COREF_WEIGHT}_linking-${LINKING_WEIGHT}_lr-${LEARNING_RATE_FIRST}_epoch-${EPOCH_FIRST}_in_batch-${IN_BATCH_WEIGHT}_recap-${RECAP_WEIGHT}_lr2-${LEARNING_RATE_SECOND}_epoch2-${EPOCH_SECOND}_size-${MODEL_SIZE}/ \
    --from_pretrained=trained_model/stage1/coref-${COREF_WEIGHT}_linking-${LINKING_WEIGHT}_lr-${LEARNING_RATE_FIRST}_epoch-${EPOCH_FIRST}_in_batch-${IN_BATCH_WEIGHT}_recap-${RECAP_WEIGHT}_size-${MODEL_SIZE}/pytorch_model.pt \
    & pid2=$!

wait $pid2

python train.py \
    --test \
    --trn_data=./data/revised_dataset/trn_SpanBERTspanbert-${MODEL_SIZE}-cased_batches_recap.json \
    --dev_data=./data/revised_dataset/dev_SpanBERTspanbert-${MODEL_SIZE}-cased_batches_recap.json \
    --tst_data=./data/revised_dataset/tst_SpanBERTspanbert-${MODEL_SIZE}-cased_batches_recap.json \
    --model=SpanBERT/spanbert-${MODEL_SIZE}-cased \
    --dev_keys=../data/data_set_keys_scene_dev.json  \
    --trn_keys=../data/data_set_keys_scene_trn.json \
    --tst_keys=../data/data_set_keys_scene_tst.json \
    --from_pretrained=trained_model/stage1/coref-${COREF_WEIGHT}_linking-${LINKING_WEIGHT}_lr-${LEARNING_RATE_FIRST}_epoch-${EPOCH_FIRST}_in_batch-${IN_BATCH_WEIGHT}_recap-${RECAP_WEIGHT}_size-${MODEL_SIZE}/pytorch_model.pt \
