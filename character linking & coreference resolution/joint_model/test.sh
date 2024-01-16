python train.py \
    --test \
    --trn_data=./data/revised_dataset/trn_SpanBERTspanbert-large-cased_batches_recap.json \
    --dev_data=./data/revised_dataset/dev_SpanBERTspanbert-large-cased_batches_recap.json \
    --tst_data=./data/revised_dataset/tst_SpanBERTspanbert-large-cased_batches_recap.json \
    --model=SpanBERT/spanbert-large-cased \
    --dev_keys=../data/data_set_keys_scene_dev.json  \
    --trn_keys=../data/data_set_keys_scene_trn.json \
    --tst_keys=../data/data_set_keys_scene_tst.json \
    --from_pretrained=trained_model/pytorch_model.pt \