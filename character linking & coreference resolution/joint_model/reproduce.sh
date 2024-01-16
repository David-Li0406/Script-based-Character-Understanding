LEARNING_RATE=3e-5
EPOCH=101

COREF_WEIGHT=0.0
LINKING_WEIGHT=1.0
MODEL_SIZE=base
MODEL=bert

python train.py \
    --train \
    --trn_data=./data/revised_dataset/trn_${MODEL}-${MODEL_SIZE}-cased_batches_recap.json \
    --dev_data=./data/revised_dataset/dev_${MODEL}-${MODEL_SIZE}-cased_batches_recap.json \
    --tst_data=./data/revised_dataset/tst_${MODEL}-${MODEL_SIZE}-cased_batches_recap.json \
    --model=${MODEL}-${MODEL_SIZE}-cased \
    --dev_keys=../data/data_set_keys_scene_dev.json  \
    --trn_keys=../data/data_set_keys_scene_trn.json \
    --tst_keys=../data/data_set_keys_scene_tst.json \
    --num_epochs=$EPOCH \
    --learning_rate=$LEARNING_RATE \
    --coref_weight=$COREF_WEIGHT \
    --linking_weight=$LINKING_WEIGHT \
    --output_dir=trained_model/reproduce/coref-${COREF_WEIGHT}_linking-${LINKING_WEIGHT}_lr-${LEARNING_RATE}_epoch-${EPOCH}_model_${MODEL}_size-${MODEL_SIZE}/ \
    & pid1=$!

wait $pid1

python train.py \
    --test \
    --trn_data=./data/revised_dataset/trn_${MODEL}-${MODEL_SIZE}-cased_batches_recap.json \
    --dev_data=./data/revised_dataset/dev_${MODEL}-${MODEL_SIZE}-cased_batches_recap.json \
    --tst_data=./data/revised_dataset/tst_${MODEL}-${MODEL_SIZE}-cased_batches_recap.json \
    --model=${MODEL}-${MODEL_SIZE}-cased \
    --dev_keys=../data/data_set_keys_scene_dev.json  \
    --trn_keys=../data/data_set_keys_scene_trn.json \
    --tst_keys=../data/data_set_keys_scene_tst.json \
    --from_pretrained=trained_model/reproduce/coref-${COREF_WEIGHT}_linking-${LINKING_WEIGHT}_lr-${LEARNING_RATE}_epoch-${EPOCH}_model_${MODEL}_size-${MODEL_SIZE}/pytorch_model.pt \