export CUDA_VISIBLE_DEVICES=2

python -u main.py --model_type mental/mental-bert-base-uncased --bs 64 --lr 3e-4 --input_dir ../data/symp_data --patience 4 --loss_mask --uncertain='only' --exp_name mbert_uncertain_only_666 --write_result_dir ./lightning_logs/baseline_records.json

# {'test_loss': 0.44582399725914, 'test_uncertain_mae': 0.13599330186843872}