export CUDA_VISIBLE_DEVICES=2

python -u main.py --model_type mental/mental-bert-base-uncased --bs 64 --lr 3e-4 --input_dir ../data/symp_data_w_control --patience 4 --loss_mask --uncertain='exclude' --exp_name mbert_label_enhance_666 --write_result_dir ./lightning_logs/baseline_records.json

# {'test_loss': 0.06668533384799957,
#  'test_macro_acc': 0.9958280920982361,
#  'test_macro_auc': 0.5481703281402588,
#  'test_macro_f': 0.0,
#  'test_macro_p': 0.0,
#  'test_macro_r': 0.0,
#  'test_micro_acc': 0.9958060383796692,
#  'test_micro_auc': 0.6309939026832581,
#  'test_micro_f': 0.0,
#  'test_micro_p': 0.0,
#  'test_micro_r': 0.0}