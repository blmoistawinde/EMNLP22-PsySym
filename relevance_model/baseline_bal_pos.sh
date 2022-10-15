export CUDA_VISIBLE_DEVICES=2

python -u main.py --model_type mental/mental-bert-base-uncased --bs 64 --lr 3e-4 --input_dir ../data/symp_data_w_control --patience 4 --loss_mask --pos_weight_setting 'balance' --uncertain='exclude' --exp_name mbert_label_enhance_bal_pos_666 --write_result_dir ./lightning_logs/baseline_records.json

# {'test_loss': 3.126127004623413,
#  'test_macro_acc': 0.8391344547271729,
#  'test_macro_auc': 0.6269316673278809,
#  'test_macro_f': 0.0011961734853684902,
#  'test_macro_p': 0.0006005625473335385,
#  'test_macro_r': 0.15789473056793213,
#  'test_micro_acc': 0.8380345702171326,
#  'test_micro_auc': 0.5364418029785156,
#  'test_micro_f': 0.007400503382086754,
#  'test_micro_p': 0.0037978659383952618,
#  'test_micro_r': 0.14396536350250244}