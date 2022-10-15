export CUDA_VISIBLE_DEVICES=1

python -u main.py --model_type mental/mental-bert-base-uncased --bs 64 --lr 3e-4 --input_dir ../data/symp_data_w_control --patience 4 --bal_sample --control_ratio 0.5 --loss_mask --uncertain='exclude' --exp_name mbert_label_enhance_bal_sample_050_666 --write_result_dir ./lightning_logs/bal_sample_records.json

# {'test_loss': 0.02534688264131546,
#  'test_macro_acc': 0.9974651336669922,
#  'test_macro_auc': 0.9854118227958679,
#  'test_macro_f': 0.6702682971954346,
#  'test_macro_p': 0.668752908706665,
#  'test_macro_r': 0.6935033202171326,
#  'test_micro_acc': 0.9974464178085327,
#  'test_micro_auc': 0.9802484512329102,
#  'test_micro_f': 0.7046985626220703,
#  'test_micro_p': 0.6841658353805542,
#  'test_micro_r': 0.7265018820762634}