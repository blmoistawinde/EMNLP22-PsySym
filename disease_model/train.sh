export CUDA_VISIBLE_DEVICES=3

for sel_disease in depression anxiety autism adhd schizophrenia bipolar ocd ptsd eating
do
    # symptom features only
    python -u main.py --input_dir "./symp_dataset_tiny" --bs 64 --feat_type prob --sel_disease ${sel_disease} --exp_name tiny_${sel_disease}_concat --lr 0.01 --write_result_dir ./lightning_logs/records_concat_666.json

    # including status and subject feature
    # python -u main.py --input_dir "./symp_dataset_tiny" --bs 64 --feat_type prob --sel_disease ${sel_disease} --concat_feats --exp_name tiny_${sel_disease}_concat --lr 0.01 --write_result_dir ./lightning_logs/records_concat_666.json

    # bert feature
    # python -u main.py --input_dir "./dataset_org_bert_avg" --bs 64 --feat_type emb --sel_disease ${sel_disease} --concat_feats --exp_name bert_avg_${sel_disease}_emb_concat --lr 0.003 --write_result_dir ./lightning_logs/records_org_bert_avg_concat.json
done