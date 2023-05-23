# PsySym

The code for the EMNLP 2022 paper [Symptom Identification for Interpretable Detection of Multiple Mental Disorders](https://arxiv.org/abs/2205.11308).

Dataset can be provided upon request, please contact blmoistawinde@qq.com or chensiyuan925@sjtu.edu.cn 

## Directory Organization

- data/ (post data can be provided upon request)
    - desc_from_post/ : descriptions used to retrieve candidate posts for some symptoms
    - symp_data/ : the annotated sentences in PsySym
        - train/test/val.csv : the split with multiple diseases combined
        - other folders contain the split for each disease. (Some classes are removed from the single disease dataset if the samples are too few)
    - symp_data_w_control/ : combined set of annotated and control sentences
    - symptom_kg.owl : the KG of PsySym
    - parsed_kg_info.json : main information of the KG in JSON format
    - raw_annos.csv : the raw annotation results, contains the annotation results from different annotators (distinguished with the `round` column), and the original symptom-level status annotations
- relevance_model/ : code for the relevance judgment model
    - use `bal_sample_050.sh` to train the best performing model on `symp_data_w_control/` with proposed balanced sampler
    - `infer_smhd_feats.py` is used to infer symptom features for the disease detection model
- status_model/ : code for the status inference model
    - use `train.sh` to train the best performing model with proposed balanced sampler
    - `infer_smhd_feats.py` is used to infer symptom features for the disease detection model
- disease_model/ : code for the disease detection model
    - check `train.sh` for running these models
        - you may need to infer features with the previous models first


## Citation

If this repository helps you, please cite this paper:

```bibtex
@article{zhang2022symptom,
  title={Symptom Identification for Interpretable Detection of Multiple Mental Disorders},
  author={Zhang, Zhiling and Chen, Siyuan and Wu, Mengyue and Zhu, Kenny Q},
  journal={arXiv preprint arXiv:2205.11308},
  year={2022}
}
```
