# ICD-MSMN
The offical implementation of "Code Synonyms Do Matter: Multiple Synonyms Matching Network for Automatic ICD Coding" [ACL 2022]
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/code-synonyms-do-matter-multiple-synonyms-1/medical-code-prediction-on-mimic-iii)](https://paperswithcode.com/sota/medical-code-prediction-on-mimic-iii?p=code-synonyms-do-matter-multiple-synonyms-1)

# Environment
All codes are tested under Python 3.7, PyTorch 1.7.0.
Need to install opt_einsum for einsum calculations.
At least 32GB GPU are needed for training MIMIC-III full setting.

# Dataset
We only put several samples for each dataset.
One need to obtain licences to download MIMIC-III dataset.
Once you obtain the MIMIC-III dataset, please follow [caml-mimic](https://github.com/jamesmullenbach/caml-mimic) to preprocess the dataset.
You should obtain **train_full.csv**, **test_full.csv**, **dev_full.csv**, **train_50.csv**, **test_50.csv**, **dev_50.csv** after preprocessing.
Please put them under **sample_data/mimic3**.
Then you should use **preprocess/generate_data_new.ipynb** for generating json format dataset.

# Word embedding
Please download [word2vec_sg0_100.model](https://github.com/aehrc/LAAT/blob/master/data/embeddings/word2vec_sg0_100.model) from LAAT.
You need to change the path of word embedding.

# Use our code
MIMIC-III Full (1 GPU):
```
CUDA_VISIBLE_DEVICES=0 python main.py --n_gpu 8 --version mimic3 --combiner lstm --rnn_dim 256 --num_layers 2 --decoder MultiLabelMultiHeadLAATV2 --attention_head 4 --attention_dim 512 --learning_rate 5e-4 --train_epoch 20 --batch_size 2 --gradient_accumulation_steps 8 --xavier --main_code_loss_weight 0.0 --rdrop_alpha 5.0 --est_cls 1  --term_count 4  --sort_method random --word_embedding_path word_embedding_path
```

MIMIC-III Full (8 GPUs):
```
NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 --master_port=1212 --use_env  main.py --n_gpu 8 --version mimic3 --combiner lstm --rnn_dim 256 --num_layers 2 --decoder MultiLabelMultiHeadLAATV2 --attention_head 4 --attention_dim 512 --learning_rate 5e-4 --train_epoch 20 --batch_size 2 --gradient_accumulation_steps 1 --xavier --main_code_loss_weight 0.0 --rdrop_alpha 5.0 --est_cls 1  --term_count 4  --sort_method random --word_embedding_path word_embedding_path
```

MIMIC-III 50:
```
CUDA_VISIBLE_DEVICES=0 python main.py --version mimic3-50 --combiner lstm --rnn_dim 512 --num_layers 1 --decoder MultiLabelMultiHeadLAATV2 --attention_head 8 --attention_dim 512 --learning_rate 5e-4 --train_epoch 20 --batch_size 16 --gradient_accumulation_steps 1 --xavier --main_code_loss_weight 0.0 --rdrop_alpha 5.0 --est_cls 1 --term_count 8 --word_embedding_path word_embedding_path
```

# Citation
```
@article{yuan2022code,
  title={Code Synonyms Do Matter: Multiple Synonyms Matching Network for Automatic ICD Coding},
  author={Yuan, Zheng and Tan, Chuanqi and Huang, Songfang},
  journal={arXiv preprint arXiv:2203.01515},
  year={2022}
}
```