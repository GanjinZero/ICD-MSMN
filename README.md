# ICD-MSMN
The offical implementation of "Code Synonyms Do Matter: Multiple Synonyms Matching Network for Automatic ICD Coding" [ACL 2022]

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
CUDA_VISIBLE_DEVICES=0 python main.py --n_gpu 1 --version mimic3 --combiner lstm --rnn_dim 256 --num_layers 2 --decoder MultiLabelMultiHeadLAATV2 --attention_head 4 --attention_dim 512 --learning_rate 5e-4 --train_epoch 20 --batch_size 2 --gradient_accumulation_steps 8 --xavier --main_code_loss_weight 0.0 --rdrop_alpha 5.0 --est_cls 1  --term_count 4  --sort_method random --word_embedding_path word_embedding_path
```

MIMIC-III Full (8 GPUs):
```
NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 --master_port=1212 --use_env  main.py --n_gpu 8 --version mimic3 --combiner lstm --rnn_dim 256 --num_layers 2 --decoder MultiLabelMultiHeadLAATV2 --attention_head 4 --attention_dim 512 --learning_rate 5e-4 --train_epoch 20 --batch_size 2 --gradient_accumulation_steps 1 --xavier --main_code_loss_weight 0.0 --rdrop_alpha 5.0 --est_cls 1  --term_count 4  --sort_method random --word_embedding_path word_embedding_path
```

MIMIC-III 50:
```
CUDA_VISIBLE_DEVICES=0 python main.py --version mimic3-50 --combiner lstm --rnn_dim 512 --num_layers 1 --decoder MultiLabelMultiHeadLAATV2 --attention_head 8 --attention_dim 512 --learning_rate 5e-4 --train_epoch 20 --batch_size 16 --gradient_accumulation_steps 1 --xavier --main_code_loss_weight 0.0 --rdrop_alpha 5.0 --est_cls 1 --term_count 8 --word_embedding_path word_embedding_path
```

# Evaluate checkpoints
```
python eval_model.py MODEL_CHECKPOINT
```
[mimic3 checkpoint](https://drive.google.com/file/d/1Ru9AM3FJuBVWSvPDUV13vWhWDzFw_qAD/view?usp=sharing)

[mimic3-50 checkpoint](https://drive.google.com/file/d/18Ny2R9WLWWa2UpyReaBn-zoSb1Uga9yX/view?usp=sharing)

# Citation
```
@inproceedings{yuan-etal-2022-code,
    title = "Code Synonyms Do Matter: Multiple Synonyms Matching Network for Automatic {ICD} Coding",
    author = "Yuan, Zheng  and
      Tan, Chuanqi  and
      Huang, Songfang",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-short.91",
    pages = "808--814",
    abstract = "Automatic ICD coding is defined as assigning disease codes to electronic medical records (EMRs).Existing methods usually apply label attention with code representations to match related text snippets.Unlike these works that model the label with the code hierarchy or description, we argue that the code synonyms can provide more comprehensive knowledge based on the observation that the code expressions in EMRs vary from their descriptions in ICD. By aligning codes to concepts in UMLS, we collect synonyms of every code. Then, we propose a multiple synonyms matching network to leverage synonyms for better code representation learning, and finally help the code classification. Experiments on the MIMIC-III dataset show that our proposed method outperforms previous state-of-the-art methods.",
}
```
