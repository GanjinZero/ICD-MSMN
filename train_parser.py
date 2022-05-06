import argparse


def generate_parser():
    parser = argparse.ArgumentParser()

    # input related
    parser.add_argument("--version", type=str, default="mimic3-50",
                        choices=["mimic2", "mimic3", "mimic3-50"], help="Dataset version.")
    parser.add_argument("--label_truncate_length", type=int, default=30)
    
    parser.add_argument("--n_gpu", type=int, default=1)
    
    # word encoder
    parser.add_argument("--word", action="store_true")
    parser.add_argument("--word_embedding_path", type=str)
    parser.add_argument("--word_dim", type=int, default=100)
    parser.add_argument("--word_dp", type=float, default=0.2)
    parser.add_argument("--word_frz", action="store_true")
    
    # combiner
    parser.add_argument("--combiner", type=str, default='lstm',
                    choices=['naive', 'lstm', 'reformer'])
    
    # lstm encoder
    parser.add_argument("--lstm_dp", type=float, default=0.1)
    parser.add_argument("--rnn_dim", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    
    # reformer
    parser.add_argument("--reformer_path", type=str, default="")
    parser.add_argument("--reformer_head", type=int, default=8)
    parser.add_argument("--n_hashes", type=int, default=8)
    parser.add_argument("--local_attention_head", type=int, default=2)
    parser.add_argument("--pkm_layers", type=str, default="")
    
    # rac
    # parser.add_argument("--num_cnn_layers", type=int, default=2)
    # parser.add_argument("--kernel_size", type=int, default=10)
    # parser.add_argument("--conv_dp", type=float, default=0.1)
    # parser.add_argument("--transformer_head", type=int, default=1)
    # parser.add_argument("--transformer_ff", type=int, default=1024)
    # parser.add_argument("--transformer_dp", type=float, default=0.1)
    # parser.add_argument("--transformer_activation", type=str, default='gelu')
    # parser.add_argument("--num_transformer_layers", type=int, default=4)
    
    # transformer
    parser.add_argument("--pos_embed", type=str, 
                        choices=["none", "learn", "axial", "tri"], default="none")
    parser.add_argument("--axial", type=str, default='(64,64)')
    parser.add_argument("--layer_norm", action="store_true")
    
    # label related
    parser.add_argument("--label_num_layers", type=int, default=0)
    parser.add_argument("--label_dropout", type=float, default=0.0)
    parser.add_argument("--label_pooling", type=str, default='max',
                        choices=['max', 'mean', 'last'])
    # parser.add_argument("--est_cls", action='store_true')
    parser.add_argument("--est_cls", type=int, default=0)
    
    parser.add_argument("--term_count", type=int, default=1)
    parser.add_argument("--sort_method", type=str, default='random',
                        choices=['max', 'mean', 'random'])
    
    # decoder related
    # parser.add_argument("--attention", type=str, choices=["caml", "laat", "average-word", "none", "logsumexp", "multi-head-laat", "multi-head-laat-v2", 
    #                                                       "double", "mil", 'relulaat'], default="caml")
    parser.add_argument("--decoder", type=str, choices=['MultiLabelMultiHeadLAATV2'])
    parser.add_argument("--attention_dim", type=int, default=512)
    parser.add_argument("--code_embedding_path", type=str)
    parser.add_argument("--rep_dropout", type=float, default=0.2)
    parser.add_argument("--att_dropout", type=float, default=0.0)
    parser.add_argument("--xavier", action="store_true")
    parser.add_argument("--attention_head", type=int, default=4) # For MultiHeadLAAT
    
    parser.add_argument("--head_pooling", type=str, default="max", choices=["max", "concat"])
    parser.add_argument("--act_fn_name", type=str, default="tanh", choices=['tanh', 'relu'])
    
    # Train setting
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--optimizer", type=str, default="AdamW",
                        choices=["AdamW", "SGD", "Adam"])
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--train_epoch", type=int, default=20)
    parser.add_argument("--early_stop_epoch", type=int, default=-1)
    parser.add_argument("--early_stop_metric", type=str, default="prec_at_8")
    # parser.add_argument("--early_stop_strategy", type=str, choices=["best_dev", "last_dev"], default="best_dev")
    parser.add_argument("--truncate_length", type=int, default=4000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--output_base_dir", type=str, default="./output/")
    parser.add_argument("--prob_threshold", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument("--scheduler", type=str, default="linear",
                        choices=['linear', 'constant', 'cosine'])

    # loss
    parser.add_argument("--loss_name", type=str, default="ce",
                        choices=['ce', 'focal', 'asy', 'kexue', 'ldam', 'maskce', 'hungarian'])
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--focal_alpha", type=float, default=0.25)
    parser.add_argument("--asy_gamma_neg", type=float, default=4.0)
    parser.add_argument("--asy_gamma_pos", type=float, default=1.0)
    parser.add_argument("--asy_clip", type=float, default=0.05)
    parser.add_argument("--able_torch_grad_focal_loss", action="store_true")
    parser.add_argument("--ldam_c", type=float, default=3.0)
    
    parser.add_argument("--rdrop_alpha", type=float, default=0.0)
   
    parser.add_argument("--main_code_loss_weight", type=float, default=1.0)
    parser.add_argument("--code_loss_weight", type=float, default=1.0)


    
    # knowledge_distill
    parser.add_argument("--knowledge_distill", action="store_true") # not avaliable
    
    # # others
    parser.add_argument("--adv_training", default=None, choices=['fgm', 'pgd'])
    
    parser.add_argument("--tag", type=str, default="")

    return parser
