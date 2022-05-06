import os
from transformers import AutoModel, AutoTokenizer
from gensim.models import KeyedVectors
from sklearn.decomposition import PCA
import torch
import tqdm
import numpy as np
import sys
sys.path.append("..")
from data_util import load_code_descriptions
import numpy as np

batch_size = 32

def coder_init(model_name_or_path, ind2c, dim, device, version):
    model_suffix = model_name_or_path.split("/")[-1]
    if not model_suffix:
        model_suffix = model_name_or_path.split("/")[-2]
    output_path = os.path.join('preprocess', model_suffix + f"_len{len(ind2c)}" + f"_dim{dim}.txt")
    # print(output_path)
    if os.path.exists(output_path):
        return load_code_embedding(output_path)
    model = AutoModel.from_pretrained(model_name_or_path).to(device)
    tok = AutoTokenizer.from_pretrained(model_name_or_path)
    phrase_list = []
    desc_dict = load_code_descriptions(version)
    for i in range(len(ind2c)):
        phrase_list.append(desc_dict[ind2c[i]])
    print(phrase_list[0:10])
    code_bert_embed = get_bert_embed(phrase_list, model, tok, device, tqdm_bar=True).detach().cpu().numpy()

    # PCA for code_bert_embed
    if dim <= len(ind2c):
        pca = PCA(n_components=dim)
        code_pca_embed = pca.fit_transform(code_bert_embed)
    elif dim > len(ind2c) and dim <= code_bert_embed.shape[1]:
        # Select dims with largest variance for simplicity
        var = np.var(code_bert_embed, axis=0)
        threshold = np.sort(var)[-dim]
        code_pca_embed = code_bert_embed[:,np.where(var >= threshold)[0]]
    else:
        normal = np.random.randn(code_bert_embed.shape[1], dim)
        code_pca_embed = np.dot(code_bert_embed, normal)
    with open(output_path, "w", encoding="utf-8") as f:
        for i in range(len(ind2c)):
            f.write(" ".join([ind2c[i]] + [str(num) for num in code_pca_embed[i].tolist()]) + "\n")

    del model
    return load_code_embedding(output_path)


def get_bert_embed(phrase_list, m, tok, device, normalize=True, summary_method="CLS", tqdm_bar=False):
    input_ids = []
    for phrase in phrase_list:
        input_ids.append(tok.encode_plus(
            phrase, max_length=32, add_special_tokens=True,
            truncation=True, pad_to_max_length=True)['input_ids'])
    m.eval()
    print(device)
    count = len(input_ids)
    now_count = 0
    with torch.no_grad():
        if tqdm_bar:
            pbar = tqdm.tqdm(total=count)
        while now_count < count:
            input_gpu_0 = torch.LongTensor(input_ids[now_count:min(
                now_count + batch_size, count)]).to(device)
            if summary_method == "CLS":
                embed = m(input_gpu_0)[1]
            if summary_method == "MEAN":
                embed = torch.mean(m(input_gpu_0)[0], dim=1)
            if normalize:
                embed_norm = torch.norm(
                    embed, p=2, dim=1, keepdim=True).clamp(min=1e-12)
                embed = embed / embed_norm
            if now_count == 0:
                output = embed
            else:
                output = torch.cat((output, embed), dim=0)
            if tqdm_bar:
                pbar.update(min(now_count + batch_size, count) - now_count)
            now_count = min(now_count + batch_size, count)
        if tqdm_bar:
            pbar.close()
    return output


def load_code_embedding(path):
    embedding_dict = {}
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines:
        line = line.split()
        if len(line) <= 2:
            continue
        name = line[0]
        vec = line[1:]
        embedding_dict[name] = np.array([float(number) for number in vec])
    return embedding_dict
