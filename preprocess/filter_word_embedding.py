import sys
import os
from constant import MIMIC_3_DIR
import pandas as pd
from tqdm import tqdm
import ujson
import gensim
import numpy as np
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')

def split(text):
    sp = re.sub(r'\n\n+|  +', '\t', text.strip()).replace("\n", " ").replace("!", "\t").replace("?", "\t").replace(".", "\t")
    sp = sp.replace(",", "\t")
    return [s.strip() for s in sp.split("\t") if s.strip()]

def tokenize(text):
    texts = split(text)
    all_text = []
    split_text = []
    sentence_index = []
    word_count = 0
    for note in texts:
        now_text = [w.lower() for w in tokenizer.tokenize(note) if not w.isnumeric()]
        if now_text:
            all_text.extend(now_text)
            split_text.append(now_text)
            word_count += len(now_text)
            sentence_index.append(word_count)
    return all_text, sentence_index, split_text

def filter_word_list():
    if os.path.exists('./embedding/word_count_dict.json'):
        with open('./embedding/word_count_dict.json', 'r') as f:
            word_count_dict = ujson.load(f)
        return word_count_dict
    word_count_dict = {}
    for mode in ['train', 'dev', 'test']:
        path = os.path.join(MIMIC_3_DIR, mode + "_full.csv")
        df = pd.read_csv(path)
        texts = df['TEXT']
        for text in tqdm(texts):
            words, _, _ = split(text)
            for word in words:
                if not word in word_count_dict:
                    word_count_dict[word] = 0
                word_count_dict[word] += 1
    with open('./embedding/word_count_dict.json', 'w') as f:
        ujson.dump(word_count_dict, f, indent=2)
    return word_count_dict


def filter_word_embedding(embed_file, word_count_dict, embedding_save_path="./embedding"):
    if embed_file.endswith('.model'):
        model = gensim.models.Word2Vec.load(embed_file)
    if embed_file.endswith('.bin'):
        model = gensim.models.KeyedVectors.load_word2vec_format(embed_file, binary=True)
      
    original_word_count = len(model.wv.vocab)  
    wv = model.wv
    words_to_trim = []
    ids_to_trim = [wv.vocab[w].index for w in words_to_trim]

    for w in words_to_trim:
        del wv.vocab[w]

    wv.vectors = np.delete(wv.vectors, ids_to_trim, axis=0)

    for i in sorted(ids_to_trim, reverse=True):
        del(wv.index2word[i])
    
    save_path = os.path.join(embedding_save_path, os.path.basename(embed_file))
    model.save_word2vec_format(save_path, binary=True)
    new_model = gensim.models.KeyedVectors.load_word2vec_format(save_path, binary=True)
    new_word_count = len(new_model.wv.vocab)
    print(original_word_count, new_word_count)
    

def main():
    word_count_dict = filter_word_list()
    #embed_file = sys.argv[1]
    #filter_word_embedding(embed_file, word_count_dict)
    
    
if __name__ == "__main__":
    main()