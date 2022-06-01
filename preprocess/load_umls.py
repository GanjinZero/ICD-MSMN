import os
from tqdm import tqdm
import re
from random import shuffle
#import ipdb

def byLineReader(filename):
    with open(filename, "r", encoding="utf-8") as f:
        line = f.readline()
        while line:
            yield line
            line = f.readline()
    return


class UMLS(object):
    def __init__(self, umls_path, source_range=None, lang_range=['ENG'], only_load_dict=False):
        self.umls_path = umls_path
        self.source_range = source_range
        self.lang_range = lang_range
        self.detect_type()
        self.load()
        # if not only_load_dict:
        #     self.load_rel()
        #     self.load_sty()

    def detect_type(self):
        if os.path.exists(os.path.join(self.umls_path, "MRCONSO.RRF")):
            self.type = "RRF"
        else:
            self.type = "txt"

    def load(self):
        reader = byLineReader(os.path.join(self.umls_path, "MRCONSO." + self.type))
        self.cui2str = {}
        self.str2cui = {}
        self.code2cui = {}
        #self.lui_status = {}
        read_count = 0
        for line in tqdm(reader, ascii=True):
            if self.type == "txt":
                l = [t.replace("\"", "") for t in line.split(",")]
            else:
                l = line.strip().split("|")
            cui = l[0]
            lang = l[1]
            # lui_status = l[2].lower() # p -> preferred
            lui = l[3]
            source = l[11]
            code = l[13]
            string = l[14]

            if source == "ICD9CM":
                self.code2cui[code] = cui

            if (self.source_range is None or source in self.source_range) and (self.lang_range is None or lang in self.lang_range):
                read_count += 1
                self.str2cui[string] = cui
                self.str2cui[string.lower()] = cui
                clean_string = self.clean(string, clean_bracket=False)
                self.str2cui[clean_string] = cui

                if not cui in self.cui2str:
                    self.cui2str[cui] = set()
                self.cui2str[cui].update([string.lower()])
                self.cui2str[cui].update([clean_string])

            # For debug
            # if read_count > 1000:
            #     break

        self.cui = list(self.cui2str.keys())
        shuffle(self.cui)
        self.cui_count = len(self.cui)

        print("cui count:", self.cui_count)
        print("str2cui count:", len(self.str2cui))
        print("MRCONSO count:", read_count)
   
    def clean(self, term, lower=True, clean_NOS=True, clean_bracket=True, clean_dash=True):
        term = " " + term + " "
        if lower:
            term = term.lower()
        if clean_NOS:
            term = term.replace(" NOS ", " ").replace(" nos ", " ")
        if clean_bracket:
            term = re.sub(u"\\(.*?\\)", "", term)
        if clean_dash:
            term = term.replace("-", " ")
        term = " ".join([w for w in term.split() if w])
        return term

    def icd2str(self, icd):
        if icd in self.code2cui:
            cui = self.code2cui[icd]
            str_list = self.cui2str[cui]
            str_list = [w for w in str_list if len(w.split()) >= 2 or len(w) >= 7]
            return list(str_list)
        return []
