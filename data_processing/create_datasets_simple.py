""" simply re-produce this paper, especially on WAE generator"""
import re, random
from pathlib import Path
import json
import pandas as pd
import numpy as np
from pandas import DataFrame
import os, sys
sys.path.append(os.path.abspath('.'))
from utils.log_util import logger
from icecream import ic
ic.configureOutput(includeContext=True, argToStringFunction=lambda _: str(_))


swiss_prot_fasta_file = '/mnt/sda/bio_drug_corpus/uniprot/uniprot_sprot_natural.csv'
dbaasp_file = ''
DATA_ROOT = Path('/home/qcdong/bio_drug_corpus/controlled-peptide-generation')
aminoacids = ["A", "C", "D", "E", "F", "G", "H", "I", "L", "M", "N", "P", "K", "Q", "R", "S", "T", "V", "W", "Y"]


def get_seq_len_less_than(df, seq_length):
    df_short = df[df['text'].apply(lambda x: len(x) <= seq_length)]
    return df_short


def is_natural_only_upper(seq):
    """ If the char is not in upper case, treat as not natural """
    if isinstance(seq, str) and seq:
        for aa in seq:
            if aa not in aminoacids:
                return False
        return True
    return False


def create_all_seq():
    """  """
    # columns: Index(['ID', 'Sequence', 'length']
    uniprot_unk = pd.read_csv(swiss_prot_fasta_file, usecols=['Sequence'])
    ic(uniprot_unk.columns)
    ic(len(uniprot_unk))
    uniprot_unk.rename(columns={'Sequence': 'text'}, inplace=True)
    uniprot_unk["source"] = ["uniprot"] * len(uniprot_unk)
    uniprot_unk["source2"] = uniprot_unk["source"]
    uniprot_unk['source'] = uniprot_unk['source'].map({'uniprot': 'unk'})
    uniprot_unk = uniprot_unk[uniprot_unk['text'].map(is_natural_only_upper)]
    ic(len(uniprot_unk))
    uniprot_unk = get_seq_len_less_than(uniprot_unk, 50)
    allseq = uniprot_unk.drop_duplicates('text').reset_index(drop=True)
    ic(len(uniprot_unk))
    allseq.text = allseq.text.apply(lambda x: " ".join(x))
    allseq.columns = ['text', 'lab_dummy', 'source']
    allseq.to_csv(DATA_ROOT / 'amp' /"unlab_.csv", index=False, header=True)


if __name__ == "__main__":
    create_all_seq()