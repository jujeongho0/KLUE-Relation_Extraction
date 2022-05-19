from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from load_data import *
import pandas as pd
import torch
import torch.nn.functional as F
import pickle as pickle
import numpy as np
import argparse
from tqdm import tqdm
from train import *
from df_edit import better_df
import streamlit as st
st.set_page_config(layout="wide")

import tokenizers
@st.cache(hash_funcs={tokenizers.Tokenizer: lambda _: None, tokenizers.AddedToken: lambda _: None})
def model_load():
  MODEL_NAME = "klue/roberta-large"
  model = Model2(MODEL_NAME)
  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
  added_token_num = tokenizer.add_special_tokens({"additional_special_tokens":["[LOC]", "[DAT]", "[NOH]", "[PER]", "[ORG]", "[POH]"]})
  model.model_config.vocab_size = len(tokenizer)
  model.model.resize_token_embeddings(len(tokenizer))
  state_dict = torch.load(os.path.join(f'./best_model', 'pytorch_model'))
  model.load_state_dict(state_dict)  
  return tokenizer, model

def main():
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  tokenizer, model = model_load()
  model.to(device)
 
  with st.form("문장 속 단어의 관계 예측하기"):
    sentence = st.text_input("문장을 입력해주세요")
    subject_entity_word = st.text_input("subject를 입력해주세요")
    subject_entity_start_idx = st.number_input("subject의 start_idx를 입력해주세요")
    subject_entity_end_idx = st.number_input("subject의 end_idx를 입력해주세요")
    subject_entity_type = st.text_input("subject의 type을 입력해주세요")
    object_entity_word = st.text_input("object을 입력해주세요")
    object_entity_start_idx = st.number_input("object의 start_idx를 입력해주세요")
    object_entity_end_idx = st.number_input("object의 end_idx를 입력해주세요")
    object_entity_type = st.text_input("object의 type을 입력해주세요")

    submitted = st.form_submit_button("결과 보기")
    if submitted:
      subject_entity = {'word': subject_entity_word, 'start_idx': subject_entity_start_idx, 'end_idx': subject_entity_end_idx, 'type': subject_entity_type}
      object_entity = {'word': object_entity_word, 'start_idx': object_entity_start_idx, 'end_idx': object_entity_end_idx, 'type': object_entity_type}

      test_dataset = pd.DataFrame({'id': [0], 'sentence':[sentence], 'subject_entity':[str(subject_entity)],'object_entity':[str(object_entity)], 'label': [100]})
      test_label = list(map(int,test_dataset['label'].values))
      test_dataset = better_df(test_dataset,1)
      test_dataset = preprocessing_dataset_with_sentence(test_dataset)
      test_dataset = tokenized_dataset(test_dataset, tokenizer)
      Re_test_dataset = RE_Dataset(test_dataset ,test_label)

      ## predict answer
      pred_answer, output_prob = inference(model, Re_test_dataset, device) # model에서 class 추론
      pred_answer = num_to_label(pred_answer) # 숫자로 된 class를 원래 문자열 라벨로 변환.
      st.write(f"<{subject_entity_word}>와(과) <{object_entity_word}>의 관계는 <{max(output_prob[0])}>의 확률로 <{pred_answer[0]}>입니다.")
      st.balloons()
  

st.title("NLP 10조 HOT6IX")
main()
