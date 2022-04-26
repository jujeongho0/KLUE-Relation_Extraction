import pickle as pickle
import os
from turtle import forward
import pandas as pd
import torch
import sklearn
import numpy as np
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoConfig, Trainer, TrainingArguments
from transformers import AutoModel
from load_data import *
# import wandb
import torch.nn as nn
import random
import argparse
from loss import *

# Dice loss 사용하기 위해선 pip insatll sadice 터미널로 실행!
# from sadice import SelfAdjDiceLoss

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

# BiGRU -> FC
class Model(nn.Module):
  def __init__(self, MODEL_NAME):
    super().__init__()
    self.model_config =  AutoConfig.from_pretrained(MODEL_NAME)
    self.model_config.num_labels = 30
    self.model = AutoModel.from_pretrained(MODEL_NAME, config = self.model_config)
    self.hidden_dim = self.model_config.hidden_size
    self.gru= nn.GRU(input_size= self.hidden_dim, hidden_size= self.hidden_dim, num_layers= 1, batch_first= True, bidirectional= True)
    self.fc = nn.Linear(self.hidden_dim * 2, self.model_config.num_labels)
  
  def forward(self, input_ids, attention_mask):
    output = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    # (batch, max_len, hidden_dim)

    hidden, last_hidden = self.gru(output)
    output = torch.cat((last_hidden[0], last_hidden[1]), dim=1)
    # hidden : (batch, max_len, hidden_dim * 2)
    # last_hidden : (2, batch, hidden_dim)
    # output : (batch, hidden_dim * 2)
    logits = self.fc(output)
    # logits : (batch, num_labels)

    return {'logits' : logits}

# BiLSTM -> FC
class Model2(nn.Module):
  def __init__(self, MODEL_NAME):
    super().__init__()
    self.model_config =  AutoConfig.from_pretrained(MODEL_NAME)
    self.model_config.num_labels = 30
    self.model = AutoModel.from_pretrained(MODEL_NAME, config = self.model_config)
    self.hidden_dim = self.model_config.hidden_size
    self.lstm= nn.LSTM(input_size= self.hidden_dim, hidden_size= self.hidden_dim, num_layers= 1, batch_first= True, bidirectional= True)
    self.fc = nn.Linear(self.hidden_dim * 2, self.model_config.num_labels)
  
  def forward(self, input_ids, attention_mask):
    output = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    # (batch, max_len, hidden_dim)

    hidden, (last_hidden, last_cell) = self.lstm(output)
    output = torch.cat((last_hidden[0], last_hidden[1]), dim=1)
    # hidden : (batch, max_len, hidden_dim * 2)
    # last_hidden : (2, batch, hidden_dim)
    # output : (batch, hidden_dim * 2)

    logits = self.fc(output)
    # logits : (batch, num_labels)

    return {'logits' : logits}

# FC
class Model3(nn.Module):
  def __init__(self, MODEL_NAME):
    super().__init__()
    self.model_config =  AutoConfig.from_pretrained(MODEL_NAME)
    self.model_config.num_labels = 30
    self.model = AutoModel.from_pretrained(MODEL_NAME, config = self.model_config)
    self.hidden_dim = self.model_config.hidden_size
    self.fc = nn.Linear(self.hidden_dim * 160, self.model_config.num_labels)
  
  def forward(self, input_ids, attention_mask):
    output = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    # (batch, max_len, hidden_dim)
    
    output = output.view((output.shape[0], -1))
    # (batch, max_len * hidden_dim)

    logits = self.fc(output)
    # logits : (batch, num_labels)

    return {'logits' : logits}

class MultiHeadedAttention(nn.Module):
    def __init__(self,d_feat=128,n_head=5,actv=F.relu,USE_BIAS=True,dropout_p=0.1):
        super(MultiHeadedAttention,self).__init__()
        if (d_feat%n_head) != 0:
            raise ValueError("d_feat(%d) should be divisible by b_head(%d)"%(d_feat,n_head)) 
        self.d_feat = d_feat
        self.n_head = n_head
        self.d_head = self.d_feat // self.n_head
        self.actv = actv
        self.USE_BIAS = USE_BIAS
        self.dropout_p = dropout_p # prob. of zeroed

        self.lin_Q = nn.Linear(self.d_feat,self.d_feat,self.USE_BIAS)
        self.lin_K = nn.Linear(self.d_feat,self.d_feat,self.USE_BIAS)
        self.lin_V = nn.Linear(self.d_feat,self.d_feat,self.USE_BIAS)
        self.lin_O = nn.Linear(self.d_feat,self.d_feat,self.USE_BIAS)

        self.dropout = nn.Dropout(p=self.dropout_p)
    
    def forward(self,Q,K,V,mask=None):
        n_batch = Q.shape[0]
        Q_feat = self.lin_Q(Q) 
        K_feat = self.lin_K(K) 
        V_feat = self.lin_V(V)
        # Q_feat: [n_batch, n_Q, d_feat]
        # K_feat: [n_batch, n_K, d_feat]
        # V_feat: [n_batch, n_V, d_feat]

        # Multi-head split of Q, K, and V (d_feat = n_head*d_head)
        Q_split = Q_feat.view(n_batch, -1, self.n_head, self.d_head).permute(0, 2, 1, 3)
        K_split = K_feat.view(n_batch, -1, self.n_head, self.d_head).permute(0, 2, 1, 3)
        V_split = V_feat.view(n_batch, -1, self.n_head, self.d_head).permute(0, 2, 1, 3)
        # Q_split: [n_batch, n_head, n_Q, d_head]
        # K_split: [n_batch, n_head, n_K, d_head]
        # V_split: [n_batch, n_head, n_V, d_head]

        # Multi-Headed Attention
        d_K = K.size()[-1] # key dimension
        scores = torch.matmul(Q_split, K_split.permute(0,1,3,2)) / np.sqrt(d_K)
        if mask is not None:
            scores = scores.masked_fill(mask==0,-1e9)
        attention = torch.softmax(scores,dim=-1)
        x_raw = torch.matmul(self.dropout(attention),V_split) # dropout is NOT mentioned in the paper
        # attention: [n_batch, n_head, n_Q, n_K]
        # x_raw: [n_batch, n_head, n_Q, d_head]

        # Reshape x
        x_rsh1 = x_raw.permute(0,2,1,3).contiguous()
        # x_rsh1: [n_batch, n_Q, n_head, d_head]
        x_rsh2 = x_rsh1.view(n_batch,-1,self.d_feat)
        # x_rsh2: [n_batch, n_Q, d_feat]

        # Linear
        x = self.lin_O(x_rsh2)
        # x: [n_batch, n_Q, d_feat]
        out = {'Q_feat':Q_feat,'K_feat':K_feat,'V_feat':V_feat,
               'Q_split':Q_split,'K_split':K_split,'V_split':V_split,
               'scores':scores,'attention':attention,
               'x_raw':x_raw,'x_rsh1':x_rsh1,'x_rsh2':x_rsh2,'x':x}
        return out

# BiGRU -> MHA -> BiGRU -> FC
class Model4(nn.Module):
  def __init__(self, MODEL_NAME):
    super().__init__()
    self.model_config =  AutoConfig.from_pretrained(MODEL_NAME)
    self.model_config.num_labels = 30
    self.model = AutoModel.from_pretrained(MODEL_NAME, config = self.model_config)
    self.hidden_dim = self.model_config.hidden_size
    self.gru1= nn.GRU(input_size= self.hidden_dim, hidden_size= self.hidden_dim // 2, num_layers= 1, batch_first= True, bidirectional= True)
    self.gru2= nn.GRU(input_size= self.hidden_dim , hidden_size= self.hidden_dim // 2, num_layers= 1, batch_first= True, bidirectional= True)
    self.mha = MultiHeadedAttention(d_feat=self.hidden_dim,n_head=4,actv=F.relu,USE_BIAS=False,dropout_p=0.1)
    self.fc = nn.Linear(self.hidden_dim, self.model_config.num_labels)
  
  def forward(self, input_ids, attention_mask):
    output = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    # (batch, max_len, hidden_dim)

    hidden, last_hidden = self.gru1(output)
    # hidden : (batch, max_len, hidden_dim)
    # last_hidden : (2, batch, hidden_dim // 2)

    output = self.mha(hidden, hidden, hidden, mask = None)['x']
    # (batch, max_len, hidden_dim)

    hidden, last_hidden = self.gru2(output)
    # hidden : (batch, max_len, hidden_dim)
    # last_hidden : (2, batch, hidden_dim // 2)

    output = torch.cat((last_hidden[0], last_hidden[1]), dim=1)
    # output : (batch, hidden_dim)
    logits = self.fc(output)
    # logits : (batch, num_labels)
    
    return {'logits' : logits}

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def compute_loss(self, model, inputs, return_outputs= False):
        device= torch.device('cuda:0' if torch.cuda.is_available else 'cpu:0')
        labels= inputs.pop('labels')
        # print(labels)
        # forward pass
        outputs= model(**inputs)
        
        # 인덱스에 맞춰서 과거 ouput을 다 저장
        if self.args.past_index >=0:
            self._past= outputs[self.args.past_index]
            
        # compute custom loss (suppose one has 3 labels with different weights)

        # custom loss func by args.criterion
        custom_loss = use_criterion(args.criterion).to(device)
        loss = custom_loss(outputs['logits'], labels)    
        return (loss, outputs) if return_outputs else loss
        

def klue_re_micro_f1(preds, labels):
    """KLUE-RE micro f1 (except no_relation)"""
    label_list = ['no_relation', 'org:top_members/employees', 'org:members',
       'org:product', 'per:title', 'org:alternate_names',
       'per:employee_of', 'org:place_of_headquarters', 'per:product',
       'org:number_of_employees/members', 'per:children',
       'per:place_of_residence', 'per:alternate_names',
       'per:other_family', 'per:colleagues', 'per:origin', 'per:siblings',
       'per:spouse', 'org:founded', 'org:political/religious_affiliation',
       'org:member_of', 'per:parents', 'org:dissolved',
       'per:schools_attended', 'per:date_of_death', 'per:date_of_birth',
       'per:place_of_birth', 'per:place_of_death', 'org:founded_by',
       'per:religion']
    no_relation_label_idx = label_list.index("no_relation")
    label_indices = list(range(len(label_list)))
    label_indices.remove(no_relation_label_idx)
    return sklearn.metrics.f1_score(labels, preds, average="micro", labels=label_indices) * 100.0


def klue_re_auprc(probs, labels):
    """KLUE-RE AUPRC (with no_relation)"""
    labels = np.eye(30)[labels]
    score = np.zeros((30,))
    for c in range(30):
        targets_c = labels.take([c], axis=1).ravel()
        preds_c = probs.take([c], axis=1).ravel()
        precision, recall, _ = sklearn.metrics.precision_recall_curve(targets_c, preds_c)
        score[c] = sklearn.metrics.auc(recall, precision)
    return np.average(score) * 100.0


def compute_metrics(pred):
  """ validation을 위한 metrics function """
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  probs = pred.predictions

  # calculate accuracy using sklearn's function
  f1 = klue_re_micro_f1(preds, labels)
  auprc = klue_re_auprc(probs, labels)
  acc = accuracy_score(labels, preds) # 리더보드 평가에는 포함되지 않습니다.
  return {
      'micro f1 score': f1,
      'auprc' : auprc,
      'accuracy': acc,
  }


def label_to_num(label):
  """ 주어진 pickle 파일로 부터 label->num list를 불러와 train_dataset(DataFrame)의 label column의 값에 대응하는 숫자를 list에 담아 전달합니다.

  Args:
      label (DataFrame.values): train_dataset(DataFrame)의 label column의 값

  Returns:
      list: 대응하는 숫자
  """  
  num_label = []
  with open('dict_label_to_num.pkl', 'rb') as f:
    dict_label_to_num = pickle.load(f)
  for v in label:
    num_label.append(dict_label_to_num[v])
  
  return num_label

def train():

  # load model and tokenizer
  MODEL_NAME = "klue/roberta-large"
  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
  added_token_num = tokenizer.add_special_tokens({"additional_special_tokens":["[LOC]", "[DAT]", "[NOH]", "[PER]", "[ORG]", "[POH]"]})
  
  # load dataset
  train_dataset = load_data("/opt/ml/dataset/train/train.csv")

  train_label = label_to_num(train_dataset['label'].values)

  # tokenizing dataset
  tokenized_train = tokenized_dataset(train_dataset, tokenizer)

  # make dataset for pytorch.
  RE_train_dataset = RE_Dataset(tokenized_train, train_label)


  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  

  model =  Model2(MODEL_NAME)
  model.model_config.vocab_size = len(tokenizer)
  model.model.resize_token_embeddings(len(tokenizer))
  state_dict = torch.load(os.path.join(f'./best_model_14', 'pytorch_model.bin'))

  model.load_state_dict(state_dict)

  model.to(device)
 
  # 사용한 option 외에도 다양한 option들이 있습니다.
  # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
  training_args = TrainingArguments(
    output_dir='./results',          # output directory
    save_strategy='no',
    save_total_limit=1,              # number of total save model.
    num_train_epochs=5,              # total number of training epochs
    learning_rate=3e-6,               # learning_rate
    per_device_train_batch_size=32,  # batch size per device during training
    gradient_accumulation_steps=2,   # gradient accumulation factor
    per_device_eval_batch_size=64,   # batch size for evaluation
    fp16=True,
    warmup_ratio = 0.1,

    weight_decay=0.01,               # strength of weight decay
    label_smoothing_factor=0.1,
    # lr_scheduler_type = 'cosine',
    logging_dir='./logs',            # directory for storing logs
    logging_steps=100,              # log saving step.
    evaluation_strategy='no', # evaluation strategy to adopt during training
                                # `no`: No evaluation during training.
                                # `steps`: Evaluate every `eval_steps`.
                                # `epoch`: Evaluate every end of epoch.
    load_best_model_at_end = True,
    report_to = 'wandb',
    # run name은 실험자명과 주요 변경사항을 기입합니다. 
    run_name = f'kiwon-len=128/Acm=2/label_sm=0.1/lr=3e-6/loss=CE/BiGRU4layer/seed={seed_value}'


  )

  trainer = CustomTrainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=RE_train_dataset,         # training dataset
    # eval_dataset=RE_train_dataset,             # evaluation dataset
    compute_metrics=compute_metrics         # define metrics function
    
  )

  
  trainer.train()
  
  torch.save(model.state_dict(), os.path.join(f'./best_model_JH_{seed_value}', 'pytorch_model_2.bin'))

def main():
  train()

if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument("--criterion", type=str, default='default', help='criterion type: label_smoothing, focal_loss')

  args = parser.parse_args()
  print(args)

  
  # run name은 실험자명과 주요 변경사항을 기입합니다.


  wandb.init(project="KLUE")
  seed_iter = 5
  seed_value = 14*seed_iter
  wandb.run.name = f'kiwon-len=128/Acm=2/label_sm=0.1/lr=3e-6/loss=CE/BiGRU4layer/seed={seed_value}'
  seed_everything(seed_value) 
  os.environ["WANDB_DISABLED"] = "true"
  
  main()

