from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from transformers import GPT2ForSequenceClassification
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers import BertTokenizer, BertModel
from einops import rearrange
from models.embed import DataEmbedding, DataEmbedding_wo_time

'''
这个模型适用于处理时间序列数据的分类任务，例如股票价格预测、天气预测或任何需要从时间序列中提取模式并进行分类的任务。
它通过将传统的 GPT-2 模型与时间序列数据特有的处理方式相结合，使得模型能够更好地理解和处理这类数据。

整体而言，gpt4ts 类提供了一种创新的方式，将 GPT-2 模型的强大能力应用于时间序列数据的分类任务，
同时通过对输入数据的特殊处理，确保模型能够有效地处理这类数据。
'''

class gpt4ts(nn.Module):

    '''
    config 和 data 为函数参数，分别包含配置信息和数据。
    pred_len 表示预测长度（初始化为 0）。
    seq_len 和 max_len 从 data 对象获取，表示序列的最大长度。
    patch_size 和 stride 从 config 获取，用于处理时间序列。
    gpt_layers 定义了 GPT-2 模型中使用的层数。
    feat_dim 表示数据的特征维度，从 data.feature_df 获取。
    num_classes 表示类别的数量，从 data.class_names 获取。
    d_model 是模型维度，从 config 获取。
    patch_num 根据序列长度、patch_size 和 stride 计算。
    使用 nn.ReplicationPad1d 和 DataEmbedding 进行数据嵌入。
    加载预训练的 GPT-2 模型并进行适当的修改。
    定义激活函数、dropout、层归一化和输出层。
    '''
    def __init__(self, config, data):
        super(gpt4ts, self).__init__()
        self.pred_len = 0
        self.seq_len = data.max_seq_len
        self.max_len = data.max_seq_len
        self.patch_size = config['patch_size']
        self.stride = config['stride']
        self.gpt_layers = 6
        self.feat_dim = data.feature_df.shape[1]
        self.num_classes = len(data.class_names)
        self.d_model = config['d_model']

        self.patch_num = (self.seq_len - self.patch_size) // self.stride + 1

        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
        self.patch_num += 1
        self.enc_embedding = DataEmbedding(self.feat_dim * self.patch_size, config['d_model'], config['dropout'])

        self.gpt2 = GPT2Model.from_pretrained('/home/sunnyhorse/JiaoChen/Emitter_Classification/gpt2', output_attentions=True, output_hidden_states=True)
        self.gpt2.h = self.gpt2.h[:self.gpt_layers]
        
        for i, (name, param) in enumerate(self.gpt2.named_parameters()):
            if 'ln' in name or 'wpe' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        device = torch.device('cuda:{}'.format(0))
        self.gpt2.to(device=device)

        self.act = F.gelu
        self.dropout = nn.Dropout(0.1)
        self.ln_proj = nn.LayerNorm(config['d_model'] * self.patch_num)
        
        self.ln_proj = nn.LayerNorm(config['d_model'] * self.patch_num)
        self.out_layer = nn.Linear(config['d_model'] * self.patch_num, self.num_classes)
    
    '''
    输入 x_enc 和 x_mark_enc 分别表示编码的数据和时间标记。
    将输入数据进行重排、填充、展开以形成补丁。
    将处理后的数据通过编码嵌入层和 GPT-2 模型。
    使用激活函数、层归一化处理 GPT-2 的输出，然后通过输出层产生最终的分类结果。
    '''
    def forward(self, x_enc, x_mark_enc):
        B, L, M = x_enc.shape
        
        input_x = rearrange(x_enc, 'b l m -> b m l')
        input_x = self.padding_patch_layer(input_x)
        input_x = input_x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        input_x = rearrange(input_x, 'b m n p -> b n (p m)')
        
        outputs = self.enc_embedding(input_x, None)
        
        outputs = self.gpt2(inputs_embeds=outputs).last_hidden_state

        outputs = self.act(outputs).reshape(B, -1)
        outputs = self.ln_proj(outputs)
        outputs = self.out_layer(outputs)
        
        return outputs

    