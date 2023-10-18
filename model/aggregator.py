""" This module contains aggregator (Attention). """
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Attention(torch.nn.Module):
    """ Attention aggregators.
    Attributes:
        - mlp_w: relation-specific transforming vector to apply on Tr(e)
        - query_relation_embedding: relation-specific z_q in nn-mechanism
        ...
    """
    def __init__(self, num_relation, num_entity, embedding_dim, max_neighbor=None, is_save_attention=False):
        super(Attention, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_relation = num_relation
        self.num_entity = num_entity
        self.max_neighbor = max_neighbor
        self.is_save_attention = is_save_attention
        self.d_model = 512
        self.n_heads = 8
        self.d_v = self.d_k = 64
        self.in_features = 100
        self.out_feature = 8
        self.dropout = 0.6
        self.alpha = 0.2
        self.d_ff = 2048
        self.n_layers = 1
        self.linear1 = nn.Linear(embedding_dim, self.d_model)
        self.linear2 = nn.Linear(self.d_model, embedding_dim)
        self.linear3 = nn.Linear(self.embedding_dim * self.n_heads, embedding_dim)
        # 多层编码层
        # self.enc_self_attn_layers = nn.ModuleList([EncoderLayer(self.d_model, self.d_ff, self.n_heads, self.d_k, self.d_v) for _ in range(self.n_layers)])
        # 单层编码层
        self.enc_self_attn = MultiHeadAttention(self.d_model, self.d_k, self.d_v, self.n_heads)
        # Transformer Encoder中的前馈神经网络
        # self.ffn = PoswiseFeedForwardNet(self.d_model, self.d_ff)
        # gat单头
        self.gat = GraphAttentionLayer(self.in_features, self.out_feature, self.dropout, self.alpha, self.max_neighbor)
        # gat多头
        # self.mutil_gat_atts = [GraphAttentionLayer(self.in_features, self.out_feature, self.dropout, self.alpha, self.max_neighbor) for _ in range(self.n_heads)]
        # for i, attention in enumerate(self.mutil_gat_atts):
        #     self.add_module('attention_{}'.format(i), attention)

        # parameters
        self.mlp_w = torch.nn.Embedding(self.num_entity * 2 + 1, self.embedding_dim)
        nn.init.xavier_normal_(self.mlp_w.weight.data)
        self.query_relation_embedding = torch.nn.Embedding(self.num_relation * 2, self.embedding_dim)
        nn.init.xavier_normal_(self.query_relation_embedding.weight.data)
        self.att_w = torch.nn.Parameter(torch.zeros(size=(self.embedding_dim * 2, self.embedding_dim * 2)))
        nn.init.xavier_normal_(self.att_w.data)
        self.att_v = torch.nn.Parameter(torch.zeros(size=(1, self.embedding_dim * 2)))
        nn.init.xavier_normal_(self.att_v.data)
        self.concat_w = torch.nn.Parameter(torch.zeros(size=(self.embedding_dim * 2, self.embedding_dim)))
        nn.init.xavier_normal_(self.concat_w.data)

        self.mask_emb = torch.cat([torch.ones([self.num_entity, 1]), torch.zeros([1, 1])], 0).\
            to(torch.cuda.current_device())
        self.mask_weight = torch.cat([torch.zeros([self.num_entity, 1]), torch.ones([1, 1])*1e19], 0).\
            to(torch.cuda.current_device())

    def forward(self, input, neighbor, query_relation_id, weight):
        input_shape = input.shape
        max_neighbors = input_shape[1]
        hidden_size = input_shape[2]

        input_relation = neighbor[:, :, 0]  # 邻居关系
        input_entity = neighbor[:, :, 1]

        transformed = self.mlp_w(input_relation)
        transformed = self._transform(input, transformed)
        mask = self.mask_emb[input_entity]
        # transformed = transformed * mask
        transformed = self.linear1(transformed)
        # for layers in self.enc_self_attn_layers:
        #     transformed, attn = layers(transformed, mask, max_neighbors)

        transformed, attention_weight = self.enc_self_attn(transformed, transformed, transformed, mask, max_neighbors)
        # transformed = self.ffn(transformed)
        transformed = self.linear2(transformed)  # self-attention层后的每个邻居结点的表示

        output = torch.mean(transformed, dim=1)
        # gatv2 没有加多头
        output, nn_attention = self.gat(transformed, output)  # 计算每个邻居结点和output的注意力权重
        # gatv2 加多头
        # att_in_output = output
        # for i, att in enumerate(self.mutil_gat_atts):
        #     if i == 0:
        #         output, nn_attention = att(transformed, att_in_output)
        #     else:
        #         att_output, att_nn_attention = att(transformed, att_in_output)
        #         output = torch.cat([output, att_output], dim=1)
        #         nn_attention = att_nn_attention
        # output = self.linear3(output)

        # query_relation = self.query_relation_embedding(query_relation_id)
        # query_relation = query_relation.unsqueeze(1)  # 第二维增加一个维度
        # query_relation = query_relation.expand(-1, max_neighbors, -1)

        # attention_logit = self.mlp(query_relation, transformed, max_neighbors)
        # mask_logit = self.mask_weight[input_entity]
        # attention_logit = attention_logit - torch.reshape(mask_logit, [-1, max_neighbors])
        # nn_weight = F.softmax(attention_logit, dim=1)
        logic_attention = weight[:, :, 0] / (weight[:, :, 1] + 1)
        # attention_weight = nn_weight + logic_attention
        # attention_weight = torch.reshape(attention_weight, [-1, max_neighbors, 1])
        # output = torch.sum(transformed * attention_weight, dim=1)
        # output = torch.cat([output, self_embedding], dim=1)
        # output = torch.matmul(output, self.concat_w)  # [batch_size, d*2] * [d*2, d]
        # attention_weight = torch.reshape(attention_weight, [-1, max_neighbors])
        return output, logic_attention, attention_weight

    def _transform(self, e, r):
        normed = F.normalize(r, p=2, dim=2)
        return e - torch.sum(e * normed, 2, keepdim=True) * normed

    def mlp(self, query, transformed, max_len):
        """ Neural network attention """
        hidden = torch.cat([query, transformed], dim=2)
        hidden = torch.reshape(hidden, [-1, self.embedding_dim * 2])
        hidden = torch.tanh(torch.matmul(hidden, self.att_w))
        hidden = torch.reshape(hidden, [-1, max_len, self.embedding_dim * 2])
        attention_logit = torch.sum(hidden * self.att_v, dim=2)
        return attention_logit


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_heads, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_model, d_k, d_v, n_heads, dropout=dropout)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_inner, dropout=dropout)

    def forward(self, enc_inputs, mask, max_neighbors):
        # 下面这个就是做自注意力层，输入是enc_inputs，形状是[batch_size x seq_len_q x d_model] 需要注意的是最初始的QKV矩阵是等同于这个输入的，去看一下enc_self_attn函数 6.
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, mask, max_neighbors)  # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size x len_q x d_model]
        return enc_outputs, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        # 输入进来的Q、K、V是相等的，我们会使用映射linear做一个映射得到参数矩阵Wq, Wk,Wv
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.linear = nn.Linear(n_heads * d_v, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, Q, K, V, mask, neighbors):
        # 这个多头分为这几个步骤，首先映射分头，然后计算atten_scores，然后计算atten_value;
        # 输入进来的数据形状： Q: [batch_size x max_neighbors x d_model], K: [batch_size x len_k x d_model], V: [batch_size x len_k x d_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        # 下面这个就是先映射，后分头；一定要注意的是q和k分头之后维度是一致的，所以一看这里都是d_k
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # q_s: [batch_size x n_heads x max_neighbors x d_k]
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # k_s: [batch_size x n_heads x max_neighbors x d_k]
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)  # v_s: [batch_size x n_heads x max_neighbors x d_v]
        # 输入进行的attn_mask形状是 batch_size x max_neighbors x len_k，然后经过下面这个代码得到 新的attn_mask : [batch_size x n_heads x max_neighbors x len_k]，就是把pad信息重复了n个头上
        # attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        # 然后我们计算 ScaledDotProductAttention 这个函数，去7.看一下
        # 得到的结果有两个：context: [batch_size x n_heads x max_neighbors x d_v], attn: [batch_size x n_heads x max_neighbors x len_k]
        attn_mask = self.get_attn_pad_mask(mask, batch_size, self.n_heads, neighbors)
        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, self.d_k, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v)  # context: [batch_size x max_neighbors x n_heads * d_v]
        context = self.dropout(self.linear(context))
        context = context + residual
        context = self.layer_norm(context)
        return context, attn  # output: [batch_size x max_neighbors x d_model]

    def get_attn_pad_mask(self, mask, batch_size, n_heads, neighbors):
        attn_mask = mask.eq(0).squeeze()
        attn_mask = attn_mask.unsqueeze(1).expand(batch_size, neighbors, -1)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
        return attn_mask


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, d_k, attn_mask):
        # 输入进来的维度分别是 [batch_size x n_heads x max_neighbors x d_k]  K： [batch_size x n_heads x len_k x d_k]  V: [batch_size x n_heads x len_k x d_v]
        # 首先经过matmul函数得到的scores形状是 : [batch_size x n_heads x max_neighbors x len_k]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)

        # 然后关键地方来了，下面这个就是用到了我们之前重点讲的attn_mask，把被mask的地方置为无限小，softmax之后基本就是0，对q的单词不起作用
        scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is one.
        # 加入位置编码

        attn = nn.Softmax(dim=-1)(scores)
        attn = self.dropout(attn)
        context = torch.matmul(attn, V)
        return context, attn


# PoswiseFeedForwardNet
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PoswiseFeedForwardNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        residual = inputs  # inputs : [batch_size, len_q, d_model]
        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2)
        output = self.dropout(output)
        output += residual
        output = self.layer_norm(output)
        return output

# gatv2
class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, max_neighbors, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.max_neighbors = max_neighbors

        self.W = nn.Parameter(torch.empty(size=(2 * in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(out_features, 1)))  # concat(V,NeigV)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, transformed, output):
        # Wh = torch.matmul(transformed, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        # a_input = self._prepare_attentional_mechanism_input(Wh)  # 每一个节点和所有节点，特征。(Vall, Vall, feature)
        output = torch.unsqueeze(output, dim=1)
        output = output.repeat_interleave(self.max_neighbors, dim=1)
        concat = torch.cat([transformed, output], dim=2)
        W_concat = torch.matmul(concat, self.W)
        a_input = self.leakyrelu(W_concat)
        e = torch.matmul(a_input, self.a)

        # e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        # 之前计算的是一个节点和所有节点的attention，其实需要的是连接的节点的attention系数
        attention = F.softmax(e, dim=1)  # 按行求softmax。 sum(axis=1) === 1
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.sum(attention * transformed, dim=1)  # 聚合邻居函数

        if self.concat:
            return F.elu(h_prime), attention  # elu-激活函数
        else:
            return h_prime, attention

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0]  # number of nodes

        # Below, two matrices are created that contain embeddings in their rows in different orders.
        # (e stands for embedding)
        # These are the rows of the first matrix (Wh_repeated_in_chunks):
        # e1, e1, ..., e1,            e2, e2, ..., e2,            ..., eN, eN, ..., eN
        # '-------------' -> N times  '-------------' -> N times       '-------------' -> N times
        #
        # These are the rows of the second matrix (Wh_repeated_alternating):
        # e1, e2, ..., eN, e1, e2, ..., eN, ..., e1, e2, ..., eN
        # '----------------------------------------------------' -> N times
        #

        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)  # 复制
        Wh_repeated_alternating = Wh.repeat(N, 1)
        # Wh_repeated_in_chunks.shape == Wh_repeated_alternating.shape == (N * N, out_features)

        # The all_combination_matrix, created below, will look like this (|| denotes concatenation):
        # e1 || e1
        # e1 || e2
        # e1 || e3
        # ...
        # e1 || eN
        # e2 || e1
        # e2 || e2
        # e2 || e3
        # ...
        # e2 || eN
        # ...
        # eN || e1
        # eN || e2
        # eN || e3
        # ...
        # eN || eN

        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        # all_combinations_matrix.shape == (N * N, 2 * out_features)

        return all_combinations_matrix.view(N, N, 2 * self.out_features)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
