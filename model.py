import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from torch.nn.utils import weight_norm
from torch.nn.utils.rnn import pack_padded_sequence

import config
import counting
import tree_feature


class Net(nn.Module):
    """ Based on ``Show, Ask, Attend, and Answer: A Strong Baseline For Visual Question Answering'' [0]

    [0]: https://arxiv.org/abs/1704.03162
    """

    def __init__(self, embedding_tokens):
        super(Net, self).__init__()
        question_features = 1024
        vision_features = config.output_features
        glimpses = 2
        self.num_models = 2
        objects = 10
        tree_hidden_size = 1024

        self.text = TextProcessor(
            embedding_tokens=embedding_tokens,
            embedding_features=300,
            lstm_features=question_features,
            drop=0.5,
        )
        self.attention = Attention(
            v_features=vision_features,
            q_features=question_features,
            mid_features=512,
            glimpses=glimpses,
            drop=0.5,
        )

        self.tree_attention = Attention(
            v_features=tree_hidden_size,
            q_features=question_features,
            mid_features=512,
            glimpses=glimpses,
            drop=0.5,
        )

        self.model_attention = ModelAttention(
            q_features=question_features,
            mid_features=1024,
            q_type_num=65,
            num_models=self.num_models,
            drop=0.5,
        )
        self.classifier = Classifier(
            in_features=(glimpses * vision_features, question_features),
            mid_features=1024,
            num_module=self.num_models,
            out_features=config.max_answers,
            tree_features=tree_hidden_size * glimpses,
            count_features=objects + 1,
            drop=0.5,
        )
        self.counter = counting.Counter(objects)
        self.tree_lstm = tree_feature.TreeFeature(objects, vision_features, tree_hidden_size)

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, v_origin, b, q, q_len, q_type):
        """
        input V: [batch, 2048, 1, 36]
        input B: [batch, 4, 36]
        input Q: [batch, 23]
        input Q len: [batch]
        input Q type: [batch]
        """
        # question embedding
        q = self.text(q, list(q_len.data)) # [batch, 1024]
        # normalized visual feature
        v_norm = v_origin / (v_origin.norm(p=2, dim=1, keepdim=True) + 1e-12).expand_as(v_origin) # [batch, 2048, 1, 36]
        # attention
        a = self.attention(v_norm, q) # [batch, num_glimpse, 1, 36]
        v = apply_attention(v_norm, a) # [batch, 4096]
        # model attention
        model_att = self.model_attention(q, q_type) # [batch, num_model]
        assert(model_att.shape[1] == self.num_models * 1024)

        # this is where the counting component is used
        # pick out the first attention map
        a1 = a[:, 0, :, :].contiguous().view(a.size(0), -1) # [batch, 36]
        #a2 = a[:, 1, :, :].contiguous().view(a.size(0), -1) # [batch, 36]
        # give it and the bounding boxes to the component
        tree_feat, rl_loss, entropy_loss = self.tree_lstm(b, a1, v_norm, v_origin, q_type) # [batch, 512, 1, 10]
        #print('tree_feat: ', tree_feat.shape)
        tree_att = self.tree_attention(tree_feat, q)
        if config.poolout_mode == "softmax":
            att_t_f = apply_attention(tree_feat, tree_att)
        elif config.poolout_mode == "sigmoid":
            att_t_f = apply_attention(tree_feat, tree_att, use_softmax=False)
        else:
            print('Error')

        #count = self.counter(b, a2) # [batch, 11]

        answer = self.classifier(v, q, att_t_f, model_att) # [batch, 3000]
        return answer, rl_loss, entropy_loss

class Fusion(nn.Module):
    """ Crazy multi-modal fusion: negative squared difference minus relu'd sum
    """
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        # found through grad student descent ;)
        return - (x - y)**2 + F.relu(x + y)

class Classifier(nn.Sequential):
    def __init__(self, in_features, mid_features, num_module, tree_features, count_features, out_features, drop=0.0):
        super(Classifier, self).__init__()
        self.drop = nn.Dropout(drop)
        self.relu = nn.ReLU()
        self.fusion = Fusion()
        self.lin11 = nn.Linear(in_features[0], mid_features)
        self.lin12 = nn.Linear(in_features[1], mid_features)
        self.lin_t1 = nn.Linear(tree_features, mid_features)
        self.lin_t2 = nn.Linear(in_features[1], mid_features)
        self.lin_c = nn.Linear(count_features, mid_features)
        self.lin2 = nn.Linear(mid_features * num_module, out_features)
        
        self.bn1 = nn.BatchNorm1d(mid_features)
        self.bn2 = nn.BatchNorm1d(mid_features)
        self.bn3 = nn.BatchNorm1d(mid_features)

    def forward(self, x, y, t, model_att):
        x = self.fusion(self.lin11(self.drop(x)), self.lin12(self.drop(y)))
        t = self.fusion(self.lin_t1(self.drop(t)), self.lin_t2(self.drop(y)))
        #c = self.relu(self.lin_c(c))
        #out = self.bn1(x) * model_att[:,0].view(-1, 1) + self.bn2(t) * model_att[:,1].view(-1, 1) + self.bn3(c) * model_att[:,2].view(-1, 1)
        out = torch.cat((self.bn1(x), self.bn2(t)), 1) * model_att
        out = self.lin2(self.drop(out))
        return out


class TextProcessor(nn.Module):
    def __init__(self, embedding_tokens, embedding_features, lstm_features, drop=0.0):
        super(TextProcessor, self).__init__()
        self.embedding = nn.Embedding(embedding_tokens, embedding_features, padding_idx=0)
        self.drop = nn.Dropout(drop)
        self.tanh = nn.Tanh()
        self.lstm = nn.GRU(input_size=embedding_features,
                           hidden_size=lstm_features,
                           num_layers=1)
        self.features = lstm_features

        self._init_lstm(self.lstm.weight_ih_l0)
        self._init_lstm(self.lstm.weight_hh_l0)
        self.lstm.bias_ih_l0.data.zero_()
        self.lstm.bias_hh_l0.data.zero_()

        init.xavier_uniform_(self.embedding.weight)

    def _init_lstm(self, weight):
        for w in weight.chunk(3, 0):
            init.xavier_uniform_(w)

    def forward(self, q, q_len):
        embedded = self.embedding(q)
        tanhed = self.tanh(self.drop(embedded))
        packed = pack_padded_sequence(tanhed, q_len, batch_first=True)
        _, h = self.lstm(packed)
        return h.squeeze(0)


class Attention(nn.Module):
    def __init__(self, v_features, q_features, mid_features, glimpses, drop=0.0):
        super(Attention, self).__init__()
        self.v_conv = nn.Conv2d(v_features, mid_features, 1, bias=False)  # let self.lin take care of bias
        self.q_lin = nn.Linear(q_features, mid_features)
        self.x_conv = nn.Conv2d(mid_features, glimpses, 1)

        self.drop = nn.Dropout(drop)
        self.relu = nn.ReLU(inplace=True)
        self.fusion = Fusion()

    def forward(self, v, q):
        #q_in = q
        v = self.v_conv(self.drop(v))
        q = self.q_lin(self.drop(q))
        q = tile_2d_over_nd(q, v)
        x = self.fusion(v, q)
        x = self.x_conv(self.drop(x))
        return x

class ModelAttention(nn.Module):
    def __init__(self, q_features, mid_features, q_type_num, num_models, drop=0.0):
        super(ModelAttention, self).__init__()
        self.q_lin_1 = nn.Linear(q_features, 256)

        self.q_type_1 = nn.Embedding(q_type_num, 256)
        self.q_type_2 = nn.Linear(256, 256)

        self.lin_fuse = nn.Linear(256, num_models*mid_features)
        self.bn1 = nn.BatchNorm1d(256)
        #self.bn2 = nn.BatchNorm1d(256)

        self.drop = nn.Dropout(drop)
        self.relu = nn.ReLU()
        self.fusion = Fusion()

        init.xavier_uniform_(self.q_lin_1.weight)
        #init.xavier_uniform_(self.q_lin_2.weight)
        init.xavier_uniform_(self.q_type_1.weight)
        init.xavier_uniform_(self.q_type_2.weight)

        self.q_lin_1.bias.data.zero_()
        #self.q_lin_2.bias.data.zero_()
        self.q_type_2.bias.data.zero_()
        

    def forward(self, q, q_type):
        q = self.q_lin_1(self.drop(q)) # [batch, 256]

        q_t = self.q_type_1(q_type)
        q_t = self.q_type_2(self.drop(q_t))

        fused_q = self.fusion(q, q_t)
        fused_q = self.lin_fuse(self.drop(self.bn1(fused_q)))

        att = F.sigmoid(fused_q)
        return att

def apply_attention(input, attention, use_softmax=True):
    """ Apply any number of attention maps over the input.
        The attention map has to have the same size in all dimensions except dim=1.
    """
    n, c = input.size()[:2]
    glimpses = attention.size(1)

    # flatten the spatial dims into the third dim, since we don't need to care about how they are arranged
    input = input.view(n, c, -1)
    attention = attention.view(n, glimpses, -1)
    s = input.size(2)

    # apply a softmax to each attention map separately
    # since softmax only takes 2d inputs, we have to collapse the first two dimensions together
    # so that each glimpse is normalized separately
    attention = attention.view(n * glimpses, -1)
    if use_softmax:
        attention = F.softmax(attention, dim=1)
    else:
        attention = F.sigmoid(attention)

    # apply the weighting by creating a new dim to tile both tensors over
    target_size = [n, glimpses, c, s]
    input = input.view(n, 1, c, s).expand(*target_size)
    attention = attention.view(n, glimpses, 1, s).expand(*target_size)
    weighted = input * attention
    # sum over only the spatial dimension
    weighted_mean = weighted.sum(dim=3, keepdim=True)
    # the shape at this point is (n, glimpses, c, 1)
    return weighted_mean.view(n, -1)


def tile_2d_over_nd(feature_vector, feature_map):
    """ Repeat the same feature vector over all spatial positions of a given feature map.
        The feature vector should have the same batch size and number of features as the feature map.
    """
    n, c = feature_vector.size()
    spatial_sizes = feature_map.size()[2:]
    tiled = feature_vector.view(n, c, *([1] * len(spatial_sizes))).expand(n, c, *spatial_sizes)
    return tiled
