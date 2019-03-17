import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import PackedSequence
import numpy as np

import tree_utils
from tree_utils import block_orthogonal


class MultiLayer_BTreeLSTM(nn.Module):
    """
    Multilayer Bidirectional Tree LSTM
    Each layer contains one forward lstm(leaves to root) and one backward lstm(root to leaves)
    """
    def __init__(self, in_dim, out_dim, num_layer):
        super(MultiLayer_BTreeLSTM, self).__init__()
        self.num_layer = num_layer
        layers = []
        layers.append(BidirectionalTreeLSTM(in_dim, out_dim))
        for i in range(num_layer - 1):
            layers.append(BidirectionalTreeLSTM(out_dim, out_dim))
        self.multi_layer_lstm = nn.ModuleList(layers)

    def forward(self, forest, features, num_obj, dropout=0.0):
        for i in range(self.num_layer):
            features = self.multi_layer_lstm[i](forest, features, num_obj, dropout)
        return features


class BidirectionalTreeLSTM(nn.Module):
    """
    Bidirectional Tree LSTM
    Contains one forward lstm(leaves to root) and one backward lstm(root to leaves)
    Dropout mask will be generated one time for all trees in the forest, to make sure the consistancy
    """
    def __init__(self, in_dim, out_dim):
        super(BidirectionalTreeLSTM, self).__init__()
        self.out_dim = out_dim
        self.treeLSTM_foreward = OneDirectionalTreeLSTM(in_dim, int(out_dim / 2), 'foreward')
        self.treeLSTM_backward = OneDirectionalTreeLSTM(in_dim, int(out_dim / 2), 'backward')

    def forward(self, forest, features, num_obj, dropout=0.0):
        foreward_output = self.treeLSTM_foreward(forest, features, num_obj, dropout)
        backward_output = self.treeLSTM_backward(forest, features, num_obj, dropout)
    
        final_output = torch.cat((foreward_output, backward_output), 2)

        return final_output

class RootCentricTreeLSTM(nn.Module):
    """
    From leaves node to root node
    """
    def __init__(self, in_dim, out_dim):
        super(RootCentricTreeLSTM, self).__init__()
        self.out_dim = out_dim
        self.treeLSTM = BiTreeLSTM_Foreward(in_dim, out_dim)
    
    def forward(self, forest, features, num_obj, dropout=0.0):
        # calc dropout mask, same for all
        if dropout > 0.0:
            dropout_mask = get_dropout_mask(dropout, self.out_dim)
        else:
            dropout_mask = None

        # tree lstm input
        final_output = None
        lstm_io = tree_utils.TreeLSTM_IO(num_obj, dropout_mask)

        # run tree lstm forward (leaves to root)
        for idx in range(len(forest)):
            _, sliced_h = self.treeLSTM(forest[idx], features[idx], lstm_io, idx)
            sliced_output = sliced_h.view(1, self.out_dim)
            if final_output is None:
                final_output = sliced_output
            else:
                final_output = torch.cat((final_output, sliced_output), 0)
            # Reset hidden
            lstm_io.reset()
        
        return final_output

class OneDirectionalTreeLSTM(nn.Module):
    """
    One Way Tree LSTM
    direction = foreward | backward
    """
    def __init__(self, in_dim, out_dim, direction):
        super(OneDirectionalTreeLSTM, self).__init__()
        self.out_dim = out_dim
        self.direction = direction
        if direction == 'foreward':
            self.treeLSTM = BiTreeLSTM_Foreward(in_dim, out_dim)
        elif direction == 'backward':
            self.treeLSTM = BiTreeLSTM_Backward(in_dim, out_dim)
        else:
            print('Error Tree LSTM Direction')

    def forward(self, forest, features, num_obj, dropout=0.0):
        # calc dropout mask, same for all
        if dropout > 0.0:
            dropout_mask = get_dropout_mask(dropout, self.out_dim)
        else:
            dropout_mask = None
        
        # tree lstm input
        final_output = None
        lstm_io = tree_utils.TreeLSTM_IO(num_obj, dropout_mask)
        # run tree lstm forward (leaves to root)
        for idx in range(len(forest)):
            if self.direction == 'foreward':
                self.treeLSTM(forest[idx], features[idx], lstm_io, idx)
            elif self.direction == 'backward':
                root_c = torch.FloatTensor(self.out_dim).cuda().fill_(0.0)
                root_h = torch.FloatTensor(self.out_dim).cuda().fill_(0.0)
                self.treeLSTM(forest[idx], features[idx], lstm_io, idx, root_c, root_h)
            else:
                print('Error Tree LSTM Direction')
            sliced_output = torch.index_select(lstm_io.hidden, 0, lstm_io.order.long()).view(1, num_obj, self.out_dim)
            if final_output is None:
                final_output = sliced_output
            else:
                final_output = torch.cat((final_output, sliced_output), 0)
            # Reset hidden
            lstm_io.reset()
        
        return final_output


class BiTreeLSTM_Foreward(nn.Module):
    """
    From leaves to root
    """
    def __init__(self, feat_dim, h_dim):
        super(BiTreeLSTM_Foreward, self).__init__()
        self.feat_dim = feat_dim
        self.h_dim = h_dim

        self.ioffux = nn.Linear(self.feat_dim, 5 * self.h_dim)
        self.ioffuh_left = nn.Linear(self.h_dim, 5 * self.h_dim)
        self.ioffuh_right = nn.Linear(self.h_dim, 5 * self.h_dim)
        #self.px = nn.Linear(self.feat_dim, self.h_dim)

        # init parameter
        #block_orthogonal(self.px.weight.data, [self.h_dim, self.feat_dim])
        block_orthogonal(self.ioffux.weight.data, [self.h_dim, self.feat_dim])
        block_orthogonal(self.ioffuh_left.weight.data, [self.h_dim, self.h_dim])
        block_orthogonal(self.ioffuh_right.weight.data, [self.h_dim, self.h_dim])

        #self.px.bias.data.fill_(0.0)
        self.ioffux.bias.data.fill_(0.0)
        self.ioffuh_left.bias.data.fill_(0.0)
        self.ioffuh_right.bias.data.fill_(0.0)
        # Initialize forget gate biases to 1.0 as per An Empirical
        # Exploration of Recurrent Network Architectures, (Jozefowicz, 2015).
        self.ioffuh_left.bias.data[2 * self.h_dim:4 * self.h_dim].fill_(0.5)
        self.ioffuh_right.bias.data[2 * self.h_dim:4 * self.h_dim].fill_(0.5)


    def node_forward(self, feat_inp, left_c, right_c, left_h, right_h, dropout_mask, has_left, has_right):
        #projected_x = self.px(feat_inp)
        if has_left and has_right:
            ioffu = self.ioffux(feat_inp) + self.ioffuh_left(left_h) + self.ioffuh_right(right_h)
        elif has_left and (not has_right):
            ioffu = self.ioffux(feat_inp) + self.ioffuh_left(left_h)
        elif has_right and (not has_left):
            ioffu = self.ioffux(feat_inp) + self.ioffuh_right(right_h)
        else:
            ioffu = self.ioffux(feat_inp)
        
        i, o, f_l, f_r, u = torch.split(ioffu, ioffu.size(1) // 5, dim=1)
        i, o, f_l, f_r, u = F.sigmoid(i), F.sigmoid(o), F.sigmoid(f_l), F.sigmoid(f_r), F.tanh(u) #, F.sigmoid(r)

        c = torch.mul(i, u) + torch.mul(f_l, left_c) + torch.mul(f_r, right_c)
        h = torch.mul(o, F.tanh(c))
        #h_final = torch.mul(r, h) + torch.mul((1 - r), projected_x)
        # Only do dropout if the dropout prob is > 0.0 and we are in training mode.
        if dropout_mask is not None and self.training:
            h = torch.mul(h, dropout_mask)
        return c, h

    def forward(self, tree, features, treelstm_io, batch_idx):
        """
        tree: The root for a tree
        features: [num_obj, featuresize]
        treelstm_io.hidden: init as None, cat until it covers all objects as [num_obj, hidden_size]
        treelstm_io.order: init as 0 for all [num_obj], update for recovering original order
        """
        # recursively search child
        if tree.left_child is not None:
            has_left = True
            left_c, left_h = self.forward(tree.left_child, features, treelstm_io, batch_idx)
        else:
            has_left = False
            left_c = torch.FloatTensor(self.h_dim).cuda().fill_(0.0)
            left_h = torch.FloatTensor(self.h_dim).cuda().fill_(0.0)

        if tree.right_child is not None:
            has_right = True
            right_c, right_h = self.forward(tree.right_child, features, treelstm_io, batch_idx)
        else:
            has_right = False
            right_c = torch.FloatTensor(self.h_dim).cuda().fill_(0.0)
            right_h = torch.FloatTensor(self.h_dim).cuda().fill_(0.0)

        # calc
        next_feature = features[tree.index].view(1, -1)

        c, h = self.node_forward(next_feature, left_c, right_c, left_h, right_h, treelstm_io.dropout_mask, has_left, has_right)

        # record hidden state
        if treelstm_io.hidden is None:
            treelstm_io.hidden = h.view(1, -1)
        else:
            treelstm_io.hidden = torch.cat((treelstm_io.hidden, h.view(1, -1)), 0)
        
        treelstm_io.order[tree.index] = treelstm_io.order_count
        treelstm_io.order_count += 1

        return c, h


class BiTreeLSTM_Backward(nn.Module):
    """
    from root to leaves
    """
    def __init__(self, feat_dim, h_dim):
        super(BiTreeLSTM_Backward, self).__init__()
        self.feat_dim = feat_dim
        self.h_dim = h_dim

        self.iofux = nn.Linear(self.feat_dim, 4 * self.h_dim)
        self.iofuh = nn.Linear(self.h_dim, 4 * self.h_dim)
        #self.px = nn.Linear(self.feat_dim, self.h_dim)

        # init parameter
        #block_orthogonal(self.px.weight.data, [self.h_dim, self.feat_dim])
        block_orthogonal(self.iofux.weight.data, [self.h_dim, self.feat_dim])
        block_orthogonal(self.iofuh.weight.data, [self.h_dim, self.h_dim])

        #self.px.bias.data.fill_(0.0)
        self.iofux.bias.data.fill_(0.0)
        self.iofuh.bias.data.fill_(0.0)
        # Initialize forget gate biases to 1.0 as per An Empirical
        # Exploration of Recurrent Network Architectures, (Jozefowicz, 2015).
        self.iofuh.bias.data[2 * self.h_dim:3 * self.h_dim].fill_(1.0)

    def node_backward(self, feat_inp, root_c, root_h, dropout_mask):
        
        #projected_x = self.px(feat_inp)
        iofu = self.iofux(feat_inp) + self.iofuh(root_h)
        i, o, f, u = torch.split(iofu, iofu.size(1) // 4, dim=1)
        i, o, f, u = F.sigmoid(i), F.sigmoid(o), F.sigmoid(f), F.tanh(u) #, F.sigmoid(r)

        c = torch.mul(i, u) + torch.mul(f, root_c)
        h = torch.mul(o, F.tanh(c))
        #h_final = torch.mul(r, h) + torch.mul((1 - r), projected_x)
        # Only do dropout if the dropout prob is > 0.0 and we are in training mode.
        if dropout_mask is not None and self.training:
            h = torch.mul(h, dropout_mask)
        return c, h

    def forward(self, tree, features, treelstm_io, batch_idx, root_c, root_h):
        """
        tree: The root for a tree
        features: [num_obj, featuresize]
        treelstm_io.hidden: init as None, cat until it covers all objects as [num_obj, hidden_size]
        treelstm_io.order: init as 0 for all [num_obj], update for recovering original order
        """
        next_features = features[tree.index].view(1, -1)

        c, h = self.node_backward(next_features, root_c, root_h, treelstm_io.dropout_mask)

        # record hidden state
        if treelstm_io.hidden is None:
            treelstm_io.hidden = h.view(1, -1)
        else:
            treelstm_io.hidden = torch.cat((treelstm_io.hidden, h.view(1, -1)), 0)
 
        treelstm_io.order[tree.index] = treelstm_io.order_count
        treelstm_io.order_count += 1

        # recursively update from root to leaves
        if tree.left_child is not None:
            self.forward(tree.left_child, features, treelstm_io, batch_idx, c, h)
        if tree.right_child is not None:
            self.forward(tree.right_child, features, treelstm_io, batch_idx, c, h)

        return

def get_dropout_mask(dropout_probability, h_dim):
    """
    Computes and returns an element-wise dropout mask for a given tensor, where
    each element in the mask is dropped out with probability dropout_probability.
    Note that the mask is NOT applied to the tensor - the tensor is passed to retain
    the correct CUDA tensor type for the mask.

    Parameters
    ----------
    dropout_probability : float, required.
        Probability of dropping a dimension of the input.
    tensor_for_masking : torch.Variable, required.


    Returns
    -------
    A torch.FloatTensor consisting of the binary mask scaled by 1/ (1 - dropout_probability).
    This scaling ensures expected values and variances of the output of applying this mask
     and the original tensor are the same.
    """
    binary_mask = Variable(torch.FloatTensor(h_dim).cuda().fill_(0.0))
    binary_mask.data.copy_(torch.rand(h_dim) > dropout_probability)
    # Scale mask by 1/keep_prob to preserve output statistics.
    dropout_mask = binary_mask.float().div(1.0 - dropout_probability)
    return dropout_mask
