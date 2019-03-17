import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import tree_def, tree_lstm, tree_utils, gen_tree_net, q_type_module
from utils import PiecewiseLin

import config

class TreeFeature(nn.Module):
    def __init__(self, objects, visual_dim, hidden_dim):
        super().__init__()
        """ overlap_tree | arbitrary_trees_attention | arbitrary_trees_transfer | accumulate_module_trees """
        """ sigmoid | softmax """
        self.gen_tree_mode = config.gen_tree_mode
        self.poolout_mode = config.poolout_mode
        if config.use_rl:
            self.dropout = 0.0
        else:
            self.dropout = 0.5
        self.objects = objects
        self.visual_dim = visual_dim
        self.hidden_dim = hidden_dim

        self.tree_lstm = tree_lstm.BidirectionalTreeLSTM(visual_dim, hidden_dim)

        self.gen_tree_net = gen_tree_net.GenTreeModule()

        if self.gen_tree_mode == "accumulate_module_trees":
            self.accumulate_module = q_type_module.AccumulatingModule(65, 151)
        #self.f = nn.ModuleList([PiecewiseLin(16) for _ in range(2)])
        
    def forward(self, boxes, attention_orig, visual_feature, v_origin, que_type):
        # only care about the highest scoring object proposals
        # the ones with low score will have a low impact on the count anyway
        boxes, attention_orig, visual_feature, v_origin = self.filter_most_important(self.objects, boxes, attention_orig, self.resize_visual_feature(visual_feature), self.resize_visual_feature(v_origin))

        if self.gen_tree_mode == "overlap_tree":
            #only use box info to generate overlap tree 
            forest = tree_utils.generate_tree(torch.transpose(boxes, 1, 2), "overlap_tree")
        elif self.gen_tree_mode == "arbitrary_trees_transfer":
            #use transfered tree-parser network to generate score matrix
            attention = F.sigmoid(attention_orig) # [batch_size, num_obj]
            relevancy = self.outer_product(attention)  
            scores = self.gen_tree_net(v_origin, boxes) 
            forest, rl_loss, entropy_loss = tree_utils.generate_tree((scores * relevancy, self.training), "arbitrary_trees")
        elif self.gen_tree_mode == "accumulate_module_trees":
            #use accumulate module to generate score matrix
            attention = F.sigmoid(attention_orig) # [batch_size, num_obj]
            #relevancy = self.outer_product(attention) # [batch_size, 10, 10]
            bbox_sim = self.iou(boxes, boxes)
            _, obj_label = self.gen_tree_net.get_label(v_origin) #[batch_size, 10, 151], [batch_size, 10]
            #print('obj_dist: ', obj_dist.shape)
            #print('obj_label: ', obj_label.data.cpu().numpy())
            packed_input = (self.accumulate_module, attention, obj_label, que_type,  bbox_sim, self.training)
            forest = tree_utils.generate_tree(packed_input, "accumulate_module_trees")
        else:
            print('Error: Please select a proper gen-tree method')

        if self.training:
            visual_hidden = self.tree_lstm(forest, torch.transpose(visual_feature, 1, 2), self.objects, self.dropout)
        else:
            visual_hidden = self.tree_lstm(forest, torch.transpose(visual_feature, 1, 2), self.objects, 0.0) # [batch_size, num_obj, hidden_size]

        del forest

        batch_size, num_obj, hidden_size = visual_hidden.shape
        return torch.transpose(visual_hidden, 1, 2).contiguous().view(batch_size, hidden_size, 1, num_obj), rl_loss, entropy_loss

    def filter_most_important(self, n, boxes, attention, visual_feature, v_origin):
        """ Only keep top-n object proposals, scored by attention weight """
        attention, idx = attention.topk(n, dim=1, sorted=False)
        idx_box = idx.unsqueeze(dim=1).expand(boxes.size(0), boxes.size(1), idx.size(1))
        boxes = boxes.gather(2, idx_box)
        idx_feat = idx.unsqueeze(dim=1).expand(visual_feature.size(0), visual_feature.size(1), idx.size(1))
        visual_feature = visual_feature.gather(2, idx_feat)
        v_origin = v_origin.gather(2, idx_feat)
        return boxes, attention, visual_feature, v_origin

    def resize_visual_feature(self, visual_feature):
        batch_size, feature_dim, _, num_obj = visual_feature.shape
        return visual_feature.view(batch_size, feature_dim, num_obj)

    def outer(self, x):
        size = tuple(x.size()) + (x.size()[-1],)
        a = x.unsqueeze(dim=-1).expand(*size)
        b = x.unsqueeze(dim=-2).expand(*size)
        return a, b

    def outer_product(self, x):
        # Y_ij = x_i * x_j
        a, b = self.outer(x)
        return a * b

    def outer_diff(self, x):
        # like outer products, except taking the absolute difference instead
        # Y_ij = | x_i - x_j |
        a, b = self.outer(x)
        return (a - b).abs()

    def iou(self, a, b):
        # this is just the usual way to IoU from bounding boxes
        inter = self.intersection(a, b)
        area_a = self.area(a).unsqueeze(2).expand_as(inter)
        area_b = self.area(b).unsqueeze(1).expand_as(inter)
        return inter / (area_a + area_b - inter + 1e-12)

    def area(self, box):
        x = (box[:, 2, :] - box[:, 0, :]).clamp(min=0)
        y = (box[:, 3, :] - box[:, 1, :]).clamp(min=0)
        return x * y

    def intersection(self, a, b):
        size = (a.size(0), 2, a.size(2), b.size(2))
        min_point = torch.max(
            a[:, :2, :].unsqueeze(dim=3).expand(*size),
            b[:, :2, :].unsqueeze(dim=2).expand(*size),
        )
        max_point = torch.min(
            a[:, 2:, :].unsqueeze(dim=3).expand(*size),
            b[:, 2:, :].unsqueeze(dim=2).expand(*size),
        )
        inter = (max_point - min_point).clamp(min=0)
        area = inter[:, 0, :, :] * inter[:, 1, :, :]
        return area