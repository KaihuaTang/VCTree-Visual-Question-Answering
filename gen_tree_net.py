
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

class GenTreeModule(nn.Module):
    """ 
    Calculate Scores to Generate Trees
    """
    def __init__(self):
        super().__init__()
        # score fc: get 151 class distribution from bbox feature
        self.score_fc = nn.Linear(2048, 151)
        init.xavier_uniform_(self.score_fc.weight)
        self.score_fc.bias.data.zero_()
        # context for calculating gen-tree score
        self.context = LinearizedContext()

    def forward(self, visual_feat, bbox):
        """
        visual_feat: [batch, 2048, num_box]
        bbox: [batch, 4, num_box]  (x1,y1,x2,y2)
        """
        batch_size, feat_size, box_num = visual_feat.shape
        visual_feat = torch.transpose(visual_feat, 1, 2).contiguous() # [batch, num_box, feat_size]
        assert(visual_feat.shape[2] == feat_size)
        visual_feat = visual_feat.view(-1, feat_size) # [batch * num_box, feat_size]
        # prepare obj distribution
        obj_predict = self.score_fc(visual_feat)
        #obj_distrib = F.softmax(obj_predict, dim=1)[:, 1:]
        assert(obj_predict.shape[1] == 151) # [batch * num_box, 150]
        # prepare bbox feature
        bbox_trans = torch.transpose(bbox, 1, 2).contiguous()
        assert(bbox_trans.shape[2] == 4)
        bbox_feat = bbox_trans.view(-1, 4)
        bbox_embed = get_box_info(bbox_feat) # [batch * num_box, 8]
        #print('bbox_embed', bbox_embed)
        # prepare overlap feature
        overlab_embed = get_overlap_info(bbox_trans)
        #print('overlab_embed: ', overlab_embed)

        return self.context(visual_feat, obj_predict, bbox_embed, overlab_embed, batch_size, box_num)

    def get_label(self, visual_feat):
        """
        visual_feat: [batch, 2048, num_box]
        output[0]: object distribution, [batch_size, box_num, 151]
        output[1]: object label, [batch_size, box_num] 
        """
        batch_size, feat_size, box_num = visual_feat.shape
        visual_feat = torch.transpose(visual_feat, 1, 2).contiguous() # [batch, num_box, feat_size]
        assert(visual_feat.shape[2] == feat_size)
        visual_feat = visual_feat.view(-1, feat_size) # [batch * num_box, feat_size]
        obj_predict = self.score_fc(visual_feat)
        obj_distrib = F.softmax(obj_predict, dim=1)
        obj_label = obj_distrib.max(1)[1]
        return obj_distrib.view(batch_size, box_num, -1), obj_label.view(batch_size, box_num)


class LinearizedContext(nn.Module):
    """
    The name is meaningless, we just need to maintain the same structure to load transferred model.
    """
    def __init__(self):
        super().__init__()
        self.num_classes = 151
        self.embed_dim = 200
        self.obj_embed = nn.Embedding(self.num_classes, self.embed_dim)
        #self.virtual_node_embed = nn.Embedding(1, self.embed_dim)

        self.co_occour = np.load('data/co_occour_count.npy')
        self.co_occour = self.co_occour / self.co_occour.sum()

        self.rl_input_size = 256
        self.rl_hidden_size = 256
        self.feat_preprocess_net = RLFeatPreprocessNet(feat_size=2048, embed_size=self.embed_dim, bbox_size=8, overlap_size=6, output_size=self.rl_input_size)

        self.rl_sub = nn.Linear(self.rl_input_size, self.rl_hidden_size)
        self.rl_obj = nn.Linear(self.rl_input_size, self.rl_hidden_size)
        self.rl_scores = nn.Linear(self.rl_hidden_size * 3 + 3, 1)  # (left child score, right child score)

        init.xavier_uniform_(self.rl_sub.weight)
        init.xavier_uniform_(self.rl_obj.weight)
        init.xavier_uniform_(self.rl_scores.weight)
        
        self.rl_sub.bias.data.zero_()
        self.rl_obj.bias.data.zero_()
        self.rl_scores.bias.data.zero_()

    def forward(self, visual_feat, obj_predict, bbox_embed, overlap_embed, batch_size, box_num):
        """
        total = batch_size * box_num
        visual_feat: [total, 2048]
        obj_predict: [total, 151]
        bbox_embed: [total, 8]
        overlap_embed: [total, 6]
        """
        # object label embed and prediction
        num_class = 150
        obj_embed = F.softmax(obj_predict, dim=1) @ self.obj_embed.weight
        obj_distrib = F.softmax(obj_predict, dim=1)[:,1:].view(batch_size, box_num, num_class)
        # co_occour
        cooccour_matrix = Variable(torch.from_numpy(self.co_occour).float().cuda())
        class_scores = cooccour_matrix.sum(1).view(-1)

        # preprocessed features
        prepro_feat = self.feat_preprocess_net(visual_feat, obj_embed, bbox_embed, overlap_embed)
        rl_sub_feat = self.rl_sub(prepro_feat)
        rl_obj_feat = self.rl_obj(prepro_feat)
        rl_sub_feat = F.relu(rl_sub_feat).view(batch_size, box_num, -1)
        rl_obj_feat = F.relu(rl_obj_feat).view(batch_size, box_num, -1)

        # score matrix generation
        hidden_size = self.rl_hidden_size
        tree_matrix = Variable(torch.FloatTensor(batch_size, box_num * box_num).zero_().cuda())
        for i in range(batch_size):
            sliced_sub_feat = rl_sub_feat[i].view(1, box_num, hidden_size).expand(box_num, box_num, hidden_size)
            sliced_obj_feat = rl_obj_feat[i].view(box_num, 1, hidden_size).expand(box_num, box_num, hidden_size)
            sliced_sub_dist = obj_distrib[i].view(1, box_num, num_class).expand(box_num, box_num, num_class).contiguous().view(-1, num_class)
            sliced_obj_dist = obj_distrib[i].view(box_num, 1, num_class).expand(box_num, box_num, num_class).contiguous().view(-1, num_class)
            sliced_dot_dist = sliced_sub_dist.view(-1, num_class, 1) @ sliced_obj_dist.view(-1, 1, num_class) # [num_pair, 150, 150]
            sliced_dot_score = sliced_dot_dist * cooccour_matrix # [num_pair, 150, 150]

            sliced_pair_score = sliced_dot_score.view(box_num * box_num, num_class * num_class).sum(1).view(box_num, box_num, 1)
            sliced_sub_score = (sliced_sub_dist * class_scores).sum(1).view(box_num, box_num, 1)
            sliced_obj_score = (sliced_obj_dist * class_scores).sum(1).view(box_num, box_num, 1)
            sliced_pair_feat = torch.cat((sliced_sub_feat * sliced_obj_feat, sliced_sub_feat, sliced_obj_feat, sliced_pair_score, sliced_sub_score, sliced_obj_score), 2)

            sliced_pair_output = self.rl_scores(sliced_pair_feat.view(-1, hidden_size * 3 + 3))
            sliced_pair_gates = F.sigmoid(sliced_pair_output).view(-1,1) # (relation prob)
            sliced_rel_scores = (sliced_pair_score.view(-1,1) * sliced_pair_gates).view(-1) 

            tree_matrix[i, :] = sliced_rel_scores

        return tree_matrix.view(batch_size, box_num, box_num)

class RLFeatPreprocessNet(nn.Module):
    """
    Preprocess Features
    1. visual feature
    2. label prediction embed feature
    3. box embed
    4. overlap embed
    """
    def __init__(self, feat_size, embed_size, bbox_size, overlap_size, output_size):
        super(RLFeatPreprocessNet, self).__init__()
        self.feature_size = feat_size
        self.embed_size = embed_size
        self.box_info_size = bbox_size
        self.overlap_info_size = overlap_size
        self.output_size = output_size

        # linear layers
        self.resize_feat = nn.Linear(self.feature_size, int(output_size / 4))
        self.resize_embed = nn.Linear(self.embed_size, int(output_size / 4))
        self.resize_box = nn.Linear(self.box_info_size, int(output_size / 4))
        self.resize_overlap = nn.Linear(self.overlap_info_size, int(output_size / 4))

        # init
        self.resize_feat.weight.data.normal_(0, 0.001)
        self.resize_embed.weight.data.normal_(0, 0.01)
        self.resize_box.weight.data.normal_(0, 1)
        self.resize_overlap.weight.data.normal_(0, 1)
        self.resize_feat.bias.data.zero_()
        self.resize_embed.bias.data.zero_()
        self.resize_box.bias.data.zero_()
        self.resize_overlap.bias.data.zero_()

    def forward(self, obj_feat, obj_embed, box_info, overlap_info):
        resized_obj = self.resize_feat(obj_feat)
        resized_embed = self.resize_embed(obj_embed)
        resized_box = self.resize_box(box_info)
        resized_overlap = self.resize_overlap(overlap_info)

        output_feat = torch.cat((resized_obj, resized_embed, resized_box, resized_overlap), 1)
        return output_feat

def get_box_info(boxes):
    """
    input: [batch_size, (x1,y1,x2,y2)]
    output: [batch_size, (x1,y1,x2,y2,cx,cy,w,h)]
    """
    return torch.cat((boxes, center_size(boxes)), 1)

def center_size(boxes):
    """ Convert prior_boxes to (cx, cy, w, h)
    representation for comparison to center-size form ground truth data.
    Args:
        boxes: (tensor) point_form boxes
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    wh = boxes[:, 2:] - boxes[:, :2]
    return torch.cat((boxes[:, :2] + 0.5 * wh, wh), 1)

def get_overlap_info(bbox):
    """
    input:
        box_priors: [batch_size, number_obj, 4]
    output: [number_object, 6]
        number of overlapped obj (self not included)
        sum of all intersection area (self not included)
        sum of IoU (Intersection over Union)
        average of all intersection area (self not included)
        average of IoU (Intersection over Union)
        roi area
    """
    batch_size, num_obj, bsize = bbox.shape
    # generate input feat
    overlap_info = Variable(torch.FloatTensor(batch_size, num_obj, 6).zero_().cuda()) # each obj has how many overlaped objects
    reverse_eye = Variable(1.0 - torch.eye(num_obj).float().cuda()) # removed diagonal elements
    for i in range(batch_size):
        sliced_bbox = bbox[i].view(num_obj, bsize)
        sliced_intersection = bbox_intersections(sliced_bbox, sliced_bbox)
        sliced_overlap = bbox_overlaps(sliced_bbox, sliced_bbox, sliced_intersection)
        sliced_area = bbox_area(sliced_bbox)
        # removed diagonal elements
        sliced_intersection = sliced_intersection * reverse_eye
        sliced_overlap = sliced_overlap * reverse_eye
        # assign value
        overlap_info[i, :, 0] =  (sliced_intersection > 0.0).float().sum(1)
        overlap_info[i, :, 1] = sliced_intersection.sum(1)
        overlap_info[i, :, 2] = sliced_overlap.sum(1)
        overlap_info[i, :, 3] = overlap_info[i, :, 1] / (overlap_info[i, :, 0] + 1e-9)
        overlap_info[i, :, 4] = overlap_info[i, :, 2] / (overlap_info[i, :, 0] + 1e-9)
        overlap_info[i, :, 5] = sliced_area
    
    return overlap_info.view(batch_size * num_obj, 6)

def bbox_area(bbox): 
    """
    bbox: (K, 4) ndarray of float
    area: (k)
    """
    K = bbox.size(0)
    bbox_area = ((bbox[:,2] - bbox[:,0]) * (bbox[:,3] - bbox[:,1])).view(K)
    return bbox_area

def bbox_intersections(box_a, box_b):
    """
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]

def bbox_overlaps(box_a, box_b, inter=None):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    if inter is None:
        inter = bbox_intersections(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]