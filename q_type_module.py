import numpy as np
import torch
import torch.nn as nn
import itertools
from torch.autograd import Variable
import tree_def, tree_def

class AccumulatingModule(nn.Module):
    """ 
    Accumulating Module 
    According to different detailed question types (65 types in total), each question type will have an individual [num_ot, num_ot] matrix. 
    The parameters of [num_ot, num_ot] matrix are updated by statistical accumulating (not backpropagation).

    num_qt: number of question types -> 65
    num_ot: number of object types -> 151
    """
    def __init__(self, num_qt, num_ot):
        super().__init__()
        self.num_qt = num_qt
        self.num_ot = num_ot
        self.pair_num = 90
        self.score_matrix = nn.Parameter(torch.zeros(num_qt, self.pair_num, num_ot, num_ot).float().fill_(1e-12))

    def batch_update_matrix(self, obj_label, qus_type, attention):
        """
        obj_label: [batch_size, box_num]
        attention: [batch_size, box_num]
        qus_type: [batch_size]
        """
        obj_label = obj_label.data
        qus_type = qus_type.data
        attention = attention.detach().data

        batch_size, box_num = obj_label.shape
        ol1 = obj_label.view(batch_size, 1, box_num).expand(batch_size, box_num, box_num).contiguous()
        ol2 = obj_label.view(batch_size, box_num, 1).expand(batch_size, box_num, box_num).contiguous()
        eye = torch.eye(box_num).cuda().long().view(1, box_num, box_num).expand(batch_size, box_num, box_num).contiguous()
        ol1 = ol1.view(-1)[torch.nonzero((1-eye).view(-1))].view(-1)
        ol2 = ol2.view(-1)[torch.nonzero((1-eye).view(-1))].view(-1)
        assert ol1.shape[0] == batch_size * box_num * (box_num - 1)
        assert ol2.shape[0] == batch_size * box_num * (box_num - 1)
        qt = qus_type.view(batch_size, 1).expand(batch_size, box_num * (box_num - 1)).contiguous().view(-1)
        ra = torch.range(0, self.pair_num-1).cuda().long().view(1, self.pair_num).expand(batch_size, self.pair_num).contiguous().view(-1)
        # score
        at1 = attention.view(batch_size, 1, box_num).expand(batch_size, box_num, box_num).contiguous()
        at2 = attention.view(batch_size, box_num, 1).expand(batch_size, box_num, box_num).contiguous()
        at1 = at1.view(-1)[torch.nonzero((1-eye).view(-1))].view(-1)
        at2 = at2.view(-1)[torch.nonzero((1-eye).view(-1))].view(-1)
        at = at1 * at2
        # update
        self.score_matrix.data[qt, ra, ol1, ol2] += at.data

    
    def update_matrix(self, obj_label, qus_type, attention):
        """
        obj_label: [box_num]
        attention: [box_num]
        qus_type: 1
        """
        box_num = obj_label.shape[0]
        for i in range(box_num):
            for j in range(box_num):
                if i != j:
                    # make sure it is sysmetrical metrix
                    self.score_matrix[int(qus_type), int(obj_label[i]), int(obj_label[j])] = self.score_matrix[int(qus_type), int(obj_label[i]), int(obj_label[j])] + float(attention[i] * attention[j])
                    self.score_matrix[int(qus_type), int(obj_label[j]), int(obj_label[i])] = self.score_matrix[int(qus_type), int(obj_label[j]), int(obj_label[i])] + float(attention[i] * attention[j])

    def get_matrix(self, obj_label, qus_type):
        """
        obj_label: [box_num]
        qus_type: 1
        """
        box_num = obj_label.shape[0]
        sliced_matrix = self.score_matrix[int(qus_type)].sum(0).data
        normed_matrix = sliced_matrix / sliced_matrix.max()
        ol1 = obj_label.view(box_num, 1).expand(box_num, box_num)
        ol2 = obj_label.view(1, box_num).expand(box_num, box_num)
        output = normed_matrix[ol1, ol2].clone()
        return output + output.transpose(0, 1)
        
