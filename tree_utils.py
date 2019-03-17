import errno
import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
from torch.autograd import Variable
import tree_def, tree_utils, tree_def
import config

def generate_tree(gen_tree_input, tree_type="overlap_tree"):
    """
    """
    if tree_type == "overlap_tree":
        # gen_tree_input => bbox_shape : [batch, num_objs, b_dim]  (x1,y1,x2,y2)
        return generate_overlap_tree(gen_tree_input)
    elif tree_type == "arbitrary_trees":
        # gen_tree_input => scores : [batch_size, 10, 10]
        return generate_arbitrary_trees(gen_tree_input)
    else:
        print("Invalid Tree Type")
        return None


def generate_arbitrary_trees(inputpack):
    """ Generate arbiraty trees according to the bbox scores """
    scores, is_training = inputpack
    trees = []
    batch_size, num_obj, _ = scores.size()
    sym_score = scores + scores.transpose(1,2)
    rl_loss = []
    entropy_loss = []
    for i in range(batch_size):
        slice_container, index_list = return_tree_contrainer(num_obj, i)
        slice_root_score = (sym_score[i].sum(1) - sym_score[i].diag()) / 100.0
        slice_bitree = ArTree_to_BiTree(return_tree(sym_score[i], slice_root_score, slice_container, index_list, rl_loss, entropy_loss, is_training))
        trees.append(slice_bitree)

    #trees = [create_single_tree(sym_score[i], i, num_obj) for i in range(batch_size)]
    return trees, rl_loss, entropy_loss

#def create_single_tree(single_sym_score, i, num_obj):
#    slice_container = [tree_def.ArbitraryTree(index, im_idx=i) for index in range(num_obj)]
#    index_list = [index for index in range(num_obj)]
#    single_root_score = single_sym_score.sum(1) - single_sym_score.diag()
#    slice_bitree = ArTree_to_BiTree(return_tree(single_sym_score, single_root_score, slice_container, index_list))
#    return slice_bitree


def return_tree(matrix_score, root_score, node_containter, remain_list, gen_tree_loss_per_batch, entropy_loss, is_training):
    """ Generate An Arbitrary Tree by Scores """
    virtual_root = tree_def.ArbitraryTree(-1, im_idx=-1)
    virtual_root.is_root = True

    start_idx = int(root_score.argmax())
    start_node = node_containter[start_idx]
    virtual_root.add_child(start_node)
    assert(start_node.index == start_idx)
    select_list = []
    selected_node = []
    select_list.append(start_idx)
    selected_node.append(start_node)
    remain_list.remove(start_idx)
    node_containter.remove(start_node)

    not_sampled = True

    while(len(node_containter) > 0):
        wid = len(remain_list)

        select_index_var = Variable(torch.LongTensor(select_list).cuda())
        remain_index_var = Variable(torch.LongTensor(remain_list).cuda())
        select_score_map = torch.index_select( torch.index_select(matrix_score, 0, select_index_var), 1, remain_index_var ).contiguous().view(-1)

        #select_score_map = matrix_score[select_list][:,remain_list].contiguous().view(-1)
        if config.use_rl and is_training and not_sampled:
            dist = F.softmax(select_score_map, 0)
            greedy_id = select_score_map.max(0)[1]
            best_id = torch.multinomial(dist, 1)[0]
            if int(greedy_id) != int(best_id):
                not_sampled = False
                if config.log_softmax:
                    prob = dist[best_id] + 1e-20
                else:
                    prob = select_score_map[best_id] + 1e-20
                gen_tree_loss_per_batch.append(prob.log())
            #neg_entropy = dist * (dist + 1e-20).log()
            #entropy_loss.append(neg_entropy.sum())
        else:
            _, best_id = select_score_map.max(0)
        #_, best_id = select_score_map.max(0)
        depend_id = int(best_id) // wid
        insert_id = int(best_id) % wid

        best_depend_node = selected_node[depend_id]
        best_insert_node = node_containter[insert_id]
        best_depend_node.add_child(best_insert_node)

        selected_node.append(best_insert_node)
        select_list.append(best_insert_node.index)
        node_containter.remove(best_insert_node)
        remain_list.remove(best_insert_node.index)
    if not_sampled:
        gen_tree_loss_per_batch.append(Variable(torch.FloatTensor([0]).zero_().cuda()))
    return virtual_root

def return_tree_contrainer(num_nodes, batch_id):
    """ Return number of tree nodes """
    container = []
    index_list= []
    for i in range(num_nodes):
        container.append(tree_def.ArbitraryTree(i, im_idx=batch_id))
        index_list.append(i)
    return container, index_list

def ArTree_to_BiTree(arTree):
    root_node = arTree.generate_bi_tree()
    arNode_to_biNode(arTree, root_node)
    assert(root_node.index == -1)
    assert(root_node.right_child is None)
    assert(root_node.left_child is not None)
    return root_node.left_child

def arNode_to_biNode(arNode, biNode):
    if arNode.get_child_num() >= 1:
        new_bi_node = arNode.children[0].generate_bi_tree()
        biNode.add_left_child(new_bi_node)
        arNode_to_biNode(arNode.children[0], biNode.left_child)

    if arNode.get_child_num() > 1:
        current_bi_node = biNode.left_child
        for i in range(arNode.get_child_num() - 1):
            new_bi_node = arNode.children[i+1].generate_bi_tree()
            current_bi_node.add_right_child(new_bi_node)
            current_bi_node = current_bi_node.right_child
            arNode_to_biNode(arNode.children[i+1], current_bi_node)

def generate_overlap_tree(bbox_shape):
    """
    bbox_shape : [batch, num_objs, b_dim]  (x1,y1,x2,y2,w,h)
    Method:
        Iteratively generate a tree:
        1) Select a node with the largest score: num_overlap + /lambda * node_area (/lambda = 1e-10)
        2) According to the bbox center, separate the rest node into left/right subtree
    """
    batch_size, num_objs, _ = bbox_shape.size()
    # calculate information required by generation procedure
    bbox_overlap = overlap(bbox_shape)
    bbox_area = (bbox_shape[:, :, 2] - bbox_shape[:, :, 0]) * (bbox_shape[:, :, 3] - bbox_shape[:, :, 1])
    bbox_center = (bbox_shape[:, :, 0] + bbox_shape[:, :, 2]) / 2.0

    forest = []
    for i in range(batch_size):
        node_container = []
        score_list = []
        # Overlap Matrix -> Binary Matrix -> Num of Overlaped Objects
        overlap_slice = (bbox_overlap[i].view(num_objs, num_objs) > 0).sum(1) 
        for j in range(num_objs):
            # Node score = num_overlap + /lambda * node_area (/lambda = 1e-10)
            node_score = float(overlap_slice[j]) + float(bbox_area[i,j]) * 1e-10
            node_container.append(tree_def.BiTree(j, im_idx=i, node_score=node_score, center_x=float(bbox_center[i,j])))
            score_list.append(node_score)

        root = return_best_node(node_container, score_list)
        root.set_root()
        iterative_gen_tree(node_container, score_list, root)
        forest.append(root)
    return forest


def iterative_gen_tree(node_container, score_list, root):
    """
    Iterativly generate a tree
        (1) Select a root
        (2) Separete the rest nodes into two parts
        (3) Running step one for each parts
    """
    if len(node_container) == 0:
        return
    left_container, left_score, right_container, right_score = seperate_container_by_root(node_container, score_list, root)
    left_root = return_best_node(left_container, left_score)
    right_root = return_best_node(right_container, right_score)
    if left_root is not None:
        root.add_left_child(left_root)
        iterative_gen_tree(left_container, left_score, left_root)
    if right_root is not None: 
        root.add_right_child(right_root)
        iterative_gen_tree(right_container, right_score, right_root) 
    return

def return_best_node(node_container, score_list):
    """
    Given a list of nodes
        (1)Find the node with the largest score
        (2)Remove the selected node
    """
    if len(node_container) == 0:
        return None
    scoreList = torch.FloatTensor(score_list)
    ind = int(scoreList.max(0)[1])
    best_node = node_container[ind]
    return best_node

def seperate_container_by_root(node_container, score_list, root):
    """
    Given a list of nodes
        (1) Seperate the container in to two by root node
        (2) Return left/right container
    """
    left_container = []
    left_score = []
    right_container = []
    right_score = []
    for i in range(len(node_container)):
        if node_container[i].index == root.index:
            continue
        elif node_container[i].center_x < root.center_x:
            left_container.append(node_container[i])
            left_score.append(score_list[i])
        else:
            right_container.append(node_container[i])
            right_score.append(score_list[i])
    return left_container, left_score, right_container, right_score


def overlap(bbox_shape):
    """
    bbox_shape : [batch, num_objs, b_dim]  (x1,y1,x2,y2,w,h)
    """
    batch_size, num_objs, _ = bbox_shape.size()

    min_max_xy = torch.min(bbox_shape[:, :, 2:4].unsqueeze(2).expand(batch_size, num_objs, num_objs, 2),
                           bbox_shape[:, :, 2:4].unsqueeze(1).expand(batch_size, num_objs, num_objs, 2))
    max_min_xy = torch.max(bbox_shape[:, :, :2].unsqueeze(2).expand(batch_size, num_objs, num_objs, 2),
                           bbox_shape[:, :, :2].unsqueeze(1).expand(batch_size, num_objs, num_objs, 2))
    inter = torch.clamp((min_max_xy - max_min_xy), min=0)
    return inter[:, :, :, 0] * inter[:, :, :, 1]

class TreeLSTM_IO(object):
    def __init__(self, num_obj, dropout_mask):
        self.num_obj = num_obj
        self.hidden = None # Float tensor [num_obj, self.out_dim]
        self.order = Variable(torch.LongTensor(num_obj).zero_().cuda()) # Long tensor [num_obj]
        self.order_count = 0 # int
        self.dropout_mask = dropout_mask
    
    def reset(self):
        self.hidden = None # Float tensor [num_obj, self.out_dim]
        self.order = Variable(torch.LongTensor(self.num_obj).zero_().cuda()) # Long tensor [num_obj]
        self.order_count = 0 # int

def block_orthogonal(tensor, split_sizes, gain=1.0):
    """
    An initializer which allows initializing model parameters in "blocks". This is helpful
    in the case of recurrent models which use multiple gates applied to linear projections,
    which can be computed efficiently if they are concatenated together. However, they are
    separate parameters which should be initialized independently.
    Parameters
    ----------
    tensor : ``torch.Tensor``, required.
        A tensor to initialize.
    split_sizes : List[int], required.
        A list of length ``tensor.ndim()`` specifying the size of the
        blocks along that particular dimension. E.g. ``[10, 20]`` would
        result in the tensor being split into chunks of size 10 along the
        first dimension and 20 along the second.
    gain : float, optional (default = 1.0)
        The gain (scaling) applied to the orthogonal initialization.
    """
    sizes = list(tensor.size())
    if any([a % b != 0 for a, b in zip(sizes, split_sizes)]):
        raise ValueError("tensor dimensions must be divisible by their respective "
                         "split_sizes. Found size: {} and split_sizes: {}".format(sizes, split_sizes))
    indexes = [list(range(0, max_size, split))
               for max_size, split in zip(sizes, split_sizes)]
    # Iterate over all possible blocks within the tensor.
    for block_start_indices in itertools.product(*indexes):
        # A list of tuples containing the index to start at for this block
        # and the appropriate step size (i.e split_size[i] for dimension i).
        index_and_step_tuples = zip(block_start_indices, split_sizes)
        # This is a tuple of slices corresponding to:
        # tensor[index: index + step_size, ...]. This is
        # required because we could have an arbitrary number
        # of dimensions. The actual slices we need are the
        # start_index: start_index + step for each dimension in the tensor.
        block_slice = tuple([slice(start_index, start_index + step)
                             for start_index, step in index_and_step_tuples])

        # let's not initialize empty things to 0s because THAT SOUNDS REALLY BAD
        assert len(block_slice) == 2
        sizes = [x.stop - x.start for x in block_slice]
        tensor_copy = tensor.new(max(sizes), max(sizes))
        torch.nn.init.orthogonal_(tensor_copy, gain=gain)
        tensor[block_slice] = tensor_copy[0:sizes[0], 0:sizes[1]].data


def print_tree(tree):
    if tree is None:
        return
    if(tree.left_child is not None):
        print_node(tree.left_child)
    if(tree.right_child is not None):
        print_node(tree.right_child)

    print_tree(tree.left_child)
    print_tree(tree.right_child)

    return
    

def print_node(tree):
    print(' depth: ', tree.depth(), end="")
    print(' score: ', tree.node_score, end="")
    print(' child: ', tree.get_total_child())