from __future__ import print_function

import errno
import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn


class BasicBiTree(object):
    def __init__(self, idx, is_root=False):
        self.index = int(idx)
        self.is_root = is_root
        self.left_child = None
        self.right_child = None
        self.parent = None
        self.num_child = 0

    def set_root(self):
        self.is_root = True

    def add_left_child(self, child):
        if self.left_child is not None:
            print('Left child already exist')
            return
        child.parent = self
        self.num_child += 1
        self.left_child = child
    
    def add_right_child(self, child):
        if self.right_child is not None:
            print('Right child already exist')
            return
        child.parent = self
        self.num_child += 1
        self.right_child = child

    def get_total_child(self):
        sum = 0
        sum += self.num_child
        if self.left_child is not None:
            sum += self.left_child.get_total_child()
        if self.right_child is not None:
            sum += self.right_child.get_total_child()
        return sum

    def depth(self):
        if hasattr(self, '_depth'):
            return self._depth
        if self.parent is None:
            count = 1
        else:
            count = self.parent.depth() + 1
        self._depth = count
        return self._depth

    def max_depth(self):
        if hasattr(self, '_max_depth'):
            return self._max_depth
        count = 0
        if self.left_child is not None:
            left_depth = self.left_child.max_depth()
            if left_depth > count:
                count = left_depth
        if self.right_child is not None:
            right_depth = self.right_child.max_depth()
            if right_depth > count:
                count = right_depth
        count += 1
        self._max_depth = count
        return self._max_depth

class ArbitraryTree(object):
    def __init__(self, idx, im_idx=-1, is_root=False):
        self.index = int(idx)
        self.is_root = is_root
        self.children = []
        self.im_idx = int(im_idx) # which image it comes from
        self.parent = None
    
    def generate_bi_tree(self):
        # generate a BiTree node, parent/child relationship are not inherited
        return BiTree(self.index, im_idx=self.im_idx, is_root=self.is_root)

    def add_child(self, child):
        child.parent = self
        self.children.append(child)

    def print(self):
        print('====================')
        print('is root: ', self.is_root)
        print('index: ', self.index)
        print('num of child: ', len(self.children))
        for node in self.children:
            node.print()
    
    def find_node_by_index(self, index, result_node):
        if self.index == index:
            result_node = self
        elif len(self.children) > 0:
            for i in range(len(self.children)):
                result_node = self.children[i].find_node_by_index(index, result_node)
                
        return result_node

    def search_best_insert(self, matrix_score, insert_node, best_score, best_depend_node, best_insert_node):
        # virtual node will not be considerred
        if self.is_root:
            pass
        elif float(matrix_score[self.index, insert_node.index]) > float(best_score):
            best_score = matrix_score[self.index, insert_node.index]
            best_depend_node = self
            best_insert_node = insert_node
        
        # iteratively search child
        for i in range(self.get_child_num()):
            best_score, best_depend_node, best_insert_node = \
                self.children[i].search_best_insert(matrix_score, insert_node, best_score, best_depend_node, best_insert_node)

        return best_score, best_depend_node, best_insert_node

    def get_child_num(self):
        return len(self.children)
    
    def get_total_child(self):
        sum = 0
        num_current_child = self.get_child_num()
        sum += num_current_child
        for i in range(num_current_child):
            sum += self.children[i].get_total_child()
        return sum

# only support binary tree
class BiTree(BasicBiTree):
    def __init__(self, idx, im_idx, is_root=False, node_score=0.0, center_x=0.0):
        super(BiTree, self).__init__(idx, is_root)
        self.node_score = float(node_score)
        self.center_x = float(center_x)
        self.im_idx = int(im_idx) # which image it comes from
