# paths
qa_path = 'data'  # directory containing the question and annotation jsons
bottom_up_trainval_path = 'data/trainval'  # directory containing the .tsv file(s) with bottom up features
bottom_up_test_path = 'data/test2015'  # directory containing the .tsv file(s) with bottom up features
preprocessed_trainval_path = 'genome-trainval.h5'  # path where preprocessed features from the trainval split are saved to and loaded from
#preprocessed_trainval_path =  '/media/tangkaihua/Disk1/vqa/genome-trainval36.h5' # path where preprocessed features from the trainval split are saved to and loaded from
preprocessed_test_path = 'genome-test.h5'  # path where preprocessed features from the test split are saved to and loaded from
#preprocessed_test_path = '/media/tangkaihua/Disk1/vqa/genome-test.h5'  # path where preprocessed features from the test split are saved to and loaded from
vocabulary_path = 'vocab.json'  # path where the used vocabularies for question and answers are saved to

task = 'OpenEnded'
dataset = 'mscoco'

test_split = 'test2015'  # either 'test-dev2015' or 'test2015'

# preprocess config
output_size = 100  # max number of object proposals per image
output_features = 2048  # number of features in each object proposal

# training config
epochs = 200
batch_size = 256
initial_lr = 1.5e-3
lr_halflife = 50000  # in iterations
data_workers = 4
max_answers = 3000

# model config
# the method we used to generate trees
#gen_tree_mode = "overlap_tree"
gen_tree_mode = "arbitrary_trees_transfer"
# tree lstm hidden pool out mode, 
#poolout_mode = "sigmoid" 
poolout_mode = "softmax"
use_rl = True
log_softmax = False