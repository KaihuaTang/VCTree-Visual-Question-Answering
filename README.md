# VCTree-Visual-Question-Answering
Code for the Visual Question Answering (VQA) part of CVPR 2019 oral paper: "[Learning to Compose Dynamic Tree Structures for Visual Contexts][0]"

The code is directly modified from that project [Cyanogenoid/vqa-counting][1]. We mainly modified the model.py, train.py, config.py and add several files about our VCTree model, such as all tree_*.py, gen_tree_net.py.  Before we got our final model, we tried lots of different tree structures, hence you may found some strange code such as config.gen_tree_mode and the corresponding choices in tree_feature.py. Just ignore them. (I'm too lazy to purge the code, sorry about that)

## Dependencies
This code was confirmed to run with the following environment:

- Python 3.6
  - torch 0.4
  - torchvision 0.2
  - h5py 2.7
  - tqdm 4.19

# Prepare data
Please follow [Instruction][4] to prepare data. 

- In the `data` directory, execute `./download.sh` to download VQA v2 [questions, answers, and bottom-up features][4].
  - For experimenting, using 36 fixed proposals is faster, at the expense of a bit of accuracy. Uncomment the relevant lines in `download.sh` and change the paths in `config.py` accordingly. Don't forget to set `output_size` in there to 36 to actually get the speed-up.
- Prepare the data by running
```
python preprocess-images.py
python preprocess-vocab.py
```
This creates an `h5py` database (95 GiB) containing the object proposal features and a vocabulary for questions and answers at the locations specified in `config.py`.
- Download the pretrained object correlation score models
    - The proposed VCTree requires pretrained model to generate object correlation score f(xi, xj) as we mentioned in the Section3.1, such a pretrained model can be downloaded from [vgrel-19 (for 10-100 bounding box features)](https://onedrive.live.com/embed?cid=22376FFAD72C4B64&resid=22376FFAD72C4B64%21620273&authkey=AKFuFsQ90tQO4q0), [vgrel-29 (for 36 bounbing box feautures)](https://onedrive.live.com/embed?cid=22376FFAD72C4B64&resid=22376FFAD72C4B64%21620229&authkey=APSqYLYmGyfl3Mg). Since there are two different types of bottom-up-top-down features, we also have 2 pretrained object correlation score models. The object correlation score model is trained based on the faster-RCNN model (fixed) from [bottom-up-top-down model][2] and the code from [zjuchenlong(./vqa/feature_extractor/bottom_up_origin.py)][3]
    - Put corresponding models under  ./data  and change the name to vgrel-29.tar or vgrel-19.tar depending on your config.output_size

# Train your model
Note that the proposed hybird learning strategy needs to manually iteratively change the config.use_rl = False or True and use -resume to load model from previous stage (which is quite stupid). So you can just first start with config.use_rl = False

The rest instruction is similar to original project [Cyanogenoid/vqa-counting][1]

- Train the model in `model.py` with:
```
python train.py [optional-name]
```
This will alternate between one epoch of training on the train split and one epoch of validation on the validation split while printing the current training progress to stdout and saving logs in the `logs` directory.
The logs contain the name of the model, training statistics, contents of `config.py`,  model weights, evaluation information (per-question answer and accuracy), and question and answer vocabularies.
- To view training progression of a model that is currently or has finished training.
```
python view-log.py <path to .pth log>
```

- To evaluate accuracy (VQA accuracy and balanced pair accuracy; see paper for details) in various categories, you can run
```
python eval-acc.py <path to .pth log> [<more paths to .pth logs> ...]
```
If you pass in multiple paths as arguments, this gives you standard deviations as well.
To customise what categories are shown, you can modify the "accept conditions" for categories in `eval-acc.py`.


# Sometime You Need To Know
- Currently, the default setting is what I used to train my model reported in [Learning to Compose Dynamic Tree Structures for Visual Contexts][0]. However, since the model takes lots of epoches (about 80-100) to converge. It may takes a long time, so I didn't try too many hyperparameters. After the CVPR deadline, I found that using larger size of hidden dimension at some places may further improve the performance a little bit.
- The current training strategy of our VQA model is following [Learning to Count Objects in Natural Images for Visual Question Answering][5] (simple Linear + optim.Adam + continues decay at each batch + large number of epoches). However, we found that using an alternative Strategy (WeightNorm Linear + optim.Adamax + lr warm-up) will only take no more than 15 epoches to converge. So you can try this learning strategy if you are interested. You may check my [another project][6] 


[0]: https://arxiv.org/abs/1812.01880
[1]: https://github.com/Cyanogenoid/vqa-counting
[2]: https://github.com/peteanderson80/bottom-up-attention
[3]: https://github.com/zjuchenlong/faster-rcnn.pytorch
[4]: https://github.com/Cyanogenoid/vqa-counting/tree/master/vqa-v2
[5]: https://openreview.net/forum?id=B12Js_yRb
[6]: https://github.com/KaihuaTang/VQA2.0-Recent-Approachs-2018.pytorch