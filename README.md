# Inferring Past Human Actions in Homes with Abductive Reasoning
This is our PyTorch Implementation of "Inferring Past Human Actions in Homes with Abductive Reasoning" using the Action Genome dataset. 

## Requirements
We are using:
- Python=3.6.10
- PyTorch=1.10.0
- cudatoolkit=11.3

First, create a new conda environment and install PyTorch 1.10.0. Then, run the requirements.txt file.
```
conda create -n aai python=3.6.10

conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge

pip install -r requirements.txt

```

Next, build the following:
```
cd lib/draw_rectangles
python setup.py build_ext --inplace
cd ..
cd fpn/box_intersections_cpu
python setup.py build_ext --inplace
```

If you encounter any errors regarding the FasterRCNN compilation, compile the cuda dependencies in the fasterRCNN/lib folder. The fasterRCNN is built on top of this [repo](https://github.com/jwyang/faster-rcnn.pytorch/tree/pytorch-1.0). 

```
cd fasterRCNN/lib
python setup.py build develop
```

## Dataset
Download the [Action Genome](https://prior.allenai.org/projects/charades) (scaled to 480p, 13GB) dataset and dump the frames using their toolkit [here](https://github.com/JingweiJ/ActionGenome). 

We use the pre-trained FasterRCNN model checkpoint trained on Action Genome from yrcong's [STTran](https://arxiv.org/abs/2107.12309) paper. Please download it [here](https://github.com/yrcong/STTran). Place it in 
`fasterRCNN/models/faster_rcnn_ag.pth`.

We provide the frame-level action annotations obtained from Charades for the snapshots in AG and placed them in `dataloader/frame_action_list.pkl`.

We also provide the pickle files for the different dataset setups (set == verification and sequence) which do not rely on how our code retrieves the frames. 


## Action Set Training
For MLP, Relational Transformers, Graph Neural Network Encoder Decoder (GNNED), Relational Bilinear Pooling (RBP), or Bilinear Graph Encoder Decoder (BiGED) run:

`python train.py --save_path path/to/save/ --lr 1e-5 --pool_type max --num_frames -1 --lr_scheduler --semantic --model_type *select_model*`

For rule-based inference run:

`python train_relation.py --save_path path/to/save --num_frames -1 --model_type Relational`

`--save_path`: save the model checkpoint and configuration to this location

`--pool_type`: either max or avg - note that GNNED, RBP and BiGED only utilizes max pool by default

`--num_frames`: either 2 or -1. When set to 2, we abduct the set of actions from previous and current snapshot while -1 means we abduct the set of all previous actions - for more details have a look at our paper under the Experimental Setup subsection.

`--lr_scheduler`: toggle lr_scheduler

`--semantic`: toggle whether to use semantics or not - note that GNNED, RBP and BiGED utilizes semantics by default

`--model_type`: either mlp, transformer, GNNED, RBP, BiGED, or Relational

`--cross_attention`: only for transformer model, to select cross-attention instead of self-attention


## Action Set Prediction
The configuration during training is saved as a config.json file. Therefore, during evaluation, set the following paths accordingly:

`python inference.py --model_path path/to/model/checkpoint --conf_path path/to/config.json
--save_path path/to/save/results`

For rule-based inference:

`python test_relation.py --model_path path/to/model/checkpoint --conf_path path/to/config.json --save_path path/to/save/results`

If you are trying to perform inference only on the last frame, add `--infer_last`. 

## Action Sequence Training
The parameters are similar to the action set training procedure. The only difference is that you are required to select the relational model and load its checkpoint. Then, select the sequence model you want to use to decode the sequence of actions.

`python train_sequence.py --num_frames -1 --save_path path/to/save --lr 1e-5 --nepoch 100 --model_type *select_relational_model* --task sequence --semantic --model_path path/to/relational/model --seq_layer 2 --seq_model_mlp_layers 2 --seq_model *select_seq_model* --hidden_dim 1936`


## Action Sequence Prediction
The configuration during training is saved as a config.json file. Therefore, during evaluation, set the following paths accordingly:

`python infer_sequence.py --save_path path/to/save/results --model_path path/to/relational/model --conf_path /path/to/config/file --seq_model_path path/to/sequence/model/checkpoint`

## Action Verfication Training
The parameters are similar to the action set training procedure. Select the sequence model you want to use to perform action verification.

`python train_verification.py --num_frames -1 --save_path path/to/save --lr 1e-5 --nepoch 100 --model_type *select_relational_model* --task verification`


## Action Verification Prediction
The configuration during training is saved as a config.json file. Therefore, during evaluation, set the following paths accordingly:

`python infer_verification.py --save_path path/to/save/results --model_path path/to/relational/model --conf_path /path/to/config/file`


## Acknowledgements
Our code is adapted from yrcong's [STTran](https://github.com/yrcong/STTran).
