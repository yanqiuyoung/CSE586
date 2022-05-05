# CSE586
Course project - Apple Foliar Disease Classification

This repository contains training file for ResNet-9 pre-trained on PlantVillage dataset and fine-turned on Plant Pathology dataset
##############################PRE-REQUEST##############################

To correctly run the test model, the following packages are required.
1.	Python 3.8
2.	py-torch 1.9.0
3.	openCV

##############################Training##################################

run resnet9.ipynb to save the pre-trained weights
run python train.py for transfer learning
Arguments:
--train_dir = data_dir + "train"
--valid_dir = data_dir + "valid"
--batchsize: the batch size
--lr: initial learning rate
--sched: scheduler for one cycle learniing rate

##############################Testing###################################

run python test.py to get the test results
run Draw_the_Figure.ipynb to get the visualization
