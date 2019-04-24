
## Git Branches

**master branch**: Original DQN implemented in this [repo](https://github.com/metalbubble/DeepRL-Tutorials)  
**knowledge_distill branch**: Implementation of knowledge distillation method.  
**multi_dpn branch**: Implementation of multi_dqn and joint training with knowledge distillation method.  

## Running Environment
Pytho 3.6 && Pytorch 0.4.0

## Training and Testing
Downloading pretrained weights from [here](https://drive.google.com/file/d/1Qg_U0MEQLTiv91YXqljqjZCQbJV-Bbsj/view?usp=sharing). Unzip it and put it in the root directory of the project.  
**Train**: Run `sh train.sh` in root directory of the project.  
**Test**: `sh test.sh` in root directory of the project.  
The trained model weights are stored in *saved_agents/* directory.  
The training and testing logs are store in *log/* directory.  