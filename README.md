# AIRNet
This is the official implementation of AIRNet, as proposed in our upcoming publication.

# Codes
- models: Containing the codes for model construction.
  - layers.py: Basic network layers.
  - modules.py: Network blocks.
  - losses.py: Loss functions.

- dataloader: For CII-Cutmix data augmentation.
  - imaging.py: Data transformation methods.
  - cutmix.py: For CutMix augmentation.
  - mixup.py: For mixup augmentation.

- configs.py: This is for the model and its training configuration.
  
- main.py: The main code for model training and testing.
  
- trainer.py: Containing the functions for model training and testing.
  - TSINet3trainer: model training.
  - evaluator: model evaluation and testing.

# For training
Just change the training configuration in 'configs.py' and 'main.py'.
