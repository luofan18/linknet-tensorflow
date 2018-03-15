## A TensorFlow implementation of LinkNet


#### Dependencies
1. All dependencies are indicated in `env.txt`
2. Download weights of `Resnet-18` and put into `code/`
3. Link to weights of `Resnet-18`
    https://d2j0dndfm35trm.cloudfront.net/resnet-18.t7
   The weights will be automatically loaded into respective layers in the beginning of training

#### Instructions
1. This example runs on cityscapes dataset, and data are prepared used official scripts. Link to cityscapes dataset https://www.cityscapes-dataset.com/
2. Put your data into `data/`   
3. All scripts are in `code/`  
The label mapping is in `labels.py`
Run `python segmentation.py` to train a network, 5 latest checkpoint will be kept during train  
If you prefer using Jupyter Notebook, you may look into `segmentation.ipynb`

#### Reference

1. LinkNet: Exploiting Encoder Representations for Efficient Semantic Segmentation https://arxiv.org/abs/1707.03718  
2. Pytorch implimentation from the author https://github.com/e-lab/pytorch-linknet  
3. Torch implementation of the author (in Lua) https://github.com/e-lab/LinkNet  
4. Enet in Tensorflow https://github.com/kwotsin/TensorFlow-ENet