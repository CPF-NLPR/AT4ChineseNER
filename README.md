# AT4ChineseNER
This is the source code for the paper ''Adversarial Transfer Learning for Chinese Named Entity Recognition with Self-Attention Mechanism'' accepted by EMNLP2018. The paper can be download from http://aclweb.org/anthology/D18-1017. 
## Requirements
  * TensorFlow >= v1.2.0
  * numpy
  * python 2.7
## Usage
### Download datasets
Please download the [WeiboNER dataset](https://github.com/hltcoe/golden-horse/tree/master/data), [SighanNER dataset](http://sighan.cs.uchicago.edu/bakeoff2006/) and [MSR dataset](http://sighan.cs.uchicago.edu/bakeoff2005/), respectively. The dataset files are put in `data` directory.
### Train model
For training the model on WeiboNER dataset, you need to type the following commands:
 * python preprocess_weibo.py
 * python train_weibo.py

For SighanNER dataset, the operation is similar. 
### Test model
We have provided our best model on the original WeiboNER dataset in the `ckpt` directory. You just run the model like:
 * python preprocess_weibo.py
 * python test_weibo.py
 
 In addition, if you adjust certain hyper-parameters and train the model, you can test the model with restoring certain checkpoint.
 ## Citation
If you use the code, please cite this paper:

Pengfei Cao, Yubo Chen, Kang Liu, Jun Zhao. Adversarial Transfer Learning for Chinese Named Entity Recognition with Self-Attention Mechanism. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing (EMNLP2018).
