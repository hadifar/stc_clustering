# Short text clustering


This repository contains the code presented in rep4NLP 2019:

[A Self-training Approach for Short Text Clustering](https://sites.google.com/view/repl4nlp2019/accepted-papers?authuser=0)

If you use part of the code please cite:  

```  
@InProceedings{hadifar2019stc,
  author = 	"Hadifar, Amir
		and Sterckx, Lucas
		and Demeester, Thomas
		and Develder, Chris",
  title = 	"A Self-Training Approach for Short Text Clustering",
  booktitle = 	"Representation Learning for NLP workshop (Rep4NLP), ",
  year = 	"2019",
}
```


### Pre-requisites ###

> pip install -r requirements.txt 


#### Reproduce results for stackoverflow ###

<pre> python STC.py --maxiter 1500 --ae_weights data/stackoverflow/results/ae_weights.h5 --save_dir data/stackoverflow/results/
</pre>

if you have used the datasets, please cite the following paper:

```
@article{xu2017self,
title={Self-Taught Convolutional Neural Networks for Short Text Clustering},
author={Xu, Jiaming and Xu, Bo and Wang, Peng and Zheng, Suncong and Tian, Guanhua and Zhao, Jun and Xu, Bo},
journal={Neural Networks},    
volume={88},
pages={22-31},
year={2017}
}
```

#### Acknowledge

This code is based on repo from [here](https://github.com/XifengGuo/DEC-keras).
