# Short text clustering


This repository contains the code presented in the work:

[Short text clustering](https://github.com/hadifar)

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


#### Reproduce results for search snippets ###

<code> python STC.py --maxiter 100 --ae_weights data/search_snippets/results/ae_weights.h5 --save_dir data/search_snippets/results/
</code>

if you have used the datasets, please also cite the following paper:

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

This code is based on code from [here](https://github.com/XifengGuo/DEC-keras).