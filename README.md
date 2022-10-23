# Short text clustering


This repository contains the code used in EduQG paper:

[EduQG: A Multi-format Multiple Choice Dataset for the Educational Domain](https://arxiv.org/abs/2210.06104)

If you use the dataset please cite:  

```  
@misc{2210.06104,
Author = {Amir Hadifar and Semere Kiros Bitew and Johannes Deleu and Chris Develder and Thomas Demeester},
Title = {EduQG: A Multi-format Multiple Choice Dataset for the Educational Domain},
Year = {2022},
Eprint = {arXiv:2210.06104},
}
```


### Pre-requisites ###

> pip install -r requirements.txt 


#### pretrain from scratch ###

<pre>
sh run_qg_exp.sh
</pre>

### Load ans-agnostic models:
```
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("hadifar/openstax_qg_agno")

model = AutoModelForSeq2SeqLM.from_pretrained("hadifar/openstax_qg_agno")
```

### Load ans-aware models
```
comming soon...
```

