# Arbitrary-Order-Infnet
Code to train model from "[An Exploration of Arbitrary-Order Sequence Labeling via Energy-Based Inference Networks](https://arxiv.org/pdf/2010.02789.pdf)", accept by EMNLP 2020.

In this work, we propose several high-order energy terms to capture complex dependencies among labels in sequence labeling, including several that consider the entire label sequence. We use neural parameterizations for these energy terms, drawing from convolutional, recurrent, and selfattention networks. We use the framework of learning [energy-based inference network (Tu and Gimpel, 2018)](https://arxiv.org/abs/1803.03376) for dealing with the difficulties of training and inference with such models.


## Some Examples to Train High-Order models

Skip-Chain Energies
```
python BertInfNet.py --Wyy_form decom --M 3
```


High-Order Energies With Vectorized Kronecker Product Parameterization
```
python BertInfNet.py --Wyy_form horderm --M 3
```

High-Order Energies With CNN Parameterization
```
python BertInfNet.py --Wyy_form cnn --M 1
```

High-Order Energies With Tag Language Model Parameterization
```
python BertInfNet.py --Wyy_form taglm-ylog --M 1
```

High-Order Energies With Self-Attention Parameterization
```
python BertInfNet.py --Wyy_form selfatt --M 5
```



## References
```
@inproceedings{tu-etal-2020-exploration,
    title = "{A}n {E}xploration of {A}rbitrary-{O}rder {S}equence {L}abeling via {E}nergy-{B}ased {I}nference {N}etworks",
    author = "Tu, Lifu  and
      Liu, Tianyu  and
      Gimpel, Kevin",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-main.449",
    doi = "10.18653/v1/2020.emnlp-main.449",
    pages = "5569--5582",
    abstract = "Many tasks in natural language processing involve predicting structured outputs, e.g., sequence labeling, semantic role labeling, parsing, and machine translation. Researchers are increasingly applying deep representation learning to these problems, but the structured component of these approaches is usually quite simplistic. In this work, we propose several high-order energy terms to capture complex dependencies among labels in sequence labeling, including several that consider the entire label sequence. We use neural parameterizations for these energy terms, drawing from convolutional, recurrent, and self-attention networks. We use the framework of learning energy-based inference networks (Tu and Gimpel, 2018) for dealing with the difficulties of training and inference with such models. We empirically demonstrate that this approach achieves substantial improvement using a variety of high-order energy terms on four sequence labeling tasks, while having the same decoding speed as simple, local classifiers. We also find high-order energies to help in noisy data conditions.",
}
```
