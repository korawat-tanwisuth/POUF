
# POUF (Prompt-oriented unsupervised fine-tuning for large pre-trained models)

- [Introduction](#introduction)
- [Instruction](#instruction)
- [Citation](#citation)

## Introduction
*POUF* is an unsupervised fine-tuning framework for adapting prompt-based zero-shot models on the target data.

## Instruction

The code for running the experiments are stored in the path `pouf_demo/examples/pouf/image_classification`.

1. Run `pip install -r requirements.txt`. 
2. Run `cd pouf_demo/examples/pouf/image_classification`. 
2. Run the following code to reproduce the result for model tuning and prompt tuning. Alternatively, run `pouf.sh`.
   - Model tuning
     > python pouf.py data/office31 -d Office31 -s A -t W --epochs 10 --seed 1 --log logs/pouf/Office31_A2W
   - Prompt tuning  
     > python pouf.py data/office31 -d Office31 -s A -t W --epochs 10 --seed 1 --log logs/pouf/Office31_A2W --learn-prompt -plr 0.1
    

## Citation

We adapt our code from the following codebases:

**TLLIB**
```bibtex
@misc{tllib,
author = {Junguang Jiang, Baixu Chen, Bo Fu, Mingsheng Long},
title = {Transfer-Learning-library},
year = {2020},
publisher = {GitHub},
journal = {GitHub repository},
}
```

**CLIP**
```bibtex
@inproceedings{radford2021learning,
title={Learning transferable visual models from natural language supervision},
author={Radford, Alec and Kim, Jong Wook and Hallacy, Chris and Ramesh, Aditya and Goh, Gabriel and Agarwal, Sandhini and Sastry, Girish and Askell, Amanda and Mishkin, Pamela and Clark, Jack and others},
booktitle={International Conference on Machine Learning},
pages={8748--8763},
year={2021},
organization={PMLR}
}
```

**CoOp**
```bibtex
@article{zhou2022learning,
title={Learning to prompt for vision-language models},
author={Zhou, Kaiyang and Yang, Jingkang and Loy, Chen Change and Liu, Ziwei},
journal={International Journal of Computer Vision},
volume={130},
number={9},
pages={2337--2348},
year={2022},
publisher={Springer}
}
```
