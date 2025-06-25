---
layout: post
category: research
title: refactoring the TUnA repo
---

### writing beautiful code
My coding was experience very limited before my PhD--I used a few Python scripts to automate a AutoDock-Vina docking pipeline. I was the only person working on the code, so I never considered how code should be written for other people to read and edit.

When I started my PhD, I watched and read enough content about programming that I knew I shouldn't write code this way. So for my first PhD project, I attempted to write "clean and maintainable code." I say **attempt** because looking at it now, I can see several things that can be greatly improved.

I'm going to explain what my project was, what needed to be included in the repository, and what my original approach was.

### what is TUnA?
My first PhD project, TUnA[^1], is based on a new method for predicting protein-protein interactions from sequence alone. In short, TUnA uses a combination of ESM2 embeddings, a Transformer block, and a Gaussian Process layer to output a binary label.

### why is the TUnA repo a mess?
However, because we wanted to compare TUnA to other methods, we also had to include the code for those models and their results. We also used two different datasets for benchmarking.

So I was not sure how I should set up the repo such that it holds both the code to reproduce all the results of the paper and also have the TUnA model be easily accessible, fine-tuned, and used for inference.

In the end, I ended up making the repo about just reproducing the results, without much thought into how people might want to use it beyond that.

Let me just describe the state of the repo:
The main code is stored in a `/results`. Inside, we split into the two datasets. Within each dataset, we have another directory for each model type. Within each model directory, we have the config, training loop, utils, and inference code.

So even for reproducing the results, you have to navigate to a specific model's subdirectory, and run the train.py from within. A lot of the code is repeated and I don't think it's very easy to understand the structure of this repo.

### so let's fix it!
It's been about one year since the original repo was published and I definitely have a better idea for organizing this repo. Like I mentioned, the two main priorities are that users should be able to:
1. Reproducing the results
2. And use any model for their own purposes (e.g. fine-tune, inference)

#### reproducing results
Let's start with the first part. Instead of having to navigate to every subdirectory and running training, I want every result to be reproducible from the project root. There will be a `/src` directory, that contain the code for the different models. A config file will allow users to specify which model and which dataset they want to run, as well as control model-specific hyperparameters.

I've only recently started using PyTorch Lightning and have been really enjoying the quality of life improvements it provides. So needless to say, I'm going to use Lightning, do logging with WandB, and config/hyperparameter management with Hydra (shoutout to Kapil for introducing me to this). 

#### improving usability
About two months ago, someone asked me about how TUnA can be actually used for inference[^2]. It was only then I realized I can greatly improve how users can interact with the model, namely making new predictions. Like training, we can have inference code in the project root that can also be controlled by a config file. This will be the first obvious change.

Initially, for reproduction, generating ESM2 embeddings, which are used as inputs to TUnA, were treated as a pre-processing step, that users must run before trying to reproduce any results. While this works for this purpose, for inference on new proteins, I will need to set up code so that users can simply pass in an amino acid sequence and the embeddings will be generated on the fly. This removes this additional step that can cause friction for someone trying to quickly use this tool.

So essentially, I want a command line interface that be used like:
```bash
python predict.py proteinA_seq proteinB_seq
```
or for batch predictions, you can pass in a csv file that contains two rows of protein sequences.

### lets implement these changes
After I make these changes, I'll follow up on what worked and document any new issues I run into, and what I did to resolve them. Oh, I almost forgot, I'm going to use uv[^3], because that's another new technology I want to get more familiar with.

---
{: data-content="footnotes"}
[^1]: [TUnA: an uncertainty-aware transformer model for sequence-based protein–protein interaction prediction](https://academic.oup.com/bib/article/25/5/bbae359/7720609)
[^2]: Thanks [wangxinfeng-w](https://github.com/Wang-lab-UCSD/TUnA/issues/1) for raising the first issue on my repo!
[^3]: Resolving dependencies in conda envs can take literal hours. Hopefully [uv](https://docs.astral.sh/uv/) makes this process smoother for everyone.