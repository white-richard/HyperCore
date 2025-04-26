# HyperCore

**HyperCore** is a library built upon PyTorch to easily write and train hyperbolic foundation models for a wide range of applications in diverse modalities. [PAPER LINK](https://arxiv.org/abs/2504.08912)

We provide the core modules and functionalities that makes this process simple for users of all levels of backgrounds in differential geometry. These include methods and algorithms for hyperbolic neural networks and foundation models, optimization techiniques, and manifold operations. These come together to enable intuitive constructions of hyperbolic foundation models architectures and pipelines, e.g. [hyperbolic Transformer encoder](example_usage/Hyperbolic_Transformers/lorentz_transformer.py), [hyperbolic ViT](example_usage/Hyperbolic_Transformers/vision_transformer.py), [hyperbolic fine-tuning](example_usage/Fine-Tuning/fine_tuning_example.py), [hyperbolic GraphRAG](example_usage/Hyperbolic_GraphRAG/graphRAG_example.py) and much more (see our paper and tutorials below!). 

- [Library Highlights](#library-highlights)
- [Installation](#installation)
- [Quick Start: Build Hyperbolic Foundation Models](#quick-start-build-hyperbolic-foundation-models)
- [Library Overview](#library-overview)
- [Implemented Modules and Details](#implemented-modules-and-details)


## Library Highlights
HyperCore is accessible to experts in hyperbolic deep learning, the more general AI audience, and first-time user of deep learning toolkits alike. Here are some reasons you might want to use HyperCore for building, training, or using foundation models in hyperbolic space!

- **Flexible and Intuitive Foundation Model Support:** HyperCore it is capable of doing much more than reproducing existing models—its components can be effortlessly combined to construct novel hyperbolic foundation models that have yet to be proposed. See [example usage](example_usage) for extensive examples of how to build hyperbolic foundation models with HyperCore!
- **Easy-to-use Modules:** As the API is designed almost identically to Euclidean counterparts, users only need a high-level understanding of foundation models to build hyperbolic ones using HyperCore. All it takes is a few lines to create hyperbolic foundation models components like fully hyperbolic Transformer encoder blocks (see [here](#quick-start-build-hyperbolic-foundation-models) for a quick start guide)!
- **Comprehensive Modules and Model Support:** Unlike anything out there, hyperCore provides a comprehensive and extensive list of essential hyperbolic modules for building a wide range of hyperbolic foundation models for learning across diverse modalities. 

## Installation
For now, the dependencies for HyperCore can be installed via 
```
pip install -r requirements.txt
```
Installation via pip directly is ***coming soon...***

## Quick Start: Build Hyperbolic Foundation Models
In this quick start guide, we highlight the ease of creating and training a hyperbolic foudnation model model with HyperCore.
### Creating Your Own Hyperbolic Transformer Encoder Block
In the first glimpse of HyperCore, we build the encoder block of a hyperbolic Transformer, using hyperbolic-tailored modules such as [LorentzMultiheadAttention](hypercore/nn/attention/lorentz_former_conv.py) for multiple attention and [LResNet](hypercore/nn/conv/conv_util_layers.py) for residual connection. 

```python
import torch
import torch.nn as nn
import hypercore.nn as hnn

class LTransformerBlock(nn.Module):
    def __init__(self, manifold, d_model: int, n_head: int):
        super().__init__()
        dim_per_head = d_model // n_head
        out_dim = d_model - 1
        mlp_dim = d_model * 4 - 1
        self.manifold = manifold
        self.attn = hnn.LorentzMultiheadAttention(manifold, dim_per_head, dim_per_head, n_head, attention_type='full', trans_heads_concat=True)
        self.ln_1 = hnn.LorentzLayerNorm(manifold, out_dim)
        self.mlp = nn.Sequential(
            hnn.LorentzLinear(manifold, d_model, mlp_dim),
            hnn.LorentzActivation(manifold, activation=nn.GELU()),
            hnn.LorentzLinear(manifold, mlp_dim+1, out_dim),
        )
        self.ln_2 = hnn.LorentzLayerNorm(manifold, out_dim)
        self.res1 = hnn.LResNet(manifold, use_scale=True)
        self.res2 = hnn.LResNet(manifold, use_scale=True)

    def forward(self, x, attn_mask=None):
        # x: embedded features matrix of shape [batch, seq_len, dim_per_head * num_heads]
        lx = self.ln_1(x)
        ax = self.attn(lx, lx, mask=attn_mask)
        x = self.res1(x, ax)
        x = self.res2(x, self.mlp(self.ln_2(x)))
        return x
```

For more examples of how to employ hyperbolic foundation model counponents in downstream tasks, please see [example usages](example_usage)

## Library Overview

## Implemented Modules and Details
