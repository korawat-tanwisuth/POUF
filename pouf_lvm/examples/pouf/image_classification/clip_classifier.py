"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
from typing import Tuple, Optional, List, Dict
import numpy as np
import torch.nn as nn
import torch
import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()
__all__ = ['ClipClassifier']


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

class PromptLearner(nn.Module):
    def __init__(self, classnames, clip_model, device, ctx_init="a photo of a"):
        super().__init__()
        n_cls = len(classnames)
        ctx_init = ctx_init
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = clip_imsize

       
        # use given words to initialize context vectors
        ctx_init = ctx_init.replace("_", " ")
        n_ctx = len(ctx_init.split(" "))
        prompt = clip.tokenize(ctx_init).to(device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(prompt).type(dtype)
        ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
        prompt_prefix = ctx_init

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctx,     # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )

        return prompts

class ClipClassifier(nn.Module):
    """A generic Classifier class for domain adaptation.

    Args:
        backbone (torch.nn.Module): Any backbone to extract 2-d features from data
        num_classes (int): Number of classes
        bottleneck (torch.nn.Module, optional): Any bottleneck layer. Use no bottleneck by default
        bottleneck_dim (int, optional): Feature dimension of the bottleneck layer. Default: -1
        head (torch.nn.Module, optional): Any classifier head. Use :class:`torch.nn.Linear` by default
        finetune (bool): Whether finetune the classifier or train from scratch. Default: True

    .. note::
        Different classifiers are used in different domain adaptation algorithms to achieve better accuracy
        respectively, and we provide a suggested `Classifier` for different algorithms.
        Remember they are not the core of algorithms. You can implement your own `Classifier` and combine it with
        the domain adaptation algorithm in this algorithm library.

    .. note::
        The learning rate of this classifier is set 10 times to that of the feature extractor for better accuracy
        by default. If you have other optimization strategies, please over-ride :meth:`~Classifier.get_parameters`.

    Inputs:
        - x (tensor): input data fed to `backbone`

    Outputs:
        - predictions: classifier's predictions
        - features: features after `bottleneck` layer and before `head` layer

    Shape:
        - Inputs: (minibatch, *) where * means, any number of additional dimensions
        - predictions: (minibatch, `num_classes`)
        - features: (minibatch, `features_dim`)

    """

    def __init__(self, model: nn.Module, class_names: List[str], num_classes: int, learn_prompt: bool, bottleneck: Optional[nn.Module] = None, bottleneck_dim: Optional[int] = -1, head: Optional[nn.Module] = None, finetune=True, pool_layer=None, device=None):
        super(ClipClassifier, self).__init__()
        self.learn_prompt = learn_prompt
        self.class_names = [" ".join(c.split("_")) for c in class_names]
        self.model = model
        self.text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in self.class_names]).to(device) 
        self.logit_scale = model.logit_scale
        self.num_classes = num_classes
        if pool_layer is None:
            self.pool_layer = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                nn.Flatten()
            )
        else:
            self.pool_layer = pool_layer
        if bottleneck is None:
            self.bottleneck = nn.Identity()
            self._features_dim = model.visual.output_dim
        else:
            self.bottleneck = bottleneck
            assert bottleneck_dim > 0
            self._features_dim = bottleneck_dim
        
        self.image_bottleneck = nn.Identity()
        self.text_bottleneck = nn.Identity()
        self.finetune = finetune

        if self.learn_prompt:
            self.model.text_encoder = TextEncoder(model)
            self.prompt_learner = PromptLearner(self.class_names, self.model, device)
            self.turn_off_model_gradient()

    @property
    def features_dim(self) -> int:
        """The dimension of features before the final `head` layer"""
        return self._features_dim

    def turn_off_model_gradient(self):
        for name, param in self.model.named_parameters():
            if "logit" not in name:
                param.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """"""
        image_features = self.image_bottleneck(self.model.encode_image(x)) 
        f = image_features
 
        if self.learn_prompt:
            prompts = self.prompt_learner()
            text_features = self.text_bottleneck(self.model.text_encoder(prompts, self.prompt_learner.tokenized_prompts)                            )
        else:
            text_features = self.text_bottleneck(self.model.encode_text(self.text_inputs))

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        sim = image_features @ text_features.T
        if self.training:
            return sim, f
        else:
            return sim 

    def get_parameters(self, base_lr=1.0, prompt_lr=5e-4, model_lr=5e-5) -> List[Dict]:
        """A parameter list which decides optimization hyper-parameters,
            such as the relative learning rate of each layer
        """
        if self.learn_prompt:
            params = [
                    {"params": self.prompt_learner.parameters(), "lr": prompt_lr * base_lr},
                    {"params": self.logit_scale, "lr": model_lr * base_lr}
            ]
        else:
            params = [
                {"params": self.model.parameters(), "lr": model_lr * base_lr if self.finetune else 1.0 * base_lr},
                {"params": self.image_bottleneck.parameters(), "lr": 1.0 * base_lr},
                {"params": self.text_bottleneck.parameters(), "lr": 1.0 * base_lr },
            ] 
        return params
