# Copyright (c) 2019 Shigeki Karita
#               2020 Mobvoi Inc (Binbin Zhang)
#               2022 Xingchen Song (sxc19@mails.tsinghua.edu.cn)
#               2024 Alan (alanfangemail@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Multi-Head Attention layer definition with lora."""

from typing import Optional, List

import torch
from torch import nn

from wenet.transformer.attention import (MultiHeadedAttention,
                                         RelPositionMultiHeadedAttention)
import wenet.finetune.lora.layers as lora


class LoRAMultiHeadedAttention(MultiHeadedAttention):
    """Multi-Head Attention layer with lora.

    Args:
        *args: Arguments for the MultiHeadedAttention.
        **kwargs: Keyword arguments for the MultiHeadedAttention.
        lora_rank (int): Rank for LoRA.
        lora_alpha (int): Alpha for LoRA.
        lora_dropout (float): Dropout rate for LoRA.
        lora_list (Optional[List[str]]): List of layers to apply LoRA.
    """

    def __init__(
        self,
        *args,
        lora_rank: int = 8,
        lora_alpha: int = 8,
        lora_dropout: float = 0.0,
        lora_list: Optional[List[str]] = None,
        **kwargs
    ):
        """Construct an MultiHeadedAttention object."""
        super().__init__(*args, **kwargs)

        n_feat = args[1]  # Assuming the second argument is n_feat
        n_head = args[0]  # Assuming the first argument is n_head
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head

        self.linear_out = lora.Linear(
            n_feat,
            n_feat,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout
        ) if lora_list and "o" in lora_list else nn.Linear(n_feat, n_feat)

        lora_qkv_dict = {
            "q": lora_list and "q" in lora_list,
            "k": lora_list and "k" in lora_list,
            "v": lora_list and "v" in lora_list
        }
        bias_dict = {
            "q": kwargs.get('query_bias', True),
            "k": kwargs.get('key_bias', True),
            "v": kwargs.get('value_bias', True)
        }

        for key, value in lora_qkv_dict.items():
            setattr(
                self, f"linear_{key}",
                lora.Linear(n_feat,
                            n_feat,
                            r=lora_rank,
                            lora_alpha=lora_alpha,
                            lora_dropout=lora_dropout,
                            bias=bias_dict[key]) if value else nn.Linear(
                                n_feat, n_feat, bias_dict[key]))
        self.dropout = nn.Dropout(p=kwargs.get('dropout_rate', 0.1))


class LoRARelPositionMultiHeadedAttention(LoRAMultiHeadedAttention,
                                          RelPositionMultiHeadedAttention):
    """Multi-Head Attention layer with relative position encoding.
    Paper: https://arxiv.org/abs/1901.02860
    Args:
        *args: Arguments for the MultiHeadedAttention.
        **kwargs: Keyword arguments for the MultiHeadedAttention.
        lora_rank (int): Rank for LoRA.
        lora_alpha (int): Alpha for LoRA.
        lora_dropout (float): Dropout rate for LoRA.
        lora_list (Optional[List[str]]): List of layers to apply LoRA.
    """

    def __init__(
        self,
        *args,
        lora_rank: int = 8,
        lora_alpha: int = 8,
        lora_dropout: float = 0.0,
        lora_list: Optional[List[str]] = None,
        **kwargs
    ):
        """Construct an RelPositionMultiHeadedAttention object."""
        super().__init__(*args, lora_rank=lora_rank, lora_alpha=lora_alpha,
                         lora_dropout=lora_dropout, lora_list=lora_list,
                         **kwargs)
        
        n_feat = args[1]  # Assuming the second argument is n_feat
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        # these two learnable bias are used in matrix c and matrix d
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.h, self.d_k))
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)
