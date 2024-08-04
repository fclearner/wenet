# Copyright (c) 2021 Mobvoi Inc (Binbin Zhang, Di Wu)
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
# Modified from ESPnet(https://github.com/espnet/espnet)
"""Encoder definition with lora."""
from wenet.transformer.encoder import TransformerEncoder, ConformerEncoder
from wenet.finetune.lora.utils import WENET_LORA_ATTENTION_CLASSES


class LoRATransformerEncoder(TransformerEncoder):
    """Transformer encoder module with lora."""

    def __init__(
        self,
        input_size: int,
        **kwargs
    ):
        """Construct TransformerEncoder with LoRA parameters

        Args:
            input_size (int): input dim.
            **kwargs: Keyword arguments for the LoRATransformerEncoder.
        """
        # filter parameters
        transformer_encoder_kwargs = {
            k: v for k, v in kwargs.items() \
            if k in TransformerEncoder.__init__.__code__.co_varnames
        }
        super().__init__(input_size=input_size, **transformer_encoder_kwargs)

        selfattention_layer_type = kwargs.get('selfattention_layer_type',
                                              'selfattn')
        assert selfattention_layer_type in ['selfattn']

        # self-attention module definition
        encoder_selfattn_layer_args = {
            'attention_heads': kwargs.get('attention_heads', 4),
            'output_size': kwargs.get('output_size', 256),
            'attention_dropout_rate': kwargs.get('attention_dropout_rate', 0.0),
            'query_bias': kwargs.get('query_bias', True),
            'key_bias': kwargs.get('key_bias', True),
            'value_bias': kwargs.get('value_bias', True),
            'use_sdpa': kwargs.get('use_sdpa', False),
            'n_kv_head': kwargs.get('n_kv_head', None),
            'head_dim': kwargs.get('head_dim', None),
            'lora_rank': kwargs.get('lora_rank', 8),
            'lora_alpha': kwargs.get('lora_alpha', 8),
            'lora_dropout': kwargs.get('lora_dropout',0.0),
            'lora_list': kwargs.get('lora_list', None),
        }
        for i in range(kwargs.get('num_blocks', 6)):
            self.encoders[i].self_attn = \
                WENET_LORA_ATTENTION_CLASSES[selfattention_layer_type](
                    *encoder_selfattn_layer_args
                )


class LoRAConformerEncoder(ConformerEncoder):
    """Conformer encoder module with lora."""

    def __init__(
        self,
        input_size: int,
        **kwargs
    ):
        """Construct ConformerEncoder with LoRA parameters

        Args:
            input_size (int): input dim.
            **kwargs: Keyword arguments for the LoRAConformerEncoder.
        """
        # filter parameters
        conformer_encoder_kwargs = {
            k: v for k, v in kwargs.items() \
            if k in ConformerEncoder.__init__.__code__.co_varnames
        }
        super().__init__(input_size=input_size, **conformer_encoder_kwargs)

        # self-attention module definition
        # self-attention module definition
        encoder_selfattn_layer_args = {
            'attention_heads': kwargs.get('attention_heads', 4),
            'output_size': kwargs.get('output_size', 256),
            'attention_dropout_rate': kwargs.get('attention_dropout_rate', 0.0),
            'query_bias': kwargs.get('query_bias', True),
            'key_bias': kwargs.get('key_bias', True),
            'value_bias': kwargs.get('value_bias', True),
            'use_sdpa': kwargs.get('use_sdpa', False),
            'n_kv_head': kwargs.get('n_kv_head', None),
            'head_dim': kwargs.get('head_dim', None),
            'lora_rank': kwargs.get('lora_rank', 8),
            'lora_alpha': kwargs.get('lora_alpha', 8),
            'lora_dropout': kwargs.get('lora_dropout',0.0),
            'lora_list': kwargs.get('lora_list', None),
        }
        for i in range(kwargs.get('num_blocks', 6)):
            self.encoders[i].self_attn = WENET_LORA_ATTENTION_CLASSES[
                kwargs.get('selfattention_layer_type', 'rel_selfattn')
            ](*encoder_selfattn_layer_args)
