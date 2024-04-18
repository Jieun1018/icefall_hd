# Copyright 2021 Xuankai Chang
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Encoder definition."""
import contextlib
import time
import copy
import math 
import logging
import os
from typing import List, Optional, Tuple
import warnings

import torch
from filelock import FileLock
#from typeguard import check_argument_types

from nets_utils import make_pad_mask
from encoder_interface import EncoderInterface
from torch import Tensor, nn

from scaling import ScaledLinear

from icefall.utils import make_pad_mask, subsequent_chunk_mask
try:
    import fairseq
    from xlsr_audio import *
except Exception as e:
    print("Error: FairSeq is not properly installed.")
    print(
        "Please install FairSeq: cd ${MAIN_ROOT}/tools && make fairseq.done"
    )
    raise e


class XLSREncoder(EncoderInterface):
    """FairSeq Wav2Vec2 encoder module.

    Args:
        input_size: input dim
        output_size: dimension of attention
        w2v_url: url to Wav2Vec2.0 pretrained model
        w2v_dir_path: directory to download the Wav2Vec2.0 pretrained model.
        normalize_before: whether to use layer_norm before the first block
        finetune_last_n_layers: last n layers to be finetuned in Wav2Vec2.0
                                0 means to finetune every layer if freeze_w2v=False.
    """

    def __init__(
        self,
        input_size: int,
        w2v_url: str,
        w2v_dir_path: str = "./",
        output_size: int = 256,
        freeze_finetune_updates: int = 0,
        additional_block: bool = False,
    ):
        #assert check_argument_types()
        super().__init__()
                    
        '''
        if os.path.exists('/home/work/workspace/models/data2vec_model/audio_base_ls.pt'):
            self.w2v_model_path = '/home/work/workspace/models/data2vec_model/audio_base_ls.pt'
        if os.path.exists('/workspace/models/audio_base_ls.pt'):
            self.w2v_model_path = '/workspace/models/audio_base_ls.pt'
        '''
        self.w2v_model_path = download_xlsr()
        self._output_size = output_size

        models, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [self.w2v_model_path],
            strict=False,
        )
        model = models[0]
        model.feature_grad_mult = 0.0 ## for conv network freeze
        model.mask_prob = 0.5 ## for conv network freeze
        
        self.encoders = model
        self.pretrained_params = copy.deepcopy(model.state_dict())

        if model.cfg.encoder_embed_dim != output_size or additional_block:
            # TODO(xkc09): try LSTM
            self.output_layer = torch.nn.Sequential(
                ScaledLinear(model.cfg.encoder_embed_dim, output_size),
                #torch.nn.Linear(model.cfg.encoder_embed_dim, output_size),
                torch.nn.LayerNorm(output_size),
                torch.nn.GELU(),
            )
        else:
            self.output_layer = None

        self.freeze_finetune_updates = freeze_finetune_updates
        self.num_updates = 0

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        warmup = None,
        prev_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        xs_pad = x
        ilens = x_lens
        """Forward FairSeqWav2Vec2 Encoder.

        Args:
            xs_pad: input tensor (B, L, D)
            ilens: input length (B)
            prev_states: Not to be used now.
        Returns:
            position embedded tensor and mask
        """
        with torch.no_grad():
            xs_pad = torch.nn.functional.layer_norm(xs_pad, xs_pad.shape)

        masks = make_pad_mask(ilens).to(xs_pad.device)

        ft = (self.freeze_finetune_updates <= self.num_updates) and self.encoders.training
        if self.num_updates <= self.freeze_finetune_updates:
            self.num_updates += 1
        elif ft and self.num_updates == self.freeze_finetune_updates + 1:
            self.num_updates += 1
            logging.info("Start fine-tuning wav2vec parameters!")
        
        with torch.no_grad() if not ft else contextlib.nullcontext():
            enc_outputs = self.encoders(
                xs_pad,
                masks,
                mask = ft,
                features_only=True,
            )
            cnn_outputs = self.encoders.feature_extractor(xs_pad)

        xs_pad = enc_outputs["x"]  # (B,T,C),
        bs = xs_pad.shape[0]
        if enc_outputs["padding_mask"] is not None:
            masks = enc_outputs["padding_mask"]  # (B, T)
            olens = (~masks).sum(dim=1)  # (B)
        else:
            olens = torch.IntTensor([xs_pad.shape[1]]).repeat(bs).to(xs_pad.device)

        if self.output_layer is not None:
            xs_pad = self.output_layer(xs_pad)

        return xs_pad, olens, cnn_outputs

    def reload_pretrained_parameters(self):
        self.encoders.load_state_dict(self.pretrained_params)
        logging.info("Pretrained data2vec model parameters reloaded!")


class MultiXLSREncoder(EncoderInterface):
    """FairSeq Wav2Vec2 encoder module.

    Args:
        input_size: input dim
        output_size: dimension of attention
        w2v_url: url to Wav2Vec2.0 pretrained model
        w2v_dir_path: directory to download the Wav2Vec2.0 pretrained model.
        normalize_before: whether to use layer_norm before the first block
        finetune_last_n_layers: last n layers to be finetuned in Wav2Vec2.0
                                0 means to finetune every layer if freeze_w2v=False.
    """

    def __init__(
        self,
        input_size: int,
        w2v_url: str,
        w2v_dir_path: str = "./",
        output_size: int = 256,
        freeze_finetune_updates: int = 0,
        additional_block: bool = False,
        language_num: int = 2,
    ):
        #assert check_argument_types()
        super().__init__()
                    
        '''
        if os.path.exists('/home/work/workspace/models/data2vec_model/audio_base_ls.pt'):
            self.w2v_model_path = '/home/work/workspace/models/data2vec_model/audio_base_ls.pt'
        if os.path.exists('/workspace/models/audio_base_ls.pt'):
            self.w2v_model_path = '/workspace/models/audio_base_ls.pt'
        '''
        self.w2v_model_path = download_xlsr()
        self._output_size = output_size

        self.encoders = []

        for i in range(language_num):
            models, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task(
                [self.w2v_model_path],
                strict=False,
            )
            model = models[0]
            model.feature_grad_mult = 0.0 ## for conv network freeze
            model.mask_prob = 0.5 ## for conv network freeze
            
            self.pretrained_params = copy.deepcopy(model.state_dict())

            self.encoders.append(model)
            
        if model.cfg.encoder_embed_dim != output_size or additional_block:
            # TODO(xkc09): try LSTM
            self.output_layer = [torch.nn.Sequential(
                ScaledLinear(model.cfg.encoder_embed_dim, output_size),
                #torch.nn.Linear(model.cfg.encoder_embed_dim, output_size),
                torch.nn.LayerNorm(output_size),
                torch.nn.GELU(),
            )
                for i in range(language_num)]
        else:
            self.output_layer = None
        self.freeze_finetune_updates = freeze_finetune_updates
        self.num_updates = 0

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        lstm = None,
        linear = None,
        softmax = None,
        ctc_output = None,
        warmup = None,
        prev_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        xs_pad = x
        ilens = x_lens
        """Forward FairSeqWav2Vec2 Encoder.

        Args:
            xs_pad: input tensor (B, L, D)
            ilens: input length (B)
            prev_states: Not to be used now.
        Returns:
            position embedded tensor and mask
        """
        with torch.no_grad():
            xs_pad = torch.nn.functional.layer_norm(xs_pad, xs_pad.shape)

        masks = make_pad_mask(ilens).to(xs_pad.device)
        
        ft = (self.freeze_finetune_updates <= self.num_updates) and self.encoders[0].training
        if self.num_updates <= self.freeze_finetune_updates:
            self.num_updates += 1
        elif ft and self.num_updates == self.freeze_finetune_updates + 1:
            self.num_updates += 1
            logging.info("Start fine-tuning wav2vec parameters!")
        
        with torch.no_grad() if not ft else contextlib.nullcontext():
            cnn_outputs = self.encoders[0].feature_extractor(xs_pad)
            cnn_outputs = cnn_outputs.transpose(1, 2) 
            output = lstm(cnn_outputs)

            final = []
            final = torch.tensor(final).to('cuda')
            
            final = output[0][:, -1, :]
            #final = output[0][:, 100, :]
            '''
            for i in range(len(x_lens)):
                new_output = output[0][i, x_lens[i]-1, :]
                new_output = new_output.reshape(1, -1)
                final = torch.cat((final, new_output), dim=0)
            '''
            lid_final = linear(final)
            lid_final = softmax(lid_final)

            max_lid = torch.argmax(lid_final, dim=1)
            
            enc_outputs = None

            for i, lid in enumerate(max_lid):
                lid = lid.item()
                new_xs = xs_pad[i].unsqueeze(0)
                new_masks = masks[i].unsqueeze(0)
                enc_outputs_part = self.encoders[lid](
                    new_xs,
                    new_masks,
                    mask = ft,
                    features_only=True,
                )

                if i == 0:
                    enc_outputs = enc_outputs_part
                ## TODO: batch decoding....
        
        xs_pad = enc_outputs["x"]  # (B,T,C),
        bs = xs_pad.shape[0]
        if enc_outputs["padding_mask"] is not None:
            masks = enc_outputs["padding_mask"]  # (B, T)
            olens = (~masks).sum(dim=1)  # (B)
        else:
            olens = torch.IntTensor([xs_pad.shape[1]]).repeat(bs).to(xs_pad.device)
        
        xs_pad_new = None
        if self.output_layer is not None:
            for i, lid in enumerate(max_lid):
                new_xs = xs_pad[i].unsqueeze(0)
                new_xs = self.output_layer[lid](new_xs)
                
                if i == 0:
                    xs_pad_new = new_xs
                ## TODO: batch decoding...
        
        xs_pad = xs_pad_new
        
        '''
        for i, lid in enumerate(max_lid):
            ctc_res = ctc_output[lid](xs_pad[i].unsqueeze(0))
            ctc_res = softmax(ctc_res)
            ctc_prob, ctc_idx = ctc_res.max(-1)
            ctc_prob = ctc_prob[ctc_idx!=0]
        '''
        #if lstm == None:
        #    return xs_pad, olens, cnn_outputs
        #else:
        #    return xs_pad, olens, max_lid
        return xs_pad, olens, cnn_outputs if lstm == None else max_lid

    def reload_pretrained_parameters(self):
        self.encoders.load_state_dict(self.pretrained_params)
        logging.info("Pretrained data2vec model parameters reloaded!")


def download_xlsr(model_url='https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr2_300m.pt', dir_path='./models'):
    os.makedirs(dir_path, exist_ok=True)

    #model_name = model_url.split("/")[-1]
    model_name = 'xlsr2_300m.pt'
    model_path = os.path.join(dir_path, model_name)

    dict_url = "https://dl.fbaipublicfiles.com/fairseq/wav2vec/dict.ltr.txt"
    dict_path = os.path.join(dir_path, dict_url.split("/")[-1])

    with FileLock(model_path + ".lock"):
        if not os.path.exists(model_path):
            torch.hub.download_url_to_file(model_url, model_path)
            torch.hub.download_url_to_file(dict_url, dict_path)
            logging.info(f"data2vec model downloaded {model_path}")
        else:
            logging.info(f"data2vec model {model_path} already exists.")

    return model_path


if __name__ == '__main__':
    d2v = FairSeqData2VecEncoder(input_size=768, w2v_url='ww', output_size=768)
    inputs = torch.randn([1, 211564])
    #a = torch.ones([1000]
    #b = torch.ones([10000])
    #c = torch.ones([10000])
    length = torch.tensor([211564])
    outputs = d2v(inputs, length)
    print(outputs[0].size())

    #for n, p in d2v.named_parameters():
    #    print(n)