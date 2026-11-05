import torch
import torch.nn as nn

from .backbone import MultimodalBackbone
import timm
from timm.models.layers import  trunc_normal_
assert timm.__version__ == "0.3.2"
import numpy as np




class Fusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dim, 1))
        self.softmax = nn.Softmax(1)

    def forward(self, eeg, eye):
        o1 = eeg @ self.weight
        o2 = eye @ self.weight
        o = torch.cat([o1, o2], dim=-1)
        alpha = self.softmax(o)
        eeg = eeg * alpha[:, 0].unsqueeze(1)
        eye = eye * alpha[:, 1].unsqueeze(1)
        out = eeg + eye
        return out

class FinetuneModel(nn.Module):

    def __init__(self,
                 model_config,
                 model_device,
                 seq_length=5,
                 eeg_input_dim=310,
                 eye_input_dim=50,
                 n_class=7,
                 mode='multi'

                ):
        super().__init__()
        self.mode = mode
        self.config = model_config
        self.model_device = model_device
        embed_dim = self.config.hidden_size

        self.backbone = MultimodalBackbone(model_config)

        print(f'mode : {mode}')

        if mode == 'multi':
            self.eeg_embeds = nn.Linear(eeg_input_dim, embed_dim)
            self.eye_embeds = nn.Linear(eye_input_dim, embed_dim)
        elif mode =='eeg':
            self.eeg_embeds = nn.Linear(eeg_input_dim, embed_dim)
        elif mode == 'eye':
            self.eye_embeds = nn.Linear(eye_input_dim, embed_dim)


        self.seq_length = seq_length+1  # add cls token

        # add cls token
        if mode == 'multi':
            self.eeg_cls_token= nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.eye_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        elif mode =='eeg':
            self.eeg_cls_token= nn.Parameter(torch.zeros(1, 1, embed_dim))
        elif mode =='eye':
            self.eye_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        

        # add type embd
        if mode == 'multi':
            self.eeg_type_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.eye_type_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        elif mode =='eeg':
            self.eeg_type_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        elif mode =='eye':
            self.eye_type_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # add pos embd
        if mode == 'multi':
            self.eeg_pos_emb = nn.Parameter(torch.zeros(1, self.seq_length, embed_dim))  
            self.eye_pos_emb = nn.Parameter(torch.zeros(1, self.seq_length, embed_dim))  
        elif mode =='eeg':
            self.eeg_pos_emb = nn.Parameter(torch.zeros(1, self.seq_length, embed_dim))  
        elif mode =='eye':
            self.eye_pos_emb = nn.Parameter(torch.zeros(1, self.seq_length, embed_dim))  

        # add cls head
        if mode == 'multi':
            self.cls_fusion = Fusion(embed_dim)

        self.cls_head = nn.Linear(embed_dim, n_class)
        
        if mode == 'multi':
            trunc_normal_(self.eeg_pos_emb, std=0.02)
            trunc_normal_(self.eye_pos_emb, std=0.02)
            trunc_normal_(self.eeg_cls_token, std=0.02)
            trunc_normal_(self.eye_cls_token, std=0.02)
            trunc_normal_(self.eeg_type_embed, std=0.02)
            trunc_normal_(self.eye_type_embed, std=0.02)
        elif mode =='eeg':
            trunc_normal_(self.eeg_pos_emb, std=0.02)
            trunc_normal_(self.eeg_cls_token, std=0.02)
            trunc_normal_(self.eeg_type_embed, std=0.02)
        elif mode =='eye':
            trunc_normal_(self.eye_pos_emb, std=0.02)
            trunc_normal_(self.eye_cls_token, std=0.02)
            trunc_normal_(self.eye_type_embed, std=0.02)
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"eeg_pos_emb", "eye_pos_emb", "eeg_cls_token", "eye_cls_token", "eeg_type_embed", "eye_type_embed"}

    def forward_features(self, eeg=None, eye=None):

        
        eeg_embeds = None
        eye_embeds = None
        has_eeg=True
        has_eye=True

        if eeg is None:
            has_eeg=False
        elif eye is None:
            has_eye=False

        if has_eeg:
            batch_size = eeg.shape[0]
            eeg_embeds = self.eeg_embeds(eeg).to(self.model_device)
        if has_eye:
            batch_size = eye.shape[0]
            eye_embeds = self.eye_embeds(eye).to(self.model_device)

        if has_eeg:
            eeg_cls_tokens = self.eeg_cls_token.expand(eeg.size()[0], -1, -1)
        if has_eye:            
            eye_cls_tokens = self.eye_cls_token.expand(eye.size()[0], -1, -1)

        if has_eeg:
            eeg_embeds = torch.cat((eeg_cls_tokens, eeg_embeds), dim=1)
        if has_eye:        
            eye_embeds = torch.cat((eye_cls_tokens, eye_embeds), dim=1)
        
        if has_eeg:
            eeg_embeds = eeg_embeds + self.eeg_type_embed.expand(eeg_embeds.size()[0], eeg_embeds.size()[1], -1) + self.eeg_pos_emb.expand(eeg_embeds.size()[0], -1, -1)
        if has_eye:        
            eye_embeds = eye_embeds + self.eye_type_embed.expand(eye_embeds.size()[0], eye_embeds.size()[1], -1) + self.eye_pos_emb.expand(eye_embeds.size()[0], -1, -1)
        
        if has_eeg and has_eye:
            mask_length = self.seq_length+self.seq_length
        else:
            mask_length = self.seq_length

        ft_output = self.backbone(
            query_embeds=eye_embeds,
            mode='ft',
            text_embeds=eeg_embeds,
            attention_mask=torch.ones((batch_size, mask_length), device=self.model_device),
            return_dict=True,
            eye_first=True,
            has_eye=has_eye,
            has_eeg=has_eeg
        
        )

        last_hidden_state = ft_output.last_hidden_state
        if has_eeg and has_eye:
            eye_cls = last_hidden_state[:, 0, :]
            eeg_cls = last_hidden_state[:, self.seq_length, :]
            cls_fusion = self.cls_fusion(eye_cls,eeg_cls)
        else:
            cls_fusion = last_hidden_state[:, 0, :]
        return cls_fusion

    def forward(self, eeg=None, eye=None):
        return self.cls_head(self.forward_features(eeg=eeg, eye=eye))
    
