"""FatigueGuard supervised classifier for EEG, EOG, or paired inputs."""

import torch
import torch.nn as nn
from .backbone import FatigueGuardBackbone
from torch.nn.init import trunc_normal_


class Fusion(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dim, 1))
        self.softmax = nn.Softmax(1)

    def forward(self, eeg, eog):
        o1 = eeg @ self.weight
        o2 = eog @ self.weight
        o = torch.cat([o1, o2], dim=-1)
        alpha = self.softmax(o)
        eeg = eeg * alpha[:, 0].unsqueeze(1)
        eog = eog * alpha[:, 1].unsqueeze(1)
        out = eeg + eog
        return out


class FatigueGuard(nn.Module):

    def __init__(
        self,
        config,
        device,
        seq_length,
        eeg_input_dim,
        eog_input_dim,
        n_class=2,
        mode="multi",
    ):
        super().__init__()
        if mode not in ("multi", "eeg", "eog"):
            raise ValueError("mode must be multi, eeg, or eog")
        self.mode = mode
        self.config = config
        self.device = device
        embed_dim = self.config.hidden_size
        self.backbone = FatigueGuardBackbone(config)
        if mode == "multi":
            self.eeg_embeds = nn.Linear(eeg_input_dim, embed_dim)
            self.eog_embeds = nn.Linear(eog_input_dim, embed_dim)
        elif mode == "eeg":
            self.eeg_embeds = nn.Linear(eeg_input_dim, embed_dim)
        elif mode == "eog":
            self.eog_embeds = nn.Linear(eog_input_dim, embed_dim)
        self.seq_length = seq_length + 1
        if mode == "multi":
            self.eeg_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.eog_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        elif mode == "eeg":
            self.eeg_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        elif mode == "eog":
            self.eog_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if mode == "multi":
            self.eeg_type_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.eog_type_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        elif mode == "eeg":
            self.eeg_type_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        elif mode == "eog":
            self.eog_type_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if mode == "multi":
            self.eeg_pos_emb = nn.Parameter(torch.zeros(1, self.seq_length, embed_dim))
            self.eog_pos_emb = nn.Parameter(torch.zeros(1, self.seq_length, embed_dim))
        elif mode == "eeg":
            self.eeg_pos_emb = nn.Parameter(torch.zeros(1, self.seq_length, embed_dim))
        elif mode == "eog":
            self.eog_pos_emb = nn.Parameter(torch.zeros(1, self.seq_length, embed_dim))
        if mode == "multi":
            self.cls_fusion = Fusion(embed_dim)
        self.cls_head = nn.Linear(embed_dim, n_class)
        if mode == "multi":
            trunc_normal_(self.eeg_pos_emb, std=0.02)
            trunc_normal_(self.eog_pos_emb, std=0.02)
            trunc_normal_(self.eeg_cls_token, std=0.02)
            trunc_normal_(self.eog_cls_token, std=0.02)
            trunc_normal_(self.eeg_type_embed, std=0.02)
            trunc_normal_(self.eog_type_embed, std=0.02)
        elif mode == "eeg":
            trunc_normal_(self.eeg_pos_emb, std=0.02)
            trunc_normal_(self.eeg_cls_token, std=0.02)
            trunc_normal_(self.eeg_type_embed, std=0.02)
        elif mode == "eog":
            trunc_normal_(self.eog_pos_emb, std=0.02)
            trunc_normal_(self.eog_cls_token, std=0.02)
            trunc_normal_(self.eog_type_embed, std=0.02)
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
        return {
            "eeg_pos_emb",
            "eog_pos_emb",
            "eeg_cls_token",
            "eog_cls_token",
            "eeg_type_embed",
            "eog_type_embed",
        }

    def forward_features(self, eeg=None, eog=None):
        if eeg is None and eog is None:
            raise ValueError("At least one modality must be supplied")
        eeg_embeds = None
        eog_embeds = None
        has_eeg = True
        has_eog = True
        if eeg is None:
            has_eeg = False
        elif eog is None:
            has_eog = False
        if has_eeg:
            batch_size = eeg.shape[0]
            eeg_embeds = self.eeg_embeds(eeg).to(self.device)
        if has_eog:
            batch_size = eog.shape[0]
            eog_embeds = self.eog_embeds(eog).to(self.device)
        if has_eeg:
            eeg_cls_tokens = self.eeg_cls_token.expand(eeg.size()[0], -1, -1)
        if has_eog:
            eog_cls_tokens = self.eog_cls_token.expand(eog.size()[0], -1, -1)
        if has_eeg:
            eeg_embeds = torch.cat((eeg_cls_tokens, eeg_embeds), dim=1)
        if has_eog:
            eog_embeds = torch.cat((eog_cls_tokens, eog_embeds), dim=1)
        if has_eeg:
            eeg_embeds = (
                eeg_embeds
                + self.eeg_type_embed.expand(
                    eeg_embeds.size()[0], eeg_embeds.size()[1], -1
                )
                + self.eeg_pos_emb.expand(eeg_embeds.size()[0], -1, -1)
            )
        if has_eog:
            eog_embeds = (
                eog_embeds
                + self.eog_type_embed.expand(
                    eog_embeds.size()[0], eog_embeds.size()[1], -1
                )
                + self.eog_pos_emb.expand(eog_embeds.size()[0], -1, -1)
            )
        if has_eeg and has_eog:
            mask_length = self.seq_length + self.seq_length
        else:
            mask_length = self.seq_length
        ft_output = self.backbone(
            query_embeds=eog_embeds,
            mode="ft",
            text_embeds=eeg_embeds,
            attention_mask=torch.ones((batch_size, mask_length), device=self.device),
            return_dict=True,
            eog_first=True,
            has_eog=has_eog,
            has_eeg=has_eeg,
        )
        last_hidden_state = ft_output.last_hidden_state
        if has_eeg and has_eog:
            eog_cls = last_hidden_state[:, 0, :]
            eeg_cls = last_hidden_state[:, self.seq_length, :]
            cls_fusion = self.cls_fusion(eog_cls, eeg_cls)
        else:
            cls_fusion = last_hidden_state[:, 0, :]
        return cls_fusion

    def forward(self, eeg=None, eog=None):
        return self.cls_head(self.forward_features(eeg=eeg, eog=eog))
