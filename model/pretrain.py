"""FatigueGuard contrastive, matching, and autoregressive generation objectives."""

import torch
import torch.nn as nn
import torch.nn.functional as F
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


class FatigueGuardPretrain(nn.Module):

    def __init__(
        self,
        config,
        device,
        seq_length,
        eeg_input_dim,
        eog_input_dim,
        loss_weights=None,
    ):
        super().__init__()
        self.config = config
        self.device = device
        self.loss_weights = {"con": 1.0, "mat": 1.0, "gen": 1.0}
        if loss_weights is not None:
            self.loss_weights.update(loss_weights)
        embed_dim = self.config.hidden_size
        self.backbone = FatigueGuardBackbone(config)
        self.eeg_embeds = nn.Linear(eeg_input_dim, embed_dim)
        self.eog_embeds = nn.Linear(eog_input_dim, embed_dim)
        self.eeg_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.eog_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.seq_length = seq_length + 1
        self.eeg_type_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.eog_type_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.eeg_pos_emb = nn.Parameter(torch.zeros(1, self.seq_length, embed_dim))
        self.eog_pos_emb = nn.Parameter(torch.zeros(1, self.seq_length, embed_dim))
        self.eeg_proj = nn.Linear(embed_dim, embed_dim)
        self.eog_proj = nn.Linear(embed_dim, embed_dim)
        self.temp = nn.Parameter(0.07 * torch.ones([]))
        self.itm_head = nn.Linear(embed_dim, 2)
        self.itm_fusion = Fusion(embed_dim)
        self.eeg_mse_head = nn.Linear(embed_dim, eeg_input_dim)
        self.eog_mse_head = nn.Linear(embed_dim, eog_input_dim)
        trunc_normal_(self.eeg_pos_emb, std=0.02)
        trunc_normal_(self.eog_pos_emb, std=0.02)
        trunc_normal_(self.eeg_cls_token, std=0.02)
        trunc_normal_(self.eog_cls_token, std=0.02)
        trunc_normal_(self.eeg_type_embed, std=0.02)
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

    def forward(self, eeg=None, eog=None):
        if eeg is None or eog is None or len(eeg) < 2:
            raise ValueError(
                "Pretraining requires at least two paired EEG/EOG sequences"
            )
        batch_size = eeg.shape[0]
        eeg_embeds = self.eeg_embeds(eeg).to(self.device)
        eog_embeds = self.eog_embeds(eog).to(self.device)
        eeg_cls_tokens = self.eeg_cls_token.expand(eeg.size()[0], -1, -1)
        eog_cls_tokens = self.eog_cls_token.expand(eog.size()[0], -1, -1)
        eeg_embeds = torch.cat((eeg_cls_tokens, eeg_embeds), dim=1)
        eog_embeds = torch.cat((eog_cls_tokens, eog_embeds), dim=1)
        eeg_embeds = (
            eeg_embeds
            + self.eeg_type_embed.expand(eeg_embeds.size()[0], eeg_embeds.size()[1], -1)
            + self.eeg_pos_emb.expand(eeg_embeds.size()[0], -1, -1)
        )
        eog_embeds = (
            eog_embeds
            + self.eog_type_embed.expand(eog_embeds.size()[0], eog_embeds.size()[1], -1)
            + self.eog_pos_emb.expand(eog_embeds.size()[0], -1, -1)
        )
        con_output = self.backbone(
            query_embeds=eog_embeds,
            mode="con",
            text_embeds=eeg_embeds,
            attention_mask=torch.ones(
                (batch_size, self.seq_length + self.seq_length), device=self.device
            ),
            return_dict=True,
            eog_first=True,
        )
        if getattr(self.config, "disable_fusion_transformer", False):
            last_hidden_state = con_output.last_hidden_state
        else:
            last_hidden_state = con_output.hidden_states[-2]
        eog_output = last_hidden_state[:, : self.seq_length, :]
        eeg_output = last_hidden_state[:, self.seq_length :, :]
        eog_feats = F.normalize(self.eog_proj(eog_output[:, 0, :]), dim=-1)
        eeg_feats = F.normalize(self.eeg_proj(eeg_output[:, 0, :]), dim=-1)
        if self.config.if_DDP:
            rank = torch.distributed.get_rank()
            all_eog_feats = concat_all_gather(eog_feats)
            all_eog_feats[rank] = eog_feats
            all_eeg_feats = concat_all_gather(eeg_feats)
            all_eeg_feats[rank] = eeg_feats
            all_eog_feats = torch.cat(all_eog_feats, dim=0)
            all_eeg_feats = torch.cat(all_eeg_feats, dim=0)
            sim_eeg2eog = torch.matmul(all_eeg_feats, all_eog_feats.T) / self.temp
            sim_eog2eeg = torch.matmul(all_eog_feats, all_eeg_feats.T) / self.temp
        else:
            sim_eeg2eog = torch.matmul(eeg_feats, eog_feats.T) / self.temp
            sim_eog2eeg = torch.matmul(eog_feats, eeg_feats.T) / self.temp
        bs = sim_eeg2eog.size(0)
        targets = torch.arange(bs, dtype=torch.int64).to(self.device)
        loss_con = (
            F.cross_entropy(sim_eeg2eog, targets, label_smoothing=0.1)
            + F.cross_entropy(sim_eog2eeg, targets, label_smoothing=0.1)
        ) / 2
        bs = eog_embeds.size(0)
        with torch.no_grad():
            if self.config.if_DDP:
                rank = torch.distributed.get_rank()
                sim_eeg2eog = sim_eeg2eog[
                    rank * bs : (rank + 1) * bs, rank * bs : (rank + 1) * bs
                ]
                sim_eog2eeg = sim_eog2eeg[
                    rank * bs : (rank + 1) * bs, rank * bs : (rank + 1) * bs
                ]
            sim_eeg2eog[:, :bs].fill_diagonal_(-10000)
            sim_eog2eeg[:, :bs].fill_diagonal_(-10000)
            weights_eeg2eog = F.softmax(sim_eeg2eog, dim=1)
            weights_eog2eeg = F.softmax(sim_eog2eeg, dim=1)
        eog_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_eeg2eog[b], num_samples=1).item()
            eog_embeds_neg.append(eog_embeds[neg_idx])
        eog_embeds_neg = torch.stack(eog_embeds_neg, dim=0)
        eeg_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_eog2eeg[b], num_samples=1).item()
            eeg_embeds_neg.append(eeg_embeds[neg_idx])
        eeg_embeds_neg = torch.stack(eeg_embeds_neg, dim=0)
        eog_embdes_all = torch.cat([eog_embeds, eog_embeds, eog_embeds_neg], dim=0)
        eeg_embeds_all = torch.cat([eeg_embeds, eeg_embeds_neg, eeg_embeds], dim=0)
        eog_atts_itm = torch.ones(eog_embdes_all.size()[:-1], dtype=torch.long).to(
            eog.device
        )
        eeg_atts_itm = torch.ones(eeg_embeds_all.size()[:-1], dtype=torch.long).to(
            eeg.device
        )
        attention_mask_all = torch.cat([eog_atts_itm, eeg_atts_itm], dim=1)
        mat_output = self.backbone(
            query_embeds=eog_embdes_all,
            mode="mat",
            text_embeds=eeg_embeds_all,
            attention_mask=attention_mask_all,
            return_dict=True,
            eog_first=True,
        )
        eog_cls, eeg_cls = (
            mat_output.last_hidden_state[:, 0, :],
            mat_output.last_hidden_state[:, eog_embdes_all.size(1), :],
        )
        all_cls = self.itm_fusion(eeg_cls, eog_cls)
        vl_output = self.itm_head(all_cls)
        logits = vl_output
        itm_labels = torch.cat(
            [torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
            dim=0,
        ).to(eog.device)
        loss_mat = F.cross_entropy(logits, itm_labels)
        gen_output = self.backbone(
            query_embeds=eog_embeds,
            mode="gen",
            text_embeds=eeg_embeds,
            attention_mask=torch.ones(
                (batch_size, self.seq_length + self.seq_length), device=self.device
            ),
            return_dict=True,
            eog_first=True,
        )
        sequence_output = gen_output.last_hidden_state[:, eog_embeds.shape[1] :, :]
        prediction = self.eeg_mse_head(sequence_output)
        target = self.std_norm(eeg)
        rec = self.std_norm(prediction[:, :-1, :])
        eeg_rec_loss = F.mse_loss(rec, target)
        gen_output = self.backbone(
            query_embeds=eeg_embeds,
            mode="gen",
            text_embeds=eog_embeds,
            attention_mask=torch.ones(
                (batch_size, self.seq_length + self.seq_length), device=self.device
            ),
            return_dict=True,
            eog_first=False,
        )
        sequence_output = gen_output.last_hidden_state[:, eeg_embeds.shape[1] :, :]
        prediction = self.eog_mse_head(sequence_output)
        target = self.std_norm(eog)
        rec = self.std_norm(prediction[:, :-1, :])
        eog_rec_loss = F.mse_loss(rec, target)
        lm_loss = 0.5 * eeg_rec_loss + 0.5 * eog_rec_loss
        loss = (
            self.loss_weights["con"] * loss_con
            + self.loss_weights["mat"] * loss_mat
            + self.loss_weights["gen"] * lm_loss
        )
        return (loss, loss_con, loss_mat, lm_loss)

    def std_norm(self, x):
        mean = torch.mean(x, dim=-1, keepdim=True)
        std = torch.std(x, dim=-1, keepdim=True)
        x = (x - mean) / (std + 1e-05)
        return x


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor)
    return tensors_gather
