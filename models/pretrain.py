import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import MultimodalBackbone
import timm
from timm.models.layers import DropPath, trunc_normal_
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

class PretrainModel(nn.Module):

    def __init__(self,
                 model_config,
                 model_device,
                 seq_length=5,
                 eeg_input_dim=310,
                 eye_input_dim=33,
                ):
        super().__init__()

        self.config = model_config
        self.model_device = model_device
        embed_dim = self.config.hidden_size

        self.backbone = MultimodalBackbone(model_config)

        # get query embeddings  
        self.eeg_embeds = nn.Linear(eeg_input_dim, embed_dim)
        self.eye_embeds = nn.Linear(eye_input_dim, embed_dim)


        # add cls token
        self.eeg_cls_token= nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.eye_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.seq_length = seq_length+1  # add cls token

        # add type embd
        self.eeg_type_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.eye_type_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # add pos embd
        self.eeg_pos_emb = nn.Parameter(torch.zeros(1, self.seq_length, embed_dim)) 
        self.eye_pos_emb = nn.Parameter(torch.zeros(1, self.seq_length, embed_dim)) 

        # eeg and eye projection for eeg-eye contrastive learning
        self.eeg_proj = nn.Linear(embed_dim, embed_dim)
        self.eye_proj = nn.Linear(embed_dim, embed_dim)
        # temperature parameter for contrastive learning
        self.temp = nn.Parameter(0.07 * torch.ones([]))

        # head for eeg-eye matching
        self.itm_head = nn.Linear(embed_dim, 2)
        self.itm_fusion = Fusion(embed_dim)

        # head for eeg and eye generation
        self.eeg_mse_head = nn.Linear(embed_dim, eeg_input_dim)
        self.eye_mse_head = nn.Linear(embed_dim, eye_input_dim)           

        trunc_normal_(self.eeg_pos_emb, std=0.02)
        trunc_normal_(self.eye_pos_emb, std=0.02)
        trunc_normal_(self.eeg_cls_token, std=0.02)
        trunc_normal_(self.eye_cls_token, std=0.02)
        trunc_normal_(self.eeg_type_embed, std=0.02)
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
        return { "eeg_pos_emb", "eye_pos_emb", "eeg_cls_token", "eye_cls_token", "eeg_type_embed", "eye_type_embed"}



    def forward(self, eeg=None,eye=None):

        batch_size = eeg.shape[0]


        eeg_embeds = self.eeg_embeds(eeg).to(self.model_device)
        eye_embeds = self.eye_embeds(eye).to(self.model_device)

        eeg_cls_tokens = self.eeg_cls_token.expand(eeg.size()[0], -1, -1)
        eye_cls_tokens = self.eye_cls_token.expand(eye.size()[0], -1, -1)

        eeg_embeds = torch.cat((eeg_cls_tokens, eeg_embeds), dim=1)
        eye_embeds = torch.cat((eye_cls_tokens, eye_embeds), dim=1)

        eeg_embeds = eeg_embeds + self.eeg_type_embed.expand(eeg_embeds.size()[0], eeg_embeds.size()[1], -1) + self.eeg_pos_emb.expand(eeg_embeds.size()[0], -1, -1)
        eye_embeds = eye_embeds + self.eye_type_embed.expand(eye_embeds.size()[0], eye_embeds.size()[1], -1) + self.eye_pos_emb.expand(eye_embeds.size()[0], -1, -1)


        # ============== EEG-EYE Contrastive =================== #
        con_output = self.backbone(
            query_embeds=eye_embeds,
            mode='con',
            text_embeds=eeg_embeds,
            attention_mask=torch.ones((batch_size, self.seq_length+self.seq_length), device=self.model_device),
            return_dict=True,
            eye_first=True,
        )

        
        last_hidden_state = con_output.hidden_states[-2]

        eye_output = last_hidden_state[:, :self.seq_length, :]
        eeg_output = last_hidden_state[:, self.seq_length:, :]

        eye_feats = F.normalize(self.eye_proj(eye_output[:,0,:]), dim=-1)  #batch,embd_dim
        eeg_feats = F.normalize(self.eeg_proj(eeg_output[:,0,:]), dim=-1)

        # gather feature
        if self.config.if_DDP:
            rank = (
                torch.distributed.get_rank()
                if torch.distributed.is_available()
                and torch.distributed.is_initialized()
                else 0
            )
            all_eye_feats = concat_all_gather(eye_feats) 
            all_eye_feats[rank]=eye_feats # batch_size = n_gpu*batch_size
            all_eeg_feats = concat_all_gather(eeg_feats)
            all_eeg_feats[rank]=eeg_feats

            all_eye_feats = torch.cat(all_eye_feats,dim=0)
            all_eeg_feats = torch.cat(all_eeg_feats,dim=0)
            sim_eeg2eye = torch.matmul(all_eeg_feats,all_eye_feats.T)/ self.temp # [n_gpu*batch_size, n_gpu*batch_size]
            sim_eye2eeg = torch.matmul(all_eye_feats,all_eeg_feats.T)/ self.temp # [n_gpu*batch_size, n_gpu*batch_size]
        else:
            sim_eeg2eye = torch.matmul(eeg_feats,eye_feats.T)/ self.temp # [batch_size, batch_size]
            sim_eye2eeg = torch.matmul(eye_feats,eeg_feats.T)/ self.temp # [batch_size, batch_size]

        bs = sim_eeg2eye.size(0)
        targets = torch.arange(bs, dtype=torch.int64).to(self.model_device)
        loss_con = (
                        F.cross_entropy(sim_eeg2eye, targets, label_smoothing=0.1)
                        + F.cross_entropy(sim_eye2eeg, targets, label_smoothing=0.1)
                ) / 2

        # ============== EEG-EYE Matching ===================#
        bs = eye_embeds.size(0)

        with torch.no_grad():
            if self.config.if_DDP:
                rank = (
                    torch.distributed.get_rank()
                    if torch.distributed.is_available()
                    and torch.distributed.is_initialized()
                    else 0
                )
                sim_eeg2eye = sim_eeg2eye[rank*bs:(rank+1)*bs,rank*bs:(rank+1)*bs]
                sim_eye2eeg = sim_eye2eeg[rank*bs:(rank+1)*bs,rank*bs:(rank+1)*bs]

            sim_eeg2eye[:, :bs].fill_diagonal_(-10000)  
            sim_eye2eeg[:, :bs].fill_diagonal_(-10000)

            weights_eeg2eye = F.softmax(sim_eeg2eye, dim=1)
            weights_eye2eeg = F.softmax(sim_eye2eeg, dim=1)

        # select a negative eye for each eeg; use hard negatives
        eye_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_eeg2eye[b], num_samples=1).item()  
            eye_embeds_neg.append(eye_embeds[neg_idx])
        eye_embeds_neg = torch.stack(eye_embeds_neg, dim=0)

        # select a negative eeg for each eye; use hard negatives
        eeg_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_eye2eeg[b], num_samples=1).item() 
            eeg_embeds_neg.append(eeg_embeds[neg_idx])
        eeg_embeds_neg = torch.stack(eeg_embeds_neg, dim=0)

        
        eye_embdes_all = torch.cat([eye_embeds, eye_embeds, eye_embeds_neg], dim=0)  # pos, pos, neg  , 3*batch,seq_len,embd
        eeg_embeds_all = torch.cat([eeg_embeds, eeg_embeds_neg, eeg_embeds], dim=0)  # pos, neg, pos

        eye_atts_itm = torch.ones(eye_embdes_all.size()[:-1], dtype=torch.long).to(eye.device)
        eeg_atts_itm = torch.ones(eeg_embeds_all.size()[:-1], dtype=torch.long).to(eeg.device)
        attention_mask_all = torch.cat([eye_atts_itm, eeg_atts_itm], dim=1)

        mat_output = self.backbone(
            query_embeds=eye_embdes_all,
            mode='mat',
            text_embeds=eeg_embeds_all,
            attention_mask=attention_mask_all,
            return_dict=True,
            eye_first=True,
        )

        eye_cls,eeg_cls = mat_output.last_hidden_state[:, 0, :],mat_output.last_hidden_state[:, eye_embdes_all.size(1), :]
        all_cls = self.itm_fusion(eeg_cls,eye_cls)
        vl_output = self.itm_head(all_cls)
        logits = vl_output

        itm_labels = torch.cat(
            [torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
            dim=0,
        ).to(eye.device)
        loss_mat = F.cross_entropy(logits, itm_labels)

        # ================= EEG and EYE Generation ======================== #

        gen_output = self.backbone(
            query_embeds=eye_embeds,
            mode='gen',
            text_embeds=eeg_embeds,
            attention_mask=torch.ones((batch_size, self.seq_length+self.seq_length), device=self.model_device),
            return_dict=True,
            eye_first=True
        )

        sequence_output = gen_output.last_hidden_state[:, eye_embeds.shape[1]:, :]
        prediction = self.eeg_mse_head(sequence_output)

        target = self.std_norm(eeg)
        rec = self.std_norm(prediction[:,:-1,:])
        eeg_rec_loss = F.mse_loss(rec, target)

        gen_output = self.backbone(
            query_embeds=eeg_embeds,
            mode='gen',
            text_embeds=eye_embeds,
            attention_mask=torch.ones((batch_size, self.seq_length+self.seq_length), device=self.model_device),
            return_dict=True,
            eye_first=False
        )

        sequence_output = gen_output.last_hidden_state[:, eeg_embeds.shape[1]:, :]
        prediction = self.eye_mse_head(sequence_output)


        target = self.std_norm(eye)
        rec = self.std_norm(prediction[:,:-1,:])
        eye_rec_loss = F.mse_loss(rec, target)

        lm_loss = 0.5*eeg_rec_loss+0.5*eye_rec_loss

        # ===================== Total Loss ======================== #

        loss = loss_con + loss_mat + lm_loss
        return loss, loss_con, loss_mat, lm_loss
        
    

    def std_norm(self, x):
        mean = torch.mean(x, dim=-1, keepdim=True)
        std = torch.std(x, dim=-1, keepdim=True)
        x = (x - mean) / (std+ 1.e-5)
        return x

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor)
    return tensors_gather  
