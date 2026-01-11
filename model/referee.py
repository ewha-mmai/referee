import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path
from model.sync_model import Synchformer, init_weights

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRETRAINED_PATH = PROJECT_ROOT / "model" / "pretrained" / "23-12-23T18-33-57.pt"

# helper function: load weights
def _load_backbone_weights(self, ckpt_path: str):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt = torch.load(ckpt_path, map_location=device)
    raw_model = ckpt.get("model", ckpt.get("state_dict", ckpt))

    def normalize_key(k: str):
        if k.startswith("module."): k = k[len("module."):]
        if k.startswith("backbone."): k = k[len("backbone."):]
        return k

    backbone_state = {normalize_key(k): v for k, v in raw_model.items()}
    backbone_state = {k: v for k, v in backbone_state.items() if not k.startswith("transformer.sync_head")}
    
    missing, unexpected = self.backbone.load_state_dict(backbone_state, strict=False)
    if missing: print(f"[CKPT INFO] Missing backbone keys: {missing}")
    if unexpected: print(f"[CKPT INFO] Unexpected backbone keys: {unexpected}")
    print(f"[CKPT INFO] Loaded backbone weights from {ckpt_path}")


def _load_full_model(self, ckpt_path: str):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt = torch.load(ckpt_path, map_location=device)

    # Handle both wrapped and direct state_dict formats
    if 'model_state_dict' in ckpt:
        raw_model = ckpt['model_state_dict']
    else:
        raw_model = ckpt

    new_state_dict = {}
    for k, v in raw_model.items():
        new_k = k
        if k.startswith("module."):
            new_k = k[len("module."):]
        new_state_dict[new_k] = v

    missing, unexpected = self.load_state_dict(new_state_dict, strict=False)
    if missing:
        print(f"[ckpt load] missing keys: {missing}")
    if unexpected:
        print(f"[ckpt load] unexpected keys: {unexpected}")
    print(f"[ckpt load] loaded full model from {ckpt_path}")


class SegmentEmbedding(nn.Embedding):
    """Segment embedding for [CLS]=0 and ID queries=1."""

    def __init__(self, embed_size=768):
        super().__init__(2, embed_size)


class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.1):
        super().__init__()
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.ln_ff = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), 
            nn.GELU(), 
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model), 
            nn.Dropout(dropout)
        )

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        q, kv = self.ln_q(query), self.ln_kv(key_value)
        attn_output, _ = self.attn(query=q, key=kv, value=kv)
        x = query + attn_output
        x = x + self.ffn(self.ln_ff(x))
        return x


class IDBBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.1):
        super().__init__()
        self.ln_sa = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.cross_attn_ffn = CrossAttentionBlock(d_model, n_head, dropout)
    
    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        # self-attention among learnable ID queries
        q_sa = query
        q_norm = self.ln_sa(q_sa)
        sa_output, _ = self.self_attn(query=q_norm, key=q_norm, value=q_norm)
        x = q_sa + sa_output
        # cross-attention with visual+audio sequence
        output = self.cross_attn_ffn(query=x, key_value=key_value)

        return output # (B,Q,D)


class IdentityBottleneck(nn.Module):
    """
    Identity Bottleneck module:
    - Compress AV (visual+audio) sequence into a fixed number of identity queries.
    - ID queries undergo self-attention and cross-attention with the AV sequence.
    """
    def __init__(self, d_model: int, n_head: int, num_queries: int = 6, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.id_queries = nn.Parameter(torch.randn(1, num_queries, d_model) * 0.02) # (1,Q,D)
        self.query_pos_emb = nn.Parameter(torch.randn(1, num_queries, d_model) * 0.02)
        self.blocks = nn.ModuleList([
            IDBBlock(d_model, n_head, dropout) for _ in range(num_layers)
        ])

    def forward(self, vis_tokens: torch.Tensor) -> torch.Tensor:
        B = vis_tokens.size(0) # (B,Q,D)
        q = (self.id_queries + self.query_pos_emb).expand(B, -1, -1)
        kv = vis_tokens # (B,Tv,D)

        for block in self.blocks:
            q = block(query=q, key_value=kv)
        return q # (B,Q,D)


class Referee(nn.Module):
    def __init__(self, cfg, ckpt_path=None):
        super().__init__()
        # Backbone initialization
        self.backbone = Synchformer(
            afeat_extractor=cfg.model.params.afeat_extractor,
            vfeat_extractor=cfg.model.params.vfeat_extractor,
            aproj=cfg.model.params.aproj,
            vproj=cfg.model.params.vproj,
            transformer=cfg.model.params.transformer,
        )

        d_model = self.backbone.transformer.config.n_embd
        n_head = self.backbone.transformer.config.n_head
        dropout = self.backbone.transformer.config.embd_pdrop

        num_id_queries = cfg.model.params.identity_bottleneck.params.n_id
        num_id_layers = cfg.model.params.identity_bottleneck.params.n_layer
        num_cross_attn_layers = cfg.model.params.cross_attn_block.params.n_layer

        # Tokens and embeddings
        self.cls_tok = nn.Parameter(torch.randn(1, 1, d_model), requires_grad=True)
        self.seg_emb = SegmentEmbedding(embed_size=d_model)

        self.id_bottleneck = IdentityBottleneck(
            d_model=d_model, n_head=n_head,
            num_queries=num_id_queries, num_layers=num_id_layers, dropout=dropout
        )

        self.cross_attention_blocks = nn.ModuleList([
            CrossAttentionBlock(d_model, n_head, dropout) for _ in range(num_cross_attn_layers)
        ])
        
        # Classifier heads
        self.cls_head_rf = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(d_model // 2, 2)
        )

        self.cls_head_id = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(d_model // 2, 2)
        )

        # Load weights
        if ckpt_path:
            if ckpt_path and Path(ckpt_path).resolve() == PRETRAINED_PATH.resolve():
                print("Loading pretrained backbone weights only...")
                _load_backbone_weights(self, ckpt_path)
                
                print("Initializing modules and parameters...")
                self.id_bottleneck.apply(init_weights)
                self.cross_attention_blocks.apply(init_weights)
                self.cls_head_rf.apply(init_weights)
                self.cls_head_id.apply(init_weights)

            else:
                print(f"Loading full model checkpoint from {ckpt_path}...")
                _load_full_model(self, ckpt_path)
        

    def _process_input(self, vis: torch.Tensor, aud: torch.Tensor) -> torch.Tensor:
        """Extract and project features from Synchformer backbone."""
        vis_feats = self.backbone.extract_vfeats(vis, for_loop=False)
        aud_feats = self.backbone.extract_afeats(aud, for_loop=False)

        vis_proj = self.backbone.vproj(vis_feats)
        aud_proj = self.backbone.aproj(aud_feats)

        B, S, tv, D = vis_proj.shape
        B, S, ta, D = aud_proj.shape
        vis_flat, aud_flat = vis_proj.view(B, S * tv, D), aud_proj.view(B, S * ta, D)
        
        transformer = self.backbone.transformer

        v = transformer.vis_in_lnorm(vis_flat)
        a = transformer.aud_in_lnorm(aud_flat)

        if transformer.tok_pdrop > 0:
            v, a = transformer.tok_drop_vis(v), transformer.tok_drop_aud(a)
        return v, a  # (B,Tv,D), (B,Ta,D)


    def forward(self, 
                target_vis: torch.Tensor, target_aud: torch.Tensor, 
                ref_vis: torch.Tensor, ref_aud: torch.Tensor,
                label_rf: torch.Tensor = None,
                label_id: torch.Tensor = None):
        
        transformer = self.backbone.transformer

        # Encode target & reference
        tgt_v, tgt_a = self._process_input(target_vis, target_aud) # (B,Tv,D), (B,Ta,D)
        ref_v, ref_a = self._process_input(ref_vis, ref_aud)

        # Sequence Construction : [OFF, V, MOD, A]
        B = tgt_v.size(0)
        off_tok = transformer.OFF_tok.expand(B, -1, -1)
        mod_tok = transformer.MOD_tok.expand(B, -1, -1)

        orig_tgt = torch.cat((off_tok, tgt_v, mod_tok, tgt_a), dim=1) # (B, 1+Tv+1+Ta, D)
        orig_ref = torch.cat((off_tok, ref_v, mod_tok, ref_a), dim=1)

        if hasattr(transformer, 'pos_emb_cfg'):
            orig_tgt = transformer.pos_emb_cfg(orig_tgt)
            orig_ref = transformer.pos_emb_cfg(orig_ref)

        orig_tgt = orig_tgt[:, 1:, :]
        orig_ref = orig_ref[:, 1:, :]

        # Identity bottleneck
        id_tgt = self.id_bottleneck(orig_tgt)
        id_ref = self.id_bottleneck(orig_ref)

        # Cross-attention between ID queries
        for block in self.cross_attention_blocks:
            tgt_id_query = block(query=id_tgt, key_value=id_ref)
        id_feat = tgt_id_query.mean(dim=1)

        # Add segment embeddings
        cls_tok = self.cls_tok.expand(B, -1, -1)
        cls_label = torch.zeros(B, 1, dtype=torch.long, device=cls_tok.device)
        cls_tok = cls_tok + self.seg_emb(cls_label)
        id_label = torch.ones(B, tgt_id_query.size(1), dtype=torch.long, device=tgt_id_query.device)
        tgt_id_query = tgt_id_query + self.seg_emb(id_label)

        # Final sequence: [CLS, ID, V, MOD, A]
        x_tgt = torch.cat((cls_tok, tgt_id_query, orig_tgt), dim=1) # (B, 1+Q+Tv+1+Ta, D)
        x = transformer.drop(x_tgt)
        x = transformer.blocks(x)
        x = transformer.ln_f(x)

        # Classification heads
        cls_token = x[:, 0, :]
        logits_rf = self.cls_head_rf(cls_token)
        logits_id = self.cls_head_id(id_feat)

        if label_rf is not None and label_id is not None:
            loss_rf = F.cross_entropy(logits_rf, label_rf)
            loss_id = F.cross_entropy(logits_id, label_id)
            return loss_rf, logits_rf, loss_id, logits_id
        else:
            return logits_rf, logits_id