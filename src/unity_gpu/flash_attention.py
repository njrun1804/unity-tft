import torch
import torch.nn.functional as F
from pytorch_forecasting.models.temporal_fusion_transformer.sub_modules import InterpretableMultiHeadAttention

class FlashMHA(InterpretableMultiHeadAttention):
    def _apply_attention(self, q, k, v, attn_mask=None):
        # Use PyTorch 2.2 SDPA, which routes to FlashAttention if available
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask, dropout_p=self.dropout, is_causal=False
        )
