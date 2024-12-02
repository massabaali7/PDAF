import torch
import torch.nn as nn
from encoder.mha import MultiHeadAttention
from encoder.attentive_pooling import SelfAttentionPooling

class TransformerSelfAttention(nn.Module):
    def __init__(self, input_dim, num_heads, dim_feedforward,number_Of_spks):
        """EncoderBlock.

        Args:
            input_dim: Dimensionality of the input
            num_heads: Number of heads to use in the attention block
            dim_feedforward: Dimensionality of the hidden layer in the MLP
        """
        super().__init__()
        # Attention layer
        self.self_attn = MultiHeadAttention(input_dim, num_heads)

        
        self.attn_pooling = SelfAttentionPooling(input_dim)
        self.emb = nn.Linear(input_dim*2, dim_feedforward*8) 
        self.bn = nn.BatchNorm1d(dim_feedforward*8)
        self.act = nn.ReLU(inplace=True)
        # Layers to apply in between the main layers
        self.classifier = nn.Linear(dim_feedforward*8, number_Of_spks)

    def forward(self, x, prob_phn=None, mask=None, lambda_val=None):
        # Attention part
        attn_out, attn_mask = self.self_attn(x, x, x, prob_phn=prob_phn, mask=mask, lambda_val=lambda_val)
        #print(attn_mask.shape)
        attn_mask= attn_mask.squeeze(1)
        #print(attn_mask.shape)

        attn_out_mean,attn_out_std = self.attn_pooling(attn_out,attn_mask)
        attn_concat = torch.cat((attn_out_mean, attn_out_std),dim=1).to(dtype=torch.float32)
        
        emb = self.emb(attn_concat).to(dtype=torch.float32)
        emb = self.bn(emb)
        emb = self.act(emb)
        x = self.classifier(emb)
        return x,emb