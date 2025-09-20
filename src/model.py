"""
This file is to create a ViT model, given the 
hyper-paramters in the configuration file.

# ┌──────────────────────────────────────────────────────────────────────────┐
# │ Vision Transformer (ViT) forward pipeline                                │
# └──────────────────────────────────────────────────────────────────────────┘
# 1) Split image into fixed-size patches
#        image ──► [Patches]
# 2) Linearly embed each patch (patch embedding / projection)
#        [Patches] ──► [Patch Embeddings]
# 3) Flatten patch embeddings into a token sequence
#        [Patch Embeddings] ──► [Tokens: (num_patches × d_model)]
# 4) Add learnable positional embeddings to preserve order
#        [Tokens] + [Positional Embeddings] ──► [Positional Tokens]
# 5) Feed the sequence through Transformer encoder layers
#        [Positional Tokens] ──► [Encoded Tokens]
# 6) Classification head (e.g., take [CLS] token or pooled output → MLP)
#        [Encoded Tokens] ──► [MLP Classifier] ──► [Logits]
# 7) Compute loss / prediction
#        [Logits] ──► [CrossEntropy / Predicted Class]



"""
import torch
import torch.nn as nn
import math

class PatchCreation(nn.Module):
    def __init__(self,
                 input_color_channel: int,
                 patch_size: int,
                 embedding_dimensions : int):
        super().__init__()
        self.patch_size = patch_size
        # self.output_channels = patch_size*patch_size*input_color_channel
        self.patching_conv = nn.Conv2d(input_color_channel,
                                    #    self.output_channels,
                                    embedding_dimensions,
                                       kernel_size=patch_size,
                                       stride=patch_size,
                                       padding=0)
        
        # self.patch_embedding_layer = nn.Linear(self.output_channels,embedding_dimensions)
        self.flatten = nn.Flatten(start_dim=2)

    
    def forward(self,x):
        # print(x.shape)
        image_dimension = x.shape[-1]
        assert image_dimension%self.patch_size == 0, f"Given image dimension {image_dimension} is not divisble into perfect number of patches of size {self.patch_size}"
        if len(x.shape)==3:
            x = x.unsqueeze(0)
        return self.patching_conv(x).flatten(2).permute(0,2,1)
 

class ViTInputLayer(nn.Module):
    def __init__(self,in_channels:int,
                 patch_size: int,
                 image_size: int,
                 embedding_dimensions: int,
                 input_dropout_rate: float = 0.0):
        super().__init__()
        self.patch_embeddings = PatchCreation(in_channels,
                                              patch_size,
                                              embedding_dimensions)
        # embed_dim = patch_size * patch_size * in_channels
        # batch independent
        self.cls_token = nn.Parameter(torch.randn(1,1,embedding_dimensions),requires_grad=True)

        num_patches = (image_size//patch_size) ** 2
        num_positions = num_patches + 1

        self.positional_embeddings = nn.Parameter(torch.randn(1,num_positions,embedding_dimensions),requires_grad=True)
        
        self.dropout = nn.Dropout(p=input_dropout_rate)

    def forward(self,x):
        batch_size = x.shape[0]
        patch_embeddings = self.patch_embeddings(x)
        cls_token = self.cls_token.expand(batch_size,-1,-1)
        
        return self.input_dropout(torch.concat((cls_token,patch_embeddings),dim=1) + self.positional_embeddings)
        # print(cls_token[0][0][0],cls_token[1][0][0])
        # print(patch_embeddings.shape,cls_token.shape)

    
class LayerNormalisation(nn.Module):
    """
    Custom normalisastion fucntion we are defining.
    
    """
    def __init__(self,
                 embed_dim: int,
                 eps: float = 10**-6,):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(embed_dim))
        self.bias = nn.Parameter(torch.zeros(embed_dim))
        # self.beta = nn.para

    def forward(self,x):
        mean = x.mean(dim=-1,keepdim=True)
        std = x.std(dim=-1,keepdim=True)
        return self.alpha * (x-mean)/(std + self.eps) + self.bias



class MultiHeadAttention(nn.Module):
    def __init__(self,
                 embedding_dimension: int,
                 head: int,
                 dropout_rate: float = 0.0
                 ):
        super().__init__()
        assert embedding_dimension%head == 0, f"Embedding dimensions {embedding_dimension} is not divisible into {head} heads"
        self.w_q = nn.Linear(embedding_dimension,embedding_dimension)
        self.w_k = nn.Linear(embedding_dimension,embedding_dimension)
        self.w_v = nn.Linear(embedding_dimension,embedding_dimension)
        self.head = head
        self.d_k = embedding_dimension // head # dimension of each head
        self.w_o = nn.Linear(embedding_dimension, embedding_dimension)

        self.attention_dropout = nn.Dropout(p = dropout_rate)
        self.proj_dropout = nn.Dropout(p = dropout_rate)

    @staticmethod
    def attention(q,k,v, dropout:nn.Dropout = None):
        d_k = q.shape[-1]
        # print(v.shape)
        
        attention_scores = (torch.matmul(q,k.transpose(-2,-1))) / math.sqrt(d_k)
        # print(attention_scores.shape)
        attention_scores = attention_scores.softmax(dim=-1)
        # print(attention_scores.shape)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # try:
        #     torch.matmul(attention_scores,v)
        # except:
        #     print("error in here")

        return torch.matmul(attention_scores,v), attention_scores


    def forward(self,q,k,v):
        # print("first : ",q.shape)
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)
        

        # want to divide the embeddign dimensions into the number of heads in orfer to caclualte the self
        # attetnion

        #current dimension will be
        #.   (batch_size,no.of patches, embedding dimension)
        # then dimension will be (embedding dimension broken into equi diemnsion head)
        #     (batch_size, no.of patches , head, d_k)
        # then to work on each head we want to fead input as
        #.    (batch_size, head, no. of patches, d_k)

        query = query.view(query.shape[0],query.shape[1],self.head, self.d_k).transpose(1,2)
        key = key.view(key.shape[0],key.shape[1],self.head,self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.head, self.d_k).transpose(1,2)
        

        # contextualised representation of given input image emebeedings
        x, self.attention_score = MultiHeadAttention.attention(query,key,value,dropout=self.attention_dropout)
        # print("goot the attentin score")
        # print("attention output shape : ", x.shape)

        # returning back to the original shape
        # current shape = (batchsize, heads, no.of patches, d_k)
        # then in next step -> (batch size, no. of patches, head, d_k)
        # finally -> (batche size, no. of patches, embnedding_dimension)
        # print('in here: ',x.transpose(1,2).contiguous().shape)
        # print(self.head*self.d_k)
        try:
            # x = x.transpose(1,2).contiguous().view(x.shape[0],x.shape[1],self.head*self.d_k)
            x = x.transpose(1,2).contiguous()
            x = x.view(x.shape[0], x.shape[1], self.head * self.d_k)
        except:
            print("real error is in here")
        # print("reshapeing shape: ",x.shape)

        

        return self.proj_dropout(self.w_o(x))
    
class FeedForwardLayer(nn.Module):
    def __init__(self,d_model:int
                 ,d_ff_scale:int = 2,
                 dropout_rate:float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model,d_ff_scale*d_model),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_ff_scale*d_model,d_model),
            nn.Dropout(dropout_rate)
        )

    def forward(self,x):
        return self.mlp(x)
    
class EncoderBlock(nn.Module):
    def __init__(self,
                 embedding_dimensions,
                 heads,
                 attention_dropout_rate: float,
                 feed_forward_dropout_rate: float,
                    dff_scale:int = 2):
        super().__init__()

        # this is seperate from the the patch creation section
        # so input will be patch created
        # input dimension (bnatch size, no.of patches, embedding dimensions)
        
        # it wil have input passing through normalisation and then the MHSA
        # for residual connection copy of input getting attached to it
        self.normalisation_stage1 = LayerNormalisation(embedding_dimensions)
        self.mhsa = MultiHeadAttention(embedding_dimensions,heads,attention_dropout_rate)

        # output of above two layers is expected to be same as input 
        # (batchsize, no. of patches, embedding dimensions)

        self.normalisation_stage2 = LayerNormalisation(embedding_dimensions)
        self.feed_forward_layer = FeedForwardLayer(embedding_dimensions,d_ff_scale=dff_scale,dropout_rate=feed_forward_dropout_rate)

    def forward(self, x):
        residual1 = x
        x = self.normalisation_stage1(x)
        x = self.mhsa(x,x,x) + residual1
        # print(x)
        residual2 = x
        return self.feed_forward_layer(self.normalisation_stage2(x)) + residual2

# total encoder blocks building

class Encoder(nn.Module):
    def __init__(self,num_of_encoders: int,
                        embeddings: int,
                        dff_scale: int,
                        heads: int,
                        attention_dropout_rate: float,
                        feed_forward_dropout_rate: float,
                        ):
        super().__init__()
        self.encoder_stack = nn.ModuleList(EncoderBlock(embedding_dimensions=embeddings,
                                                        heads=heads,
                                                        dff_scale=dff_scale,
                                                        attention_dropout_rate=attention_dropout_rate,
                                                        feed_forward_dropout_rate=feed_forward_dropout_rate) for _ in range(num_of_encoders))
        # print(self.encoder_stack)
        # for item in self.encoder_stack:
        #     print(item)
    
    def forward(self,x):
        for module in self.encoder_stack:
            x = module(x)
        return x
    
class VisionTransformer(nn.Module):
    def __init__(self,
                 in_channels: int,
                 image_size: int,
                 patch_size: int,
                 number_of_encoder: int,
                 embeddings: int,
                 d_ff_scale: int,
                 heads: int,
                 input_dropout_rate: float,
                 attention_dropout_rate: float,
                 feed_forward_dropout_rate: float,
                 number_of_classes: int
                 ):
        super().__init__()
        self.input_layer = ViTInputLayer(in_channels,
                                         patch_size,
                                         image_size,
                                         embeddings,
                                         input_dropout_rate=input_dropout_rate)
        self.encoder_stack = Encoder(number_of_encoder,
                                     embeddings,
                                     d_ff_scale,
                                     heads,
                                     attention_dropout_rate=attention_dropout_rate,
                                     feed_forward_dropout_rate=feed_forward_dropout_rate)
        self.classification_head = nn.Sequential(
            nn.LayerNorm([embeddings]),
            nn.Linear(embeddings,number_of_classes)

        )

    def forward(self,x):
        # x = x[:,0,:]
        # print(x.shape)
        return self.classification_head(self.encoder_stack(self.input_layer(x))[:,0,:])

  
