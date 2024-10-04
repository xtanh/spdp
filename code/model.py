import torch
import torch.nn as nn
import torch.nn.functional as F

from module import PerResidueEncoder, ResiduePairEncoder, GAEncoder, GADecoder
from ipdb import set_trace

class MultiHeadAttentionInteract(nn.Module):
    """
        多头注意力的交互层
    """

    def __init__(self, embed_size, head_num, dropout, residual=True):
        """
        """
        super(MultiHeadAttentionInteract, self).__init__()
        self.embed_size = embed_size
        self.head_num = head_num
        self.dropout = dropout
        self.use_residual = residual
        self.attention_head_size = embed_size // head_num

        # 直接定义参数, 更加直观
        self.W_Q = nn.Parameter(torch.Tensor(embed_size, embed_size))
        self.W_K = nn.Parameter(torch.Tensor(embed_size, embed_size))
        self.W_V = nn.Parameter(torch.Tensor(embed_size, embed_size))

        if self.use_residual:
            self.W_R = nn.Parameter(torch.Tensor(embed_size, embed_size))

        # 初始化, 避免计算得到nan
        for weight in self.parameters():
            nn.init.xavier_uniform_(weight)

    def forward(self, x):
        """
            x : (batch_size, feature_fields, embed_dim)
        """

        # 线性变换到注意力空间中
        Query = torch.tensordot(x, self.W_Q, dims=([-1], [0]))
        Key = torch.tensordot(x, self.W_K, dims=([-1], [0]))
        Value = torch.tensordot(x, self.W_V, dims=([-1], [0]))

        # Head (head_num, bs, fields, D / head_num)
        Query = torch.stack(torch.split(Query, self.attention_head_size, dim=2))
        Key = torch.stack(torch.split(Key, self.attention_head_size, dim=2))
        Value = torch.stack(torch.split(Value, self.attention_head_size, dim=2))

        # 计算内积
        inner = torch.matmul(Query, Key.transpose(-2, -1))
        inner = inner / self.attention_head_size ** 0.5

        # Softmax归一化权重
        attn_w = F.softmax(inner, dim=-1)
        #         attn_w = entmax15(inner, dim=-1)

        attn_w = F.dropout(attn_w, p=self.dropout)

        # 加权求和
        results = torch.matmul(attn_w, Value)

        # 拼接多头空间
        results = torch.cat(torch.split(results, 1, ), dim=-1)
        results = torch.squeeze(results, dim=0)  # (bs, fields, D)

        # 残差学习
        if self.use_residual:
            results = results + torch.tensordot(x, self.W_R, dims=([-1], [0]))

        results = F.relu(results)

        return results


class DualInteract(nn.Module):

    def __init__(self, field_dim, embed_size, head_num, dropout=0.1):
        super(DualInteract, self).__init__()

        hidden_dim = 256

        self.vec_wise_net = MultiHeadAttentionInteract(embed_size=embed_size,#128
                                                       head_num=head_num,#8
                                                       dropout=dropout)
        self.trans_vec_nn = nn.Sequential(
            nn.LayerNorm(field_dim * embed_size),
            nn.Linear(field_dim * embed_size, hidden_dim),
            # nn.ReLU(),
            nn.GELU(),
            nn.Linear(hidden_dim, field_dim * embed_size),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
            x : batch, field_dim, embed_dim
        """

        b, f, e = x.shape
        vec_wise_x = self.vec_wise_net(x).reshape(b, f * e)

        m_vec = self.trans_vec_nn(vec_wise_x)
        m_x = m_vec + x.reshape(b,f * e)
        return m_x
    
class DDG_RDE_Network(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.args = args
        input_dim = args.node_feat_dim

        self.single_encoder = PerResidueEncoder(feat_dim=args.node_feat_dim, args=args)
        self.pair_encoder = ResiduePairEncoder(feat_dim=args.pair_feat_dim)
        self.attn_encoder = GAEncoder(args.node_feat_dim, args.pair_feat_dim, args.num_layers, args)
        
        self.esm2_encoder = nn.Sequential(nn.Linear(1280,512),
                                          nn.ReLU(),
                                          nn.Linear(512,256))

        self.ddg_readout = nn.Sequential(nn.Linear(512, args.node_feat_dim),    
                                         nn.ReLU(), 
                                         nn.Linear(args.node_feat_dim, args.node_feat_dim), 
                                         nn.ReLU(), 
                                         nn.Linear(args.node_feat_dim, 1)
                                         )   #  + args.esm_feature_dim
        self.single_fusion = nn.Sequential(nn.Linear(args.node_feat_dim, args.node_feat_dim), nn.ReLU(), nn.Linear(args.node_feat_dim, args.node_feat_dim))
        self.attn_weight = nn.Linear(3 * args.node_feat_dim, 3)
        
        self.feature_interact = DualInteract(field_dim=2, embed_size=256, head_num=4)
        self.transform = nn.Sequential(nn.LayerNorm(1024),
                                       nn.Linear(1024,512),
                                       nn.ReLU(),
                                       nn.Linear(512,128),
                                       nn.ReLU(),
                                       nn.Linear(128,1))
    def encode(self, batch, vae_model):
    
        h = self.single_encoder(batch)

        if vae_model is not None:
            e_list = vae_model.forward_encoder(batch)
            weight = self.attn_weight(torch.cat(e_list, dim=-1)).unsqueeze(dim=-2) # (N, L, 1, 3)
            e = (weight * torch.cat([e_list[0].unsqueeze(dim=-1), e_list[1].unsqueeze(dim=-1), e_list[2].unsqueeze(dim=-1)], dim=-1)).sum(dim=-1) # (N, L, F, 3) -> (N, L, F)

            intra_mask, inter_mask = self.attn_encoder.edge_construction(batch, cutoff=self.args.cutoff, K=self.args.knn)
            mask = intra_mask + inter_mask # (N, L, L)
            mask = (mask * batch['mut_flag'][:,None,:]).float().detach() # (N, L, L)
            
            eq = torch.einsum('ijk,ikn->ijn', mask, e) / (torch.sum(mask, dim=-1, keepdim=True) + 1e-6)
            h = self.single_fusion(h + eq)

        z = self.pair_encoder(batch)

        h = self.attn_encoder(batch, h, z)

        return h

    def forward(self, batch, vae_mode=None):
        batch_wt = {k: v for k, v in batch.items()}
        batch_mt = {k: v for k, v in batch.items()}


        batch_mt['aa'] = batch_mt['aa_mut']

        esm_embeddings_wt = batch_wt['esm_embeddings_wt']
        esm_embeddings_mut = batch_mt['esm_embeddings_mut']


        esm_embeddings_wt = self.esm2_encoder(esm_embeddings_wt)
        esm_embeddings_mut = self.esm2_encoder(esm_embeddings_mut)




        
        h_wt = self.encode(batch_wt, vae_mode)
        h_mt = self.encode(batch_mt, vae_mode)


        
        H_mt, H_wt = h_mt.max(dim=1)[0], h_wt.max(dim=1)[0]

        p_mt= torch.stack([H_mt, esm_embeddings_mut], dim = 1)
        p_mt = self.feature_interact(p_mt)


        p_wt= torch.stack([H_wt, esm_embeddings_wt], dim = 1)
        p_wt = self.feature_interact(p_wt)



        ddg_pred = self.ddg_readout(p_mt- p_wt).squeeze(-1)

        loss = F.mse_loss(ddg_pred, batch['ddG']) 

        out_dict = {
            'ddG_pred': ddg_pred,
            'ddG_true': batch['ddG'],
        }

        return loss, out_dict


class Codebook(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.single_encoder = PerResidueEncoder(feat_dim=args.node_feat_dim, args=args)
        self.pair_encoder = ResiduePairEncoder(feat_dim=args.pair_feat_dim)
        self.attn_encoder = GAEncoder(args.node_feat_dim, args.pair_feat_dim, args.num_layers, args, codebook_head=3)

        self.vq_layer_type = VectorQuantizer(args.node_feat_dim, args.num_embeddings, args.commitment_cost)
        self.vq_layer_angle = VectorQuantizer(args.node_feat_dim, args.num_embeddings, args.commitment_cost)
        self.vq_layer_coord = VectorQuantizer(args.node_feat_dim, args.num_embeddings, args.commitment_cost)

        self.pair_decoder = ResiduePairEncoder(feat_dim=args.pair_feat_dim)
        self.attn_decoder_type = GAEncoder(args.node_feat_dim, args.pair_feat_dim, int(args.num_layers/2), args)
        self.attn_decoder_angle = GAEncoder(args.node_feat_dim, args.pair_feat_dim, int(args.num_layers/2), args)
        self.attn_decoder_coord = GADecoder(args.node_feat_dim, args.pair_feat_dim, int(args.num_layers/2), args)

        self.aa_type_decoder = nn.Sequential(nn.Linear(args.node_feat_dim, args.node_feat_dim), nn.ReLU(), nn.Linear(args.node_feat_dim, 22))
        self.aa_angle_decoder = nn.Sequential(nn.Linear(args.node_feat_dim, args.node_feat_dim), nn.ReLU(), nn.Linear(args.node_feat_dim, 7*13))

        self.args = args

    def encode(self, batch):
    
        h, gt_mask = self.single_encoder(batch, pretrain=True)
        z = self.pair_encoder(batch)

        h = self.attn_encoder(batch, h, z)
        
        return h, gt_mask
    
    def decode(self, batch, e_type, e_angle, e_coord):
    
        z = self.pair_decoder(batch)

        h_type = self.aa_type_decoder(self.attn_decoder_type(batch, e_type, z))
        h_angle = self.aa_angle_decoder(self.attn_decoder_angle(batch, e_angle, z))
        h_coord, mask_coord = self.attn_decoder_coord(batch, e_coord, z)
        
        return h_type, h_angle, h_coord, mask_coord

    def forward(self, batch):
        batch_wt = {k: v for k, v in batch.items()}
        h_wt, gt_mask = self.encode(batch_wt)

        e_type, e_q_loss_type = self.vq_layer_type(h_wt[:, :, :self.args.node_feat_dim], batch['mask_atoms'][:, :, 1])
        e_angle, e_q_loss_angle = self.vq_layer_angle(h_wt[:, :, self.args.node_feat_dim : 2*self.args.node_feat_dim], batch['mask_atoms'][:, :, 1])
        e_coord, e_q_loss_coord = self.vq_layer_coord(h_wt[:, :, 2*self.args.node_feat_dim:], batch['mask_atoms'][:, :, 1])
        e_q_loss = e_q_loss_type + e_q_loss_angle + e_q_loss_coord

        h_type, h_angle, h_coord, mask_coord = self.decode(batch_wt, e_type, e_angle, e_coord)

        # three reconstruction losses
        s_recon_loss = F.cross_entropy(h_type.reshape(-1, 22), batch_wt['aa'].reshape(-1), reduction='none')
        s_recon_loss = torch.sum(s_recon_loss[gt_mask[1].reshape(-1)]) / gt_mask[1].sum()

        h_recon_loss = F.mse_loss(h_angle[gt_mask[2]], gt_mask[0][gt_mask[2]], reduction='mean')

        x_recon_loss = F.smooth_l1_loss(h_coord[mask_coord], batch['pos_atoms'][:, :, :4][mask_coord], reduction='mean')

        return [e_type.detach(), e_angle.detach(), e_coord.detach()], e_q_loss, s_recon_loss, h_recon_loss, x_recon_loss

    def forward_encoder(self, batch):
        batch_wt = {k: v for k, v in batch.items()}

        h = self.single_encoder(batch_wt, pretrain=False)
        z = self.pair_encoder(batch_wt)
        h_wt = self.attn_encoder(batch_wt, h, z)

        e_type, _ = self.vq_layer_type(h_wt[:, :, :self.args.node_feat_dim], batch['mask_atoms'][:, :, 1])
        e_angle, _ = self.vq_layer_angle(h_wt[:, :, self.args.node_feat_dim : 2*self.args.node_feat_dim], batch['mask_atoms'][:, :, 1])
        e_coord, _ = self.vq_layer_coord(h_wt[:, :, 2*self.args.node_feat_dim:], batch['mask_atoms'][:, :, 1])
        
        return [e_type.detach(), e_angle.detach(), e_coord.detach()]


class VectorQuantizer(nn.Module):
    """
    VQ-VAE layer: Input any tensor to be quantized. 
    Args:
        embedding_dim (int): the dimensionality of the tensors in the
        quantized space. Inputs to the modules must be in this format as well.
        num_embeddings (int): the number of vectors in the quantized space.
        commitment_cost (float): scalar which controls the weighting of the loss terms.
    """
    def __init__(self, embedding_dim, num_embeddings, commitment_cost):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        
        # initialize embeddings
        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        
    def forward(self, x, mask):    
        x = F.normalize(x, p=2, dim=-1)
        encoding_indices = self.get_code_indices(x)
        # print(torch.unique(encoding_indices).size(0))
        quantized = self.quantize(encoding_indices)

        q_latent_loss = F.mse_loss(quantized[mask], x.detach()[mask])
        e_latent_loss = F.mse_loss(x[mask], quantized.detach()[mask])
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = x + (quantized - x).detach().contiguous()

        return quantized, loss
    
    def get_code_indices(self, x):

        distances = (
            torch.sum(x ** 2, dim=-1, keepdim=True) +
            torch.sum(F.normalize(self.embeddings.weight, p=2, dim=-1).unsqueeze(0) ** 2, dim=-1) -
            2. * torch.matmul(x, F.normalize(self.embeddings.weight.t(), p=2, dim=0).unsqueeze(0))
        )
        
        encoding_indices = torch.argmin(distances, dim=-1)
        
        return encoding_indices
    
    def quantize(self, encoding_indices):
        """Returns embedding tensor for a batch of indices."""
        N, L = encoding_indices.size()
        return F.normalize(self.embeddings(encoding_indices.reshape(-1)).reshape(N, L, self.embedding_dim), p=2, dim=-1)
