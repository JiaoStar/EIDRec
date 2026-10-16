from collections import defaultdict
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn
import faiss
from modules import *
from utils import extract



class SpectralDiffusionFilter(nn.Module):
    def __init__(self, hidden_size, tau_init=0.1, lambda_scale=1.0, residual=0.5):
        super(SpectralDiffusionFilter, self).__init__()
        self.hidden_size = hidden_size
        self.lambda_scale = lambda_scale
        self.residual = residual

        tau_init = max(float(tau_init), 1e-6)
        raw_tau_init = np.log(np.expm1(tau_init))
        self.raw_tau = nn.Parameter(torch.full((hidden_size,), raw_tau_init, dtype=torch.float32))

    def forward(self, x):
        B, L, D = x.shape
        original_device = x.device
        original_dtype = x.dtype
        x_float = x.float()

        try:
            fft_input = x_float
            X_f = torch.fft.rfft(fft_input, dim=1)
        except RuntimeError as err:
            if "cuFFT" not in str(err):
                raise
            # Some CUDA/cuFFT builds fail on small batched 1D FFT workloads.
            # CPU FFT is slower, but keeps this optional filter stable.
            fft_input = x_float.cpu()
            X_f = torch.fft.rfft(fft_input, dim=1)
        num_freq = X_f.size(1)

        k = torch.arange(num_freq, device=fft_input.device).float()
        lambda_k = (2 * torch.pi * k / L) ** 2
        lambda_k = self.lambda_scale * lambda_k

        tau = F.softplus(self.raw_tau).to(device=fft_input.device)
        multiplier = torch.exp(-lambda_k.view(1, num_freq, 1) * tau.view(1, 1, D))

        X_f_filtered = X_f * multiplier
        x_filtered = torch.fft.irfft(X_f_filtered, n=L, dim=1)

        residual = min(max(self.residual, 0.0), 1.0)
        x_out = residual * x_filtered + (1.0 - residual) * fft_input
        return x_out.to(device=original_device, dtype=original_dtype)


class Diffusion():
    def __init__(self, args):
        self.timesteps = args.timesteps
        self.beta_start = args.beta_start
        self.beta_end = args.beta_end
        self.args = args

        if self.args.beta_sche == 'linear':
            self.betas = self.linear_beta_schedule(self.timesteps, self.beta_start, self.beta_end)
        elif self.args.beta_sche == 'exp':
            self.betas = self.exp_beta_schedule(self.timesteps)
        elif self.args.beta_sche =='cosine':
            self.betas = self.cosine_beta_schedule(self.timesteps)
        elif self.args.beta_sche =='sqrt':
            self.betas = torch.tensor(self.betas_for_alpha_bar(self.timesteps, lambda t: 1-np.sqrt(t + 0.0001),)).float()

        # define alphas（timesteps, ）
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod - 1)


        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def linear_beta_schedule(self, timesteps, beta_start, beta_end):
        beta_start = beta_start
        beta_end = beta_end
        return torch.linspace(beta_start, beta_end, timesteps)

    def cosine_beta_schedule(self, timesteps, s=0.008):
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def exp_beta_schedule(self, timesteps, beta_min=0.1, beta_max=10):
        x = torch.linspace(1, 2 * timesteps + 1, timesteps)
        betas = 1 - torch.exp(- beta_min / timesteps - x * 0.5 * (beta_max - beta_min) / (timesteps * timesteps))
        return betas

    def betas_for_alpha_bar(self, num_diffusion_timesteps, alpha_bar, max_beta=0.999):
        betas = []
        for i in range(num_diffusion_timesteps):
            t1 = i / num_diffusion_timesteps
            t2 = (i + 1) / num_diffusion_timesteps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
        return np.array(betas)


class EIDRec(nn.Module):

    def __init__(self, device, args, num_heads=1):
        super(EIDRec, self).__init__()
        self.args = args
        self.state_size = args.max_seq_length
        self.hidden_size = args.hidden_size
        self.item_num = args.item_size
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.diffuser_type = args.diffuser_type
        self.device = device
        self.w = args.w

        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)

        self.seq_model = SASRecModel(self.item_embeddings, args=args)

        self.diff = Diffusion(args)

        if args.spectral_view_filter:
            self.spectral_filter = SpectralDiffusionFilter(
                hidden_size=args.hidden_size,
                tau_init=args.spectral_tau_init,
                lambda_scale=args.spectral_lambda_scale,
                residual=args.spectral_filter_residual
            )
        else:
            self.spectral_filter = None

        self.none_embedding = nn.Embedding(
            num_embeddings=1,
            embedding_dim=self.hidden_size,
        )
        nn.init.normal_(self.none_embedding.weight, 0, 1)


        self.step_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size*2),
            nn.GELU(),
            nn.Linear(self.hidden_size*2, self.hidden_size),
        )

        self.emb_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size*2)
        )

        self.diff_mlp = nn.Sequential(
            nn.Linear(self.hidden_size * 3, self.hidden_size*2),
            nn.GELU(),
            nn.Linear(self.hidden_size*2, self.hidden_size),
        )

        if self.diffuser_type =='mlp1':
            self.diffuser = nn.Sequential(
                nn.Linear(self.hidden_size*3, self.hidden_size)
        )
        elif self.diffuser_type =='mlp2':
            self.diffuser = nn.Sequential(
            nn.Linear(self.hidden_size * 3, self.hidden_size*2),
            nn.GELU(),
            nn.Linear(self.hidden_size*2, self.hidden_size)
        )
            
    def forward(self, x, s, step):
        #  x:(batch, dim) , h:(batch, dim), t:(batch, )
        t = self.step_mlp(step) # t:(batch, dim)


        if self.diffuser_type == 'mlp1':
            res = self.diffuser(torch.cat((x, s, t), dim=1)) # (batch, 3 x dim) -->diffuser-->(batch, dim)
        elif self.diffuser_type == 'mlp2':
            res = self.diffuser(torch.cat((x, s, t), dim=1))
        return res # (batch, dim)
    
    def forward_uncon(self, x, step):
        h = self.none_embedding.weight[0].unsqueeze(0).expand(x.shape[0], -1)

        t = self.step_mlp(step)

        if self.diffuser_type == 'mlp1':
            res = self.diffuser(torch.cat((x, h, t), dim=1))
        elif self.diffuser_type == 'mlp2':
            res = self.diffuser(torch.cat((x, h, t), dim=1))
            
        return res
    
    def calculate_x0(self, x0):
        return self.item_embeddings(x0)

    def filter_generated_view(self, original_seq_hidden, generated_last_rep):
        """
        original_seq_hidden: [B, L, D]
        generated_last_rep: [B, D]
        return: filtered generated representation [B, D]
        """
        if self.spectral_filter is None:
            return generated_last_rep

        aug_seq = original_seq_hidden.detach().clone()
        aug_seq[:, -1, :] = generated_last_rep
        filtered_seq = self.spectral_filter(aug_seq)
        return filtered_seq[:, -1, :]

    def calculate_s(self, input_seq, p):
        
        seq_out = self.seq_model(input_seq)
        s = seq_out[:,-1,:] # B x Dim 

        B, D = s.shape[0], s.shape[1] # B = batch_size，D = embedding_dim
        mask1d = (torch.sign(torch.rand(B) - p) + 1) / 2 # (B,) 
        maske1d = mask1d.view(B, 1) # (B, 1) 。
        mask = torch.cat([maske1d] * D, dim=1) 
        mask = mask.to(self.device)

        s = s * mask + self.none_embedding(torch.tensor([0], device=self.device)) * (1 - mask)

        return s

    def calculate_entropy_aware_s(self, input_seq, p, apply_dropout=True):
        seq_out = self.seq_model(input_seq)
        q = seq_out[:, -1, :]
        valid_mask = input_seq > 0

        h_norm = F.normalize(seq_out, p=2, dim=-1)
        q_norm = F.normalize(q, p=2, dim=-1)
        scores = torch.sum(h_norm * q_norm.unsqueeze(1), dim=-1)
        scores = scores / max(float(self.args.guide_entropy_temperature), 1e-8)
        scores = scores.masked_fill(valid_mask == 0, -1e9)

        prob = torch.softmax(scores, dim=-1)
        entropy = -prob * torch.log(prob + 1e-12)

        guide_logits = scores - self.args.guide_entropy_lambda * entropy
        guide_logits = guide_logits.masked_fill(valid_mask == 0, -1e9)
        weights = torch.softmax(guide_logits, dim=-1)

        s = torch.sum(weights.unsqueeze(-1) * seq_out, dim=1)

        if apply_dropout:
            B, D = s.shape
            mask1d = (torch.sign(torch.rand(B, device=s.device) - p) + 1) / 2
            mask = mask1d.view(B, 1).expand(B, D)
            s = s * mask + self.none_embedding(torch.tensor([0], device=s.device)) * (1 - mask)

        return s

    def calculate_guidance_s(self, input_seq, p, apply_dropout=True):
        if getattr(self.args, "guide_mode", "entropy") == "last":
            if apply_dropout:
                return self.calculate_s(input_seq, p)
            seq_out = self.seq_model(input_seq)
            return seq_out[:, -1, :]
        return self.calculate_entropy_aware_s(input_seq, p, apply_dropout=apply_dropout)

    def forward_process(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = extract(self.diff.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.diff.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    @torch.no_grad()
    def sample_from_reverse_process(self, s):
        x = torch.randn_like(s)
        time_pairs = self.make_reverse_time_pairs()
        for time, time_next in time_pairs:
            t = torch.full((s.shape[0], ), time, device=self.device, dtype=torch.long)
            x = self.ddim_sample_with_guidance(
                self.forward,
                self.forward_uncon,
                x,
                s,
                t,
                time_next,
            )

        return x

    def make_reverse_time_pairs(self):
        sample_steps = int(getattr(self.args, "sample_steps", 0) or 0)
        if sample_steps <= 0 or sample_steps >= self.diff.timesteps:
            steps = list(reversed(range(0, self.diff.timesteps)))
        else:
            steps = torch.linspace(
                self.diff.timesteps - 1,
                0,
                steps=sample_steps,
                device=self.device,
            ).round().long().unique(sorted=True).tolist()
            steps = list(reversed(steps))

        return list(zip(steps, steps[1:] + [-1]))

    def alpha_bar_at(self, time, x_shape, batch_size):
        if time < 0:
            return torch.ones(
                batch_size,
                *((1,) * (len(x_shape) - 1)),
                device=self.device,
            )
        t = torch.full((batch_size, ), time, device=self.device, dtype=torch.long)
        return extract(self.diff.alphas_cumprod, t, x_shape)
    
    @torch.no_grad()
    def ddim_sample_with_guidance(self, model_forward, model_forward_uncon, x, s, t, time_next):
        x_start = (1 + self.w) * model_forward(x, s, t) - self.w * model_forward_uncon(x, t)

        alpha = extract(self.diff.alphas_cumprod, t, x.shape)
        alpha_next = self.alpha_bar_at(time_next, x.shape, x.shape[0])
        noise_scale = torch.sqrt(torch.clamp(1. - alpha, min=1e-12))
        pred_noise = (x - torch.sqrt(alpha) * x_start) / noise_scale

        eta = max(float(getattr(self.args, "sample_eta", 1.0)), 0.0)
        sigma = eta * torch.sqrt(torch.clamp(
            (1. - alpha_next) / (1. - alpha) * (1. - alpha / alpha_next),
            min=0.0,
        ))
        model_mean = (
            torch.sqrt(alpha_next) * x_start +
            torch.sqrt(torch.clamp(1. - alpha_next - sigma ** 2, min=0.0)) * pred_noise
        )

        if time_next < 0 or eta == 0.0:
            return model_mean

        return model_mean + sigma * torch.randn_like(x)
	    
    @torch.no_grad()
    def p_sample_with_guidance(self, model_forward, model_forward_uncon, x, s, t, t_index):

        # guidance, (batch, dim)
        x_start = (1 + self.w) * model_forward(x, s, t) - self.w * model_forward_uncon(x, t)
        x_t = x

        # model_mean = (batch,1) x (batch, dim) + (batch,1) x (batch, dim) = (batch, dim)
        model_mean = (
            extract(self.diff.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.diff.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.diff.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)

            return model_mean + torch.sqrt(posterior_variance_t) * noise 


class KMeans(object):
    def __init__(self, num_cluster, seed, hidden_size, device="cpu"):
        """
        Args:
            k: number of clusters
        """
        self.seed = seed
        self.num_cluster = num_cluster
        self.max_points_per_centroid = 4096
        self.min_points_per_centroid = 0
        self.gpu_id = 0
        self.device = device
        self.first_batch = True
        self.hidden_size = hidden_size
        self.clus, self.index = self.__init_cluster(self.hidden_size)
        self.centroids = []
        
        self.seq2cluster = None  
        self.cluster2sequences = defaultdict(list)  


    def __init_cluster(
        self, hidden_size, verbose=False, niter=20, nredo=5, max_points_per_centroid=4096, min_points_per_centroid=0
    ):
        # print(" cluster train iterations:", niter)
        clus = faiss.Clustering(hidden_size, self.num_cluster)
        clus.verbose = verbose
        clus.niter = niter
        clus.nredo = nredo
        clus.seed = self.seed
        clus.max_points_per_centroid = max_points_per_centroid
        clus.min_points_per_centroid = min_points_per_centroid

        if torch.cuda.is_available() and hasattr(faiss, "StandardGpuResources"):
            res = faiss.StandardGpuResources()
            res.noTempMemory()
            cfg = faiss.GpuIndexFlatConfig()
            cfg.useFloat16 = False
            cfg.device = self.gpu_id
            index = faiss.GpuIndexFlatL2(res, hidden_size, cfg)
        else:
            index = faiss.IndexFlatL2(hidden_size)
        return clus, index

    def train(self, x, sequence_ids):
        self.cluster2sequences = defaultdict(list)
        self.seq2cluster = None

        # train to get centroids
        if x.shape[0] > self.num_cluster:
            self.clus.train(x, self.index)
        # get cluster centroids
        centroids = faiss.vector_to_array(self.clus.centroids).reshape(self.num_cluster, self.hidden_size)
        # convert to cuda Tensors for broadcast
        centroids = torch.Tensor(centroids).to(self.device)
        self.centroids = nn.functional.normalize(centroids, p=2, dim=1)

        D, I = self.index.search(x, 1)  # for each sample, find cluster distance and assignments
        seq2cluster = [int(n[0]) for n in I]
        self.seq2cluster = seq2cluster  

        for seq_id, cluster_id in zip(sequence_ids, seq2cluster):
            self.cluster2sequences[cluster_id].append(seq_id)

    def query(self, x):
        D, I = self.index.search(x, 1)  # for each sample, find cluster distance and assignments
        seq2cluster = [int(n[0]) for n in I]
        seq2cluster = torch.LongTensor(seq2cluster).to(self.device)
        return seq2cluster, self.centroids[seq2cluster]


class SASRecModel(nn.Module):
    def __init__(self, item_embeddings, args):
        super(SASRecModel, self).__init__()
        self.item_embeddings = item_embeddings
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.item_encoder = Encoder(args)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.args = args

        self.criterion = nn.BCELoss(reduction="none")
        self.apply(self.init_weights)

    def item_embedding(self, sequence):
        return self.item_embeddings(sequence)
    
    # Positional Embedding
    def add_position_embedding(self, sequence):

        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)

        item_embeddings = self.item_embeddings(sequence)
        position_embeddings = self.position_embeddings(position_ids)
        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        return sequence_emb

    # model same as SASRec
    def forward(self, input_ids):

        attention_mask = (input_ids > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()

        subsequent_mask = subsequent_mask.to(input_ids.device)

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        sequence_emb = self.add_position_embedding(input_ids)

        item_encoded_layers = self.item_encoder(sequence_emb, extended_attention_mask, output_all_encoded_layers=True)

        sequence_output = item_encoded_layers[-1]
        return sequence_output

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
