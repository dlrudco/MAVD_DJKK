import torch
import torch.nn as nn
from torch.nn.modules.module import Module
from einops import rearrange
from model.self_attention import FeedForward


class MaskedAttention(Module):
    # self_attention.Attention + mask: False 인 key 는 attention 에서 제외
    #   (B, N): key mask, 모든 query 공통 / (B, N, N): query 별 key mask
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x, mask=None):
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), self.to_qkv(x).chunk(3, dim = -1))
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if mask is not None:
            mask = mask[:, None, None, :] if mask.dim() == 2 else mask[:, None]
            dots = dots.masked_fill(~mask, float('-inf'))
        out = torch.matmul(dots.softmax(dim = -1), v)
        return self.to_out(rearrange(out, 'b h n d -> b n (h d)'))


class Block(Module):
    # pre-norm transformer layer, ffn=False 이면 attention 만 (bottleneck exchange 용)
    def __init__(self, dim, heads, dim_head, mlp_dim, ffn=True):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = MaskedAttention(dim, heads, dim_head)
        self.ff = nn.Sequential(nn.LayerNorm(dim), FeedForward(dim, mlp_dim)) if ffn else None

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm(x), mask)
        if self.ff is not None:
            x = x + self.ff(x)
        return x


class BottleneckFusion(Module):
    #   A (intra)   : 모달리티별로 [토큰; 자기 bottleneck] self-attn + FFN
    #   X (exchange): 모든 모달리티의 bottleneck 끼리만 self-attn
    #   A -> (X -> A) x layers, 출력은 입력과 같은 shape (bottleneck 은 버림)
    #   residual: out = in + proj_out(h), proj_out zero-init 이라 학습 시작 시 identity
    #   temporal_sa=False: A 에서 시간 토큰은 자기 자신 + bottleneck 만 봄 (시간 토큰끼리 SA 없음),
    #                      bottleneck 은 전체를 봄 -> 모달 간 교류는 bottleneck 으로만, 시간축 mixing 은 bottleneck 경유만
    def __init__(self, dims, dim=64, n_bottleneck=4, layers=1, heads=4, residual=True, temporal_sa=True):
        super().__init__()
        self.n_bottleneck = n_bottleneck
        self.residual = residual
        self.temporal_sa = temporal_sa

        self.bottleneck = nn.Parameter(torch.randn(len(dims), n_bottleneck, dim) * 0.02)
        self.proj_in = nn.ModuleList([nn.Linear(d, dim) for d in dims])
        # head / ffn 크기는 unimodal Transformer 관례 (dim_head = dim // 2, mlp_dim = dim)
        self.intra = nn.ModuleList([nn.ModuleList([Block(dim, heads, dim // 2, dim) for _ in dims])
                                    for _ in range(layers + 1)])
        self.exchange = nn.ModuleList([Block(dim, heads, dim // 2, dim, ffn=False) for _ in range(layers)])
        self.proj_out = nn.ModuleList([nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, d)) for d in dims])
        if residual:
            for proj in self.proj_out:
                nn.init.zeros_(proj[1].weight)
                nn.init.zeros_(proj[1].bias)

    def forward(self, xs, seq_len=None):
        # xs: 모달리티별 (B, T, D_m) 리스트 -> 같은 shape 의 리스트
        B, T, _ = xs[0].shape
        n = self.n_bottleneck

        valid, mask = None, None
        if seq_len is not None:
            # padding snippet (t >= seq_len) 은 key 에서 제외, bottleneck key 는 항상 보임
            valid = torch.arange(T, device=xs[0].device)[None] < seq_len.to(xs[0].device)[:, None]   # (B, T)
            mask = torch.cat([valid, valid.new_ones(B, n)], dim=1)                  # (B, T+n)
        if not self.temporal_sa:
            # 시간 토큰 row: 자기 자신 + bottleneck 만 True, bottleneck row: 전부 True
            allow = torch.eye(T + n, dtype=torch.bool, device=xs[0].device)
            allow[:, T:] = True
            allow[T:] = True
            mask = allow[None] if mask is None else allow[None] & mask[:, None, :]  # (1 or B, T+n, T+n)

        h = [proj(x) for proj, x in zip(self.proj_in, xs)]                          # M x (B, T, dim)
        b = list(self.bottleneck.unsqueeze(1).expand(-1, B, -1, -1))                # M x (B, n, dim)

        for i, blocks in enumerate(self.intra):
            if i > 0:
                b = list(self.exchange[i - 1](torch.cat(b, dim=1)).split(n, dim=1))  # (B, M*n, dim) -> M x (B, n, dim)
            for m, block in enumerate(blocks):
                z = block(torch.cat([h[m], b[m]], dim=1), mask)                     # (B, T+n, dim)
                h[m], b[m] = z[:, :T], z[:, T:]

        outs = [proj(hm) for proj, hm in zip(self.proj_out, h)]
        # 로깅용 보정량 크기 ||proj_out(h)|| / ||in|| (유효 snippet 기준, 0 이면 fusion 안 쓰이는 것)
        with torch.no_grad():
            if valid is None:
                valid = torch.ones(B, T, dtype=torch.bool, device=xs[0].device)
            self.delta_ratio = [o[valid].norm() / x[valid].norm().clamp_min(1e-8) for o, x in zip(outs, xs)]
        if self.residual:
            outs = [x + o for x, o in zip(xs, outs)]
        return outs
