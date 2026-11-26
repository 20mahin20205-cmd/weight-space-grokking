"""Grokking baseline: (a+b) mod p, 1-layer transformer, AdamW full-batch.

Follows the Nanda-style setup: no LayerNorm, causal-free (decoder sees all
three tokens), AdamW with high weight decay drives the grokking.

Usage:
  python train.py --p 97 --wd 1.0 --lr 1e-3 --frac 0.4 --seed 0 \
      --max-steps 30000 --out runs/pilot_p97_wd1_s0
"""
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seeds(init_seed: int, data_seed: int):
    torch.manual_seed(init_seed)
    rng = np.random.default_rng(data_seed)
    return rng


def make_data(p: int, frac: float, data_seed: int):
    rng = np.random.default_rng(data_seed)
    pairs = np.array([(a, b) for a in range(p) for b in range(p)], dtype=np.int64)
    labels = (pairs[:, 0] + pairs[:, 1]) % p
    idx = rng.permutation(len(pairs))
    n_train = int(round(frac * len(pairs)))
    train_idx, test_idx = idx[:n_train], idx[n_train:]
    return (pairs[train_idx], labels[train_idx],
            pairs[test_idx], labels[test_idx])


class ModAddTransformer(nn.Module):
    """1-layer transformer: input = [a_token, b_token, equals_token] -> p-way logits."""

    def __init__(self, p: int, d_model: int = 128, n_heads: int = 4,
                 d_mlp: int | None = None, init_scale: float = 1.0):
        super().__init__()
        d_mlp = d_mlp or 4 * d_model
        self.p = p
        self.embed = nn.Embedding(p + 1, d_model)  # +1 for '=' token
        self.pos_embed = nn.Embedding(3, d_model)
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.mlp_in = nn.Linear(d_model, d_mlp, bias=False)
        self.mlp_out = nn.Linear(d_mlp, d_model, bias=False)
        self.unembed = nn.Linear(d_model, p, bias=False)
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        for mod in (self.mlp_in, self.mlp_out, self.q, self.k, self.v, self.unembed):
            nn.init.normal_(mod.weight, std=init_scale * (2.0 / d_model) ** 0.5)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (B, 3) integer
        B = tokens.shape[0]
        x = self.embed(tokens) + self.pos_embed.weight[None, :3, :]
        q, k, v = self.q(x), self.k(x), self.v(x)
        q = q.view(B, 3, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, 3, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, 3, self.n_heads, self.d_head).transpose(1, 2)
        att = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(self.d_head), dim=-1)
        h = (att @ v).transpose(1, 2).reshape(B, 3, -1)
        x = x + self.mlp_out(F.relu(self.mlp_in(h)))
        return self.unembed(x[:, -1, :])  # logits from last position


def maybe_ckpt(path: str, obj: dict):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def run(args):
    out = args.out
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    dev = "cpu"
    torch.set_num_threads(args.threads)
    if args.threads == 0:
        torch.set_num_threads(os.cpu_count() or 1)

    rng = set_seeds(args.seed, args.data_seed)
    xtr, ytr, xte, yte = make_data(args.p, args.frac, args.data_seed)
    xtr_t = torch.from_numpy(xtr).to(dev)
    ytr_t = torch.from_numpy(ytr).to(dev)
    xte_t = torch.from_numpy(xte).to(dev)
    yte_t = torch.from_numpy(yte).to(dev)

    torch.manual_seed(args.seed)
    model = ModAddTransformer(args.p, args.d_model, args.n_heads,
                              init_scale=args.init_scale).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.wd, betas=(0.9, 0.98))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 100))  # 100-step warmup

    eq = torch.full((xtr_t.shape[0], 1), args.p, dtype=torch.long, device=dev)
    eq_te = torch.full((xte_t.shape[0], 1), args.p, dtype=torch.long, device=dev)
    inp_tr = torch.cat([xtr_t, eq], dim=1)
    inp_te = torch.cat([xte_t, eq_te], dim=1)

    resume_step = 0
    metrics = []
    T_mem = T_grok = None
    grok_streak = 0
    if os.path.exists(f"{out}/state.pt") and not args.fresh:
        st = torch.load(f"{out}/state.pt", weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        resume_step = st["step"]
        T_mem, T_grok, grok_streak = st["T_mem"], st["T_grok"], st["grok_streak"]
        metrics = st["metrics"]
        print(f"resumed from step {resume_step}")

    eval_every = args.eval_every

    def evaluate(step):
        nonlocal T_mem, T_grok, grok_streak
        model.eval()
        with torch.no_grad():
            ltr = F.cross_entropy(model(inp_tr), ytr_t).item()
            lte = F.cross_entropy(model(inp_te), yte_t).item()
            atr = (model(inp_tr).argmax(-1) == ytr_t).float().mean().item()
            ate = (model(inp_te).argmax(-1) == yte_t).float().mean().item()
        model.train()
        metrics.append(dict(step=step, train_loss=ltr, test_loss=lte,
                            train_acc=atr, test_acc=ate))
        if T_mem is None and atr >= 0.99:
            T_mem = step
        if T_grok is None:
            grok_streak = grok_streak + 1 if ate >= 0.95 else 0
            if grok_streak >= 3:
                T_grok = step - 2 * eval_every
        return ltr, lte, atr, ate

    # eval at resume point for streak state
    evaluate(resume_step)
    metrics.pop()  # don't duplicate

    snap_times = []
    for step in range(resume_step + 1, args.max_steps + 1):
        logits = model(inp_tr)
        loss = F.cross_entropy(logits, ytr_t)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()

        if step % eval_every == 0 or step == args.max_steps:
            ltr, lte, atr, ate = evaluate(step)
            if step % (eval_every * 10) == 0 or step in (1, args.max_steps):
                print(f"step {step:6d}  train_loss {ltr:7.4f}  test_loss {lte:7.4f}"
                      f"  train_acc {atr:.3f}  test_acc {ate:.3f}  ({time.time()-t0:.0f}s)")
            if args.snapshot_every and (step % args.snapshot_every == 0):
                snap_dir = f"{out}/snapshots"
                os.makedirs(snap_dir, exist_ok=True)
                sd = {k: v.detach().to(torch.float16).cpu()
                      for k, v in model.state_dict().items()}
                torch.save(sd, f"{snap_dir}/step{step:07d}.pt")
                snap_times.append(step)

        if step % args.ckpt_every == 0 or step == args.max_steps:
            maybe_ckpt(f"{out}/state.pt", dict(
                step=step, model=model.state_dict(), opt=opt.state_dict(),
                sched=sched.state_dict(), T_mem=T_mem, T_grok=T_grok,
                grok_streak=grok_streak, metrics=metrics))

        if T_grok is not None and T_grok + args.post_steps < step:
            break

    wall = time.time() - t0
    meta = dict(p=args.p, d_model=args.d_model, n_heads=args.n_heads,
                seed=args.seed, data_seed=args.data_seed, lr=args.lr, wd=args.wd,
                frac=args.frac, init_scale=args.init_scale,
                max_steps=args.max_steps, eval_every=eval_every,
                T_mem=T_mem, T_grok=T_grok,
                delay=(T_grok - T_mem) if (T_grok and T_mem) else None,
                wall_seconds=wall, torch=torch.__version__,
                threads=torch.get_num_threads(), snap_steps=snap_times,
                grokked=T_grok is not None)
    with open(f"{out}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    with open(f"{out}/metrics.json", "w") as f:
        json.dump(metrics, f)
    print(json.dumps({k: meta[k] for k in
                      ("T_mem", "T_grok", "delay", "wall_seconds", "grokked")}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=97)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1.0)
    ap.add_argument("--frac", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-seed", type=int, default=1000)
    ap.add_argument("--init-scale", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=30000)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--snapshot-every", type=int, default=500)
    ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--post-steps", type=int, default=2000,
                    help="keep training this many steps past T_grok")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--out", type=str, required=True)
    run(ap.parse_args())
