import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import pad_sequences, to_grid
from .losses import TripletLoss
from .metrics import adc_search, l2_distances, recall_metrics
from .model import PDT_INPUT_NORM_EPS, TrajVQEncoder
from .raster import IMAGE_CHANNELS, build_rasters

WEIGHT_DECAY = 1e-4
TRIPLET_WINDOW = 200


def _warmup_weight(epoch, start, warmup, max_weight):
    if epoch < start:
        return 0.0
    if warmup <= 0:
        return max_weight
    return max_weight * min(1.0, float(epoch - start + 1) / float(warmup))


def _decay_weight(epoch, end, max_weight):
    if epoch >= end or end <= 0:
        return 0.0
    return max_weight * max(0.0, 1.0 - float(epoch) / float(max(1, end)))


class Trainer:
    def __init__(self, cfg, dataset, device, output_dir, backbone=None, teacher_dir=None):
        self.cfg = cfg
        self.device = torch.device(device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.backbone = backbone
        self.vq = cfg["stage"] == "vq"
        self.teacher_codes = self.teacher_codebook = None
        if teacher_dir is not None:
            self.teacher_codebook = np.load(Path(teacher_dir) / "teacher_codebook.npy").astype(np.float32, copy=False)
            self.teacher_codes = np.load(Path(teacher_dir) / "teacher_codes.npy").astype(np.int64, copy=False)

        self.trajectories, (self.n_train, self.n_query, self.n_base), train_dist, self.test_dist = dataset
        self.lon_ids, self.lat_ids, self.grid_shape, self.lon_coords, self.lat_coords = to_grid(self.trajectories)
        self.max_len = max(len(traj) for traj in self.trajectories)
        print("trajectories: {} train / {} query / {} base | grid {} | max length {}".format(
            self.n_train, self.n_query, self.n_base, self.grid_shape, self.max_len))

        n = self.n_train
        self.train_x, self.train_y, self.train_images = self._build_inputs(0, n)
        self.train_dist = train_dist[:n]
        self.train_knn = np.argsort(self.train_dist, axis=1).astype(np.int32)
        self.target_scale = None
        if cfg["metric_loss_norm"] == "target_train_mean":
            self.target_scale = float(max(np.mean(self.train_dist), 1e-8))
        if self.teacher_codes is not None:
            if len(self.teacher_codes) < n:
                raise ValueError("teacher_codes has {} rows, expected {}".format(len(self.teacher_codes), n))
            self.teacher_codes = self.teacher_codes[:n]

    def _build_inputs(self, begin, end):
        images = build_rasters(self.lon_ids[begin:end], self.lat_ids[begin:end], self.trajectories[begin:end],
                               self.grid_shape, self.cfg["image_mode"])
        return (torch.from_numpy(pad_sequences(self.lon_coords[begin:end], self.max_len)),
                torch.from_numpy(pad_sequences(self.lat_coords[begin:end], self.max_len)),
                torch.from_numpy(images))

    def build_model(self):
        cfg = self.cfg
        return TrajVQEncoder(self.grid_shape[0], self.grid_shape[1], self.max_len, IMAGE_CHANNELS[cfg["image_mode"]],
                             cfg["aux_kernel_sizes"], pdt_m=cfg["pdt_m"], pdt_k=cfg["pdt_k"], pdt_steps=cfg["pdt_steps"],
                             pdt_heads=cfg["pdt_heads"], pdt_input_norm=cfg["pdt_input_norm"]).to(self.device)

    def _embed(self, model, x, y, images, batch_size, quantize=False):
        model.eval()
        outputs = []
        with torch.no_grad():
            for i in range(0, len(x), batch_size):
                e = model.encode(x[i:i + batch_size].to(self.device), y[i:i + batch_size].to(self.device),
                                 images[i:i + batch_size].to(self.device))
                if quantize:
                    q = model.quantize(e)
                    outputs.append((q["z"].cpu(), q["codes"].cpu(), q["z_hat"].cpu()))
                else:
                    outputs.append(e.cpu())
        if quantize:
            return [torch.cat(parts, dim=0) for parts in zip(*outputs)]
        return torch.cat(outputs, dim=0)

    def _train_embeddings(self, model, batch_size, limit=None):
        limit = self.n_train if limit is None else limit
        return self._embed(model, self.train_x[:limit], self.train_y[:limit], self.train_images[:limit], batch_size)

    def _fit_pdt_input_normalizer(self, model):
        if bool(model.pdt_input_norm_ready.item()):
            return
        embeddings = self._train_embeddings(model, min(32, self.cfg["batch_size"])).float()
        mean = embeddings.mean(dim=0, keepdim=True)
        std = torch.clamp(embeddings.std(dim=0, keepdim=True, unbiased=False), min=PDT_INPUT_NORM_EPS)
        model.set_pdt_input_normalization(mean.to(self.device), std.to(self.device))

    def _init_codebook(self, model):
        quantizer = model.PDT_model.vq.quantizer
        if self.teacher_codebook is not None:
            with torch.no_grad():
                quantizer.codebook.copy_(torch.from_numpy(self.teacher_codebook).to(self.device))
        else:
            embeddings = self._train_embeddings(model, min(16, self.cfg["batch_size"])).to(self.device)
            with torch.no_grad():
                model.PDT_model.init_codebook(model.normalize_pdt_input(embeddings))
        with torch.no_grad():
            quantizer.log_temperatures.zero_()

    def _restart_dead_codes(self, model, optimizer):
        cfg = self.cfg
        codebook = model.PDT_model.vq.quantizer.codebook
        if not codebook.requires_grad:
            return
        pdt_m, pdt_k, dsub = codebook.shape
        limit = max(1, min(cfg["dead_code_sample_limit"], self.n_train))
        z, codes, z_hat = self._embed(model, self.train_x[:limit], self.train_y[:limit], self.train_images[:limit],
                                      min(cfg["batch_size"], 128), quantize=True)
        z = z.to(codebook.device).reshape(len(z), pdt_m, dsub)
        z_hat = z_hat.to(codebook.device).reshape(len(z_hat), pdt_m, dsub)
        codes = codes.to(codebook.device).long()
        group_error = torch.mean((z - z_hat) ** 2, dim=-1)

        restarted = []
        with torch.no_grad():
            for m in range(pdt_m):
                usage = torch.bincount(codes[:, m], minlength=pdt_k)
                dead = torch.nonzero(usage <= cfg["dead_code_min_usage"], as_tuple=False).flatten()
                dead = dead[:cfg["dead_code_max_per_group"]]
                worst = torch.argsort(group_error[:, m], descending=True)
                for offset, code in enumerate(dead):
                    replacement = z[worst[offset % worst.numel()], m].detach().clone()
                    if cfg["dead_code_noise"] > 0.0:
                        replacement = replacement + torch.randn_like(replacement) * cfg["dead_code_noise"]
                    codebook[m, int(code.item())].copy_(replacement)
                    restarted.append((m, int(code.item())))
        state = optimizer.state.get(codebook, {})
        for name in ("exp_avg", "exp_avg_sq"):
            if name in state:
                for m, code in restarted:
                    state[name][m, code].zero_()
        print("[dead codes] restarted {}".format(len(restarted)))

    def _build_optimizer(self, model, lr):
        main = [p for name, p in model.named_parameters() if p.requires_grad and name != "PDT_model.vq.quantizer.codebook"]
        codebook = model.PDT_model.vq.quantizer.codebook
        groups = [{"params": main, "lr": lr, "weight_decay": WEIGHT_DECAY}]
        if codebook.requires_grad:
            groups.append({"params": [codebook], "lr": lr, "weight_decay": WEIGHT_DECAY})
        return torch.optim.Adam(groups)

    def _sample_pair(self, anchor):
        knn, dist = self.train_knn[anchor], self.train_dist[anchor]
        hi = min(TRIPLET_WINDOW, max(0, len(knn) - 1))
        first = random.randint(0, hi)
        while dist[knn[first]] < 0.0001 and hi > 0:
            first = random.randint(0, hi)
        second = random.randint(0, hi)
        while dist[knn[second]] < 0.0001 and hi > 0:
            second = random.randint(0, hi)
        while second == first and hi > 0:
            second = random.randint(0, hi)
        if first > second:
            first, second = second, first
        return knn[first], knn[second]

    def _epoch_batches(self):
        batches = []
        for begin in range(0, self.n_train, self.cfg["batch_size"]):
            anchors = list(range(begin, min(begin + self.cfg["batch_size"], self.n_train)))
            pairs = [self._sample_pair(a) for a in anchors]
            batches.append((np.asarray(anchors, dtype=np.int64),
                            np.asarray([p for p, _ in pairs], dtype=np.int64),
                            np.asarray([n for _, n in pairs], dtype=np.int64)))
        return batches

    def _adc_latent(self, model, e):
        pdt = model.PDT_model
        z, soft_recon = pdt.soft_quantize_latent(model.normalize_pdt_input(e))
        hard_recon = pdt.reconstruction(pdt.get_codes(z))
        return z, soft_recon + (hard_recon.detach() - soft_recon).detach()

    def _adc_distance(self, query_z, base_recon):
        diff = query_z.reshape(len(query_z), self.cfg["pdt_m"], -1) - base_recon.reshape(len(base_recon), self.cfg["pdt_m"], -1)
        return torch.sqrt(torch.clamp(diff.pow(2).sum(dim=(1, 2)), min=1e-12))

    def _batch_loss(self, model, loss_fn, batch, epoch):
        cfg = self.cfg
        a, p, n = batch
        e_a, e_p, e_n = [model.encode(self.train_x[i].to(self.device), self.train_y[i].to(self.device),
                                      self.train_images[i].to(self.device)) for i in (a, p, n)]
        e_hat = [model.quantize(e)["e_hat"] for e in (e_a, e_p, e_n)]
        gt_ap, gt_an, gt_pn = [torch.from_numpy(np.asarray(self.train_dist[i, j])).to(self.device)
                               for i, j in ((a, p), (a, n), (p, n))]
        stats = {"e_norm": torch.norm(torch.cat((e_a, e_p, e_n)), dim=1).mean(),
                 "e_hat_norm": torch.norm(torch.cat(e_hat), dim=1).mean()}

        d_ap, d_an, d_pn = torch.norm(e_a - e_p, dim=-1), torch.norm(e_a - e_n, dim=-1), torch.norm(e_p - e_n, dim=-1)
        loss = stats["metric"] = loss_fn(epoch, d_ap, gt_ap, d_an, gt_an, d_pn, gt_pn)
        if not self.vq:
            return loss, stats

        pdt = model.PDT_model
        u_a, u_p, u_n = [model.normalize_pdt_input(e) for e in (e_a, e_p, e_n)]
        if epoch >= cfg["pdt_loss_start_epoch"]:
            stats["pdt"] = pdt(u_a, u_p, u_n, gt_pn, gt_ap, gt_an).mean()
            loss = loss + cfg["pdt_loss_weight"] * stats["pdt"]

        reg = [pdt.soft_quantize(u) for u in (u_a, u_p, u_n)]
        reg = {name: (reg[0][name] + reg[1][name] + reg[2][name]) / 3.0 for name in ("commitment", "entropy", "uniform")}

        adc_weight = _warmup_weight(epoch, cfg["adc_ste_start_epoch"], cfg["adc_ste_warmup_epochs"], cfg["adc_ste_max_weight"])
        if adc_weight > 0.0:
            (q_a, r_a), (q_p, r_p), (q_n, r_n) = [self._adc_latent(model, e) for e in (e_a, e_p, e_n)]
            stats["adc_ste"] = loss_fn(epoch, self._adc_distance(q_a, r_p), gt_ap, self._adc_distance(q_a, r_n), gt_an,
                                       self._adc_distance(q_p, r_n), gt_pn)
            loss = loss + adc_weight * stats["adc_ste"]

        if cfg["reg_start_epoch"] is not None and epoch >= cfg["reg_start_epoch"]:
            loss = loss + cfg["commitment_weight"] * reg["commitment"]
            loss = loss + cfg["entropy_weight"] * reg["entropy"]
            loss = loss + cfg["uniform_weight"] * reg["uniform"]

        teacher_weight = 0.0
        if self.teacher_codes is not None:
            teacher_weight = _decay_weight(epoch, cfg["teacher_end_epoch"], cfg["teacher_weight"])
        if teacher_weight > 0.0:
            student_z = pdt.encode(model.normalize_pdt_input(torch.cat((e_a, e_p, e_n), dim=0)))
            score = pdt.vq.quantizer.compute_score(student_z)
            target = torch.from_numpy(self.teacher_codes[np.concatenate((a, p, n))]).to(score.device)
            stats["teacher"] = F.cross_entropy(score.reshape(-1, score.shape[-1]), target.reshape(-1))
            loss = loss + teacher_weight * stats["teacher"]
        return loss, stats

    def train(self):
        cfg = self.cfg
        model = self.build_model()
        if self.backbone:
            loaded, skipped = model.load_backbone_checkpoint(self.backbone)
            print("[backbone] {}: loaded {} tensors, skipped {}".format(self.backbone, loaded, skipped))
        loss_fn = TripletLoss(cfg["epochs"], self.target_scale)
        if self.vq:
            if cfg["pdt_input_norm"]:
                self._fit_pdt_input_normalizer(model)
            self._init_codebook(model)

        teacher = self.teacher_codes is not None
        late_start = cfg["late_finetune_start_epoch"]
        codebook = model.PDT_model.vq.quantizer.codebook
        for parameter in model.parameters():
            parameter.requires_grad = True
        optimizer, state, log = None, None, []
        for epoch in range(cfg["epochs"]):
            start_time = time.time()
            frozen = teacher and epoch < cfg["teacher_codebook_freeze_epochs"]
            late = late_start is not None and epoch >= late_start
            if (frozen, late) != state:
                codebook.requires_grad = not frozen
                optimizer = self._build_optimizer(model, cfg["learning_rate"] * (cfg["late_finetune_lr_scale"] if late else 1.0))
                state = (frozen, late)
            if epoch in cfg["codebook_refresh_epochs"]:
                self._init_codebook(model)
                optimizer = self._build_optimizer(model, cfg["learning_rate"] * (cfg["late_finetune_lr_scale"] if late else 1.0))

            model.train()
            model.PDT_model.set_epoch(epoch)
            for batch in self._epoch_batches():
                loss, stats = self._batch_loss(model, loss_fn, batch, epoch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if epoch + 1 in cfg["dead_code_restart_epochs"]:
                self._restart_dead_codes(model, optimizer)

            log.append({"epoch": epoch + 1, "loss": float(loss.detach().item()), "time_sec": time.time() - start_time,
                        **{name: float(value.detach().item()) for name, value in stats.items()}})
            if (epoch + 1) % 10 == 0:
                print("epoch {:4d} | ".format(epoch + 1) + " | ".join(
                    "{} {:.4f}".format(k, v) for k, v in log[-1].items() if k != "epoch"))

        torch.save(model.state_dict(), self.output_dir / "model.pt")
        (self.output_dir / "train_log.json").write_text(json.dumps(log, indent=1))
        return self.evaluate(model)

    def load_checkpoint(self, path):
        model = self.build_model()
        result = model.load_state_dict(torch.load(path, map_location=self.device), strict=False)
        missing = set(result.missing_keys) - {"pdt_input_mean", "pdt_input_std", "pdt_input_norm_ready"}
        if missing or result.unexpected_keys:
            raise RuntimeError("checkpoint mismatch: missing {} unexpected {}".format(sorted(missing), result.unexpected_keys))
        return model

    def evaluate(self, model):
        cfg = self.cfg
        total = len(self.trajectories)
        parts = []
        for begin in range(0, total, cfg["eval_chunk_size"]):
            x, y, images = self._build_inputs(begin, min(begin + cfg["eval_chunk_size"], total))
            parts.append(self._embed(model, x, y, images, cfg["eval_batch_size"], quantize=self.vq))
        query = slice(self.n_train, self.n_train + self.n_query)
        base = slice(self.n_train + self.n_query, total)

        if not self.vq:
            embeddings = torch.cat(parts, dim=0).numpy()
            embedding_dir = self.output_dir / "embeddings"
            embedding_dir.mkdir(exist_ok=True)
            for name, rows in (("train", slice(0, self.n_train)), ("query", query), ("base", base)):
                np.save(embedding_dir / "{}.npy".format(name), embeddings[rows])
            pred_knn = np.argsort(l2_distances(embeddings[query], embeddings[base]), axis=1)[:, :100]
            metrics = {"search": "continuous_l2", **recall_metrics(self.test_dist, pred_knn)}
        else:
            z = torch.cat([part[0] for part in parts], dim=0).numpy()
            codes = torch.cat([part[1] for part in parts], dim=0).numpy()
            codebook = model.PDT_model.vq.quantizer.codebook.detach().cpu().numpy()
            pred_knn = adc_search(z[query], codes[base], codebook)
            metrics = {"search": "adc", **recall_metrics(self.test_dist, pred_knn),
                       "unique_codes_per_group": [int(np.unique(codes[:, m]).shape[0]) for m in range(codes.shape[1])],
                       "unique_code_tuples": int(np.unique(codes, axis=0).shape[0])}
        (self.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        return metrics
