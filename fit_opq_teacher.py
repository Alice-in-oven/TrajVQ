import argparse
import json
import math
from pathlib import Path

import faiss
import numpy as np

from trajvq.config import load_config


def fit_opq(train, m, nbits, niter=50):
    dim = train.shape[1]
    opq = faiss.OPQMatrix(dim, m, dim)
    opq.niter = niter
    opq.train(train)
    rotated = np.ascontiguousarray(opq.apply_py(train).astype(np.float32, copy=False))
    pq = faiss.ProductQuantizer(dim, m, nbits)
    pq.train(rotated)
    codebook = faiss.vector_to_array(pq.centroids).reshape(pq.M, pq.ksub, pq.dsub).astype(np.float32, copy=False)
    return rotated, codebook


def assign_codes(rotated, codebook):
    groups = rotated.reshape(rotated.shape[0], codebook.shape[0], codebook.shape[2])
    diff = groups[:, :, None, :] - codebook[None, :, :, :]
    return np.argmin(np.sum(diff * diff, axis=-1), axis=-1).astype(np.int64, copy=False)


def main():
    parser = argparse.ArgumentParser(description="Fit the OPQ teacher for a TrajVQ config.")
    parser.add_argument("--config", required=True, help="VQ-stage config; its pdt_m and pdt_k set the teacher shape.")
    parser.add_argument("--embeddings", required=True, help="embeddings/train.npy written by the continuous stage.")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    train = np.ascontiguousarray(np.load(args.embeddings).astype(np.float32, copy=False))
    rotated, codebook = fit_opq(train, cfg["pdt_m"], int(math.log2(cfg["pdt_k"])))
    codes = assign_codes(rotated, codebook)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "teacher_codebook.npy", codebook)
    np.save(output_dir / "teacher_codes.npy", codes)
    summary = {"embeddings": str(args.embeddings), "train_size": len(train), "codebook_shape": list(codebook.shape)}
    (output_dir / "teacher.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
