import numpy as np

RECALL_AT = {"HR@5": (5, 5), "HR@10": (10, 10), "HR@50": (50, 50), "HR@100": (100, 100), "R10@50": (10, 50)}


def l2_distances(q, x):
    sqr_q = np.sum(q ** 2, axis=1, keepdims=True)
    sqr_x = np.sum(x.T ** 2, axis=0, keepdims=True)
    l2 = sqr_q + sqr_x - 2 * q @ x.T
    l2[np.nonzero(l2 < 0)] = 0.0
    return np.sqrt(l2)


def adc_search(query_z, base_codes, codebook, topk=100):
    query_z = np.asarray(query_z, dtype=np.float32)
    base_codes = np.asarray(base_codes, dtype=np.int32)
    codebook = np.asarray(codebook, dtype=np.float32)
    M, _, dsub = codebook.shape
    query_groups = query_z.reshape(len(query_z), M, dsub)
    distance = np.zeros((len(query_z), len(base_codes)), dtype=np.float32)
    for m in range(M):
        q, c = query_groups[:, m, :], codebook[m]
        lookup = np.sum(q ** 2, axis=1, keepdims=True) - 2.0 * np.matmul(q, c.T) + np.sum(c ** 2, axis=1, keepdims=True).T
        distance += lookup[:, base_codes[:, m]]
    return np.argsort(distance, axis=1)[:, :topk]


def recall_metrics(true_distance, pred_knn):
    true_knn = np.argsort(true_distance, axis=1)
    metrics = {}
    for name, (k_true, k_pred) in RECALL_AT.items():
        hits = [len(np.intersect1d(t, list(p))) for t, p in zip(true_knn[:, :k_true], pred_knn[:, :k_pred])]
        metrics[name] = float(np.mean(np.array(hits) / float(k_true)))
    print(" | ".join("{} {:.4f}".format(name, value) for name, value in metrics.items()))
    return metrics
