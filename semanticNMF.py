# function for extraction of top N terms
def get_top_terms(coefs, features, top_N, flat=True):

    num_terms, num_topics = coefs.shape
    if flat:
        flat_coefs     = coefs.flatten()
        sorted_indices = np.argsort(-flat_coefs)
        sorted_terms          = [(features[idx // num_topics]) for idx in sorted_indices]
        sorted_terms          = sorted_terms[:top_N]
    else:
        sorted_indices = np.argsort(-coefs, axis=0)
        sorted_indices = np.array(sorted_indices[0:top_N // num_topics, :])
        sorted_terms = [
            features[term_index]
            for topic in sorted_indices.T
            for term_index in topic
            ]
    return sorted_terms

# function to calculate metrics
def calculate_metrics(true_terms, extracted_terms, all_terms):
    true_positives  = len(true_terms.intersection(extracted_terms))
    false_positives = len(extracted_terms.difference(true_terms))
    false_negatives = len(true_terms.difference(extracted_terms))
    true_negatives = len(all_terms.difference(true_terms.union(extracted_terms)))


    accuracy = (true_positives+true_negatives) / (true_positives+false_positives+false_negatives+true_negatives)
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) != 0 else 0
    recall    = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) != 0 else 0
    f1_score  = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    return np.round(accuracy, 3), np.round(precision, 3), np.round(recall, 3), np.round(f1_score, 3)

# semantic NMF algorithm
def nmf_lee_seung_gpu(V, r, M, e, l=2, max_iter=2000, eps=1e-4, nnsvd_init=True):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    m, n = V.shape

    V = torch.tensor(V, dtype=torch.float32).to(device)
    M = torch.tensor(M, dtype=torch.float32).to(device)

    if nnsvd_init:
        # Compute SVD on CPU and then move U, S, Vt to GPU if needed
        U, S, Vt = torch.linalg.svd(V.cpu(), full_matrices=False)
        U, S, Vt = U.to(device), S.to(device), Vt.to(device)

        # Initialize W and H using positive parts of U, S, Vt
        W = torch.abs(U[:, :r] @ torch.diag(torch.sqrt(S[:r])))
        H = torch.abs(torch.diag(torch.sqrt(S[:r])) @ Vt[:r, :])
    else:
        W = torch.rand(m, r, device=device)
        H = torch.rand(r, n, device=device)

    Q = torch.rand(r, e, device=device)

    for iteration in range(max_iter):
        # Update W
        W *= (V @ H.T) / (W @ H @ H.T + eps)

        # Update H
        H *= (W.T @ V + l * Q @ M.T) / (W.T @ W @ H + l * Q @ Q.T @ H + eps)

        # Update Q
        Q *= (H @ M) / (H @ H.T @ Q + eps)

        # Compute the Frobenius norm to check convergence
        errorV = torch.norm(V - W @ H, 'fro')
        errorM = torch.norm(M - H.T @ Q, 'fro')

        if errorV.item() < eps:
            break

    return W.cpu().numpy(), H.cpu().numpy(), errorV.item(), errorM.item()
