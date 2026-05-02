from sklearn.neighbors import LocalOutlierFactor

print("Training LOF detector...")

lof = LocalOutlierFactor(
    n_neighbors=20,
    contamination=0.05,
    novelty=True
)

lof.fit(normal_X)

lof_scores = -lof.decision_function(X_scaled)
