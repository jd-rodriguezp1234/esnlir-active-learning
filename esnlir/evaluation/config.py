from sklearn.metrics import accuracy_score, f1_score

AVERAGING = "macro"

METRICS = {
    "accuracy": accuracy_score,
    "f1_score": f1_score
}