from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def compute_cosine_similarity(vec1, vec2):
    """
    Υπολογίζει cosine similarity ανάμεσα σε δύο vectors.
    """
    if vec1 is None or vec2 is None:
        return 0.0
    vec1 = vec1.reshape(1, -1)
    vec2 = vec2.reshape(1, -1)
    return cosine_similarity(vec1, vec2)[0][0]