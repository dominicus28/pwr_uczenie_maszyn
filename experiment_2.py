import classifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity


messages = []

with open('synthetic_spam_tmp.txt', 'r', encoding='utf-8') as f:
    content = f.readlines()
    for message in content:
        messages.append(message)
    f.close()

SyntheticClf = classifier.SMSClassifier(sampling_method="synthetic")
embeddings = SyntheticClf._tokenize_texts(messages)
similarity_matrix = cosine_similarity(embeddings)

mean_similarity = np.mean(similarity_matrix)
variance_similarity = np.var(similarity_matrix)

print(f"Średnie podobieństwo: {mean_similarity}")
print(f"Wariancja podobieństwa: {variance_similarity}")
np.save('similarity_matrix.np', similarity_matrix)
print(similarity_matrix.shape)

plt.hist(similarity_matrix.flatten(), bins=100)
plt.title('Rozkład Cosine Similarity')
plt.xlabel('Cosine Similarity')
plt.ylabel('Liczebność')
plt.savefig('similarity_histogram.png')
plt.show()