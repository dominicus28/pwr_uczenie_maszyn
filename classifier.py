import numpy as np
import torch
# import api_calls
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import shuffle
from sklearn.svm import SVC
from transformers import BertTokenizer, BertModel
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
import random

class SMSClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, sampling_method, percent_of_synthetic_samples=1, max_length=30):
        self.sampling_method = sampling_method
        self.algorithm = SVC(class_weight='balanced')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.max_length = max_length
        self.percent_of_synthetic_samples = percent_of_synthetic_samples

    def _balance_data(self, X, y):
        if self.sampling_method == "oversampling":
            sampler = RandomOverSampler()
        elif self.sampling_method == "undersampling":
            sampler = RandomUnderSampler()
        elif self.sampling_method == "smote":
            sampler = SMOTE()
            # minority_class_count = np.sum(y == 1)
            # k_neighbors = min(5, minority_class_count - 1) if minority_class_count > 1 else 1
            # sampler = SMOTE(k_neighbors=k_neighbors)
        elif self.sampling_method == "synthetic":
            return self._generate_synthetic_data(X, y, self.percent_of_synthetic_samples)
        else:
            raise ValueError("Invalid sampling method. Choose from 'oversampling', 'undersampling', 'smote', 'synthetic'.")

        X_resampled, y_resampled = sampler.fit_resample(X, y)
        return X_resampled, y_resampled

    def _generate_synthetic_data(self, X, y, percent_of_samples):
        spam_texts = X[y == 1]
        n_samples_to_generate = int((len(X[y == 0]) - len(spam_texts)) * percent_of_samples)

        # generated_spam = [api_calls.generate_response() for _ in range(n_samples_to_generate)]
        with open('synthetic_spam_tmp.txt', 'r', encoding='utf-8') as file:
            lines = file.readlines()

        generated_spam = random.sample(lines, n_samples_to_generate)

        X_synthetic = np.concatenate([X, generated_spam])
        y_synthetic = np.concatenate([y, [1] * n_samples_to_generate])

        return shuffle(X_synthetic, y_synthetic)

    def _tokenize_texts(self, texts):
        encoded = self.tokenizer(list(texts), padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")
        with torch.no_grad():
            embeddings = self.bert_model(
                input_ids=encoded['input_ids'],
                attention_mask=encoded['attention_mask']
            ).last_hidden_state.mean(dim=1)
        return embeddings.numpy()

    def fit(self, X, y):
        if self.sampling_method == 'synthetic':
            X_balanced, y_balanced = self._balance_data(X, y)
            X_tokenized = self._tokenize_texts(X_balanced)
            self.algorithm.fit(X_tokenized, y_balanced)
        else:
            X_tokenized = self._tokenize_texts(X)
            X_balanced, y_balanced = self._balance_data(X_tokenized, y)
            self.algorithm.fit(X_balanced, y_balanced)

        return self

    def predict(self, X):
        X_tokenized = self._tokenize_texts(X)
        return self.algorithm.predict(X_tokenized)

