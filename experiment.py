import numpy as np
import classifier, dataset
import tabulate
from sklearn.model_selection import train_test_split, RepeatedKFold
from sklearn.metrics import confusion_matrix, f1_score
from scipy import stats

UnderClf = classifier.SMSClassifier(sampling_method="undersampling")
OverClf = classifier.SMSClassifier(sampling_method="oversampling")
SmoteClf = classifier.SMSClassifier(sampling_method="smote")
SyntheticClf = classifier.SMSClassifier(sampling_method="synthetic")
CLFs = { 'Random undersampling': UnderClf, 'Random oversampling': OverClf, 'SMOTE': SmoteClf, 'Synthetic oversampling': SyntheticClf }
# CLFs = { 'Random undersampling': UnderClf, 'Random oversampling': OverClf, 'SMOTE': SmoteClf }

X_raw, y_raw = dataset.prepare_dataset()

rkf = RepeatedKFold(n_repeats=5, n_splits=2, random_state=0)

n_splits_total = 5 * 2
n_classifiers = len(CLFs)
score = np.empty((n_splits_total, n_classifiers), float)

for i, (train_index, test_index) in enumerate(rkf.split(X_raw, y_raw)):
    X_train, X_test = X_raw[train_index], X_raw[test_index]
    y_train, y_test = y_raw[train_index], y_raw[test_index]

    for j, (name, clf) in enumerate(CLFs.items()):  # enumerate po CLFs
        clf.fit(X_train, y_train)
        prediction = clf.predict(X_test)
        score[i, j] = f1_score(y_test, prediction)
        # print("Confusion Matrix:", confusion_matrix(y_test, prediction))

np.save('results.npy', score)

print(tabulate.tabulate([['Mean score', '{:.1%}'.format(np.mean(score[:,0])), '{:.1%}'.format(np.mean(score[:,1])),
                 '{:.1%}'.format(np.mean(score[:,2])), '{:.1%}'.format(np.mean(score[:,3]))],
                ['Deviation score', '{:.1%}'.format(np.std(score[:,0])), '{:.1%}'.format(np.std(score[:,1])), '{:.1%}'.format(np.std(score[:,2])),
                 '{:.1%}'.format(np.std(score[:,3]))]],
               headers=[' ', 'Undersampling', 'Oversampling', 'SMOTE', 'Synthetic']))


results = np.load('results.npy')


t = np.zeros((len(CLFs),len(CLFs)))
p = np.zeros((len(CLFs),len(CLFs)))
significant = np.zeros((len(CLFs),len(CLFs))).astype(bool)
advantage = np.zeros((len(CLFs),len(CLFs))).astype(bool)

alpha = 0.05

for i in range(len(CLFs)):
    for j in range(len(CLFs)):
        t[i, j], p[i, j] = stats.ttest_rel(results[:, i], results[:, j])
        if np.mean(results[:, i]) > np.mean(results[:, j]):
            advantage[i, j] = True
        if p[i, j] < alpha:
            significant[i, j] = True

final = advantage * significant

# print("Statystyka:")
# print(t)
# print("P-wartości:")
# print(p)
# print("Przewaga:")
# print(advantage)
# print("Istotność statycztyczna:")
# print(significant)
# print("Końcowa macierz:\n")
# print(final)

means = np.mean(results, axis=0)

for i in range(len(means)):
    for j in range(len(means)):
        if final[i][j]: print(list(CLFs.keys())[i], "with", round(means[i],3), "better than", list(CLFs.keys())[j], "with", round(means[j],3))
