from impkmeans import ImpKMeans
import scipy.io

data = scipy.io.loadmat("outlier.mat")["outlier"]
X = data[:, :2]
y = data[:, 2]

model = ImpKMeans(k=10, r=0.1, random_state=42)
labels = model.fit_predict(X)

print("ARI =", model.score(X, y))