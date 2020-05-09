import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline


def make_data(N=25, err=0.8, rseed=2131):
    # randomly sample the data
    rng = np.random.RandomState(rseed)
    X = rng.rand(N, 1) ** 2
    y = 10 - 1. / (X.ravel() + 0.1)
    if err > 0:
        y += err * rng.randn(N)
    return X, y


def PolynomialRegression(degree=2, **kwargs):
    return make_pipeline(PolynomialFeatures(degree),
                         LinearRegression(**kwargs))


X, y = make_data()
xfit = np.linspace(-0.1, 1.0, 1000)[:, None]
model1 = PolynomialRegression(1).fit(X, y)
model2 = PolynomialRegression(4).fit(X, y)
model20 = PolynomialRegression(50).fit(X, y)

fig, ax = plt.subplots(1, 3, figsize=(12, 5))
fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)

X2, y2 = make_data(8, rseed=533273)

ax[0].scatter(X.ravel(), y, s=40, label='train')
ax[0].scatter(X2.ravel(), y2, s=40, color='red', label='validation')
ax[0].plot(xfit.ravel(), model1.predict(xfit), color='gray')
ax[0].legend(loc='upper left')
ax[0].axis([-0.1, 1.0, -2, 14])
# ax[0].set_title('High-bias model: Underfits the data', size=14)

ax[1].scatter(X.ravel(), y, s=40, label='train')
ax[1].scatter(X2.ravel(), y2, s=40, color='red', label='validation')
ax[1].plot(xfit.ravel(), model2.predict(xfit), color='gray')
ax[1].legend(loc='upper left')
ax[1].axis([-0.1, 1.0, -2, 14])
# ax[1].set_title('High-bias model: Underfits the data', size=14)

ax[2].scatter(X.ravel(), y, s=40, label='train')
ax[2].scatter(X2.ravel(), y2, s=40, color='red', label='validation')
ax[2].legend(loc='upper left')
ax[2].plot(xfit.ravel(), model20.predict(xfit), color='gray')
ax[2].axis([-0.1, 1.0, -2, 14])
# ax[2].set_title('High-variance model: Overfits the data', size=14)

plt.savefig('/home/samuelob/Uni/bachelor/thesis/img/overfitting_val.png')
