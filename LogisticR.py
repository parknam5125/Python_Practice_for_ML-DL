from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1, as_frame=False)

X = mnist.data.astype('float32') / 255.0
y = mnist.target.astype('int64')

from sklearn.model_selection import train_test_split
train_img, test_img, train_lbl, test_lbl = train_test_split(
    X, y, test_size=1/7.0, random_state=0, stratify=y
)

from sklearn.linear_model import LogisticRegression
logisticRegr = LogisticRegression(
    solver='lbfgs', multi_class='multinomial', max_iter=1000
)
logisticRegr.fit(train_img, train_lbl)

logisticRegr.predict(test_img[0].reshape(1, -1))
logisticRegr.predict(test_img[0:10])

predictions = logisticRegr.predict(test_img)
score = logisticRegr.score(test_img, test_lbl)
print(score)
