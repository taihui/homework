import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

X = [[20,32],[23,32],[16,32],[120,2],[30,222],[220,12],[30,222]]
X = np.asarray(X)
y = [0,0,0,0,1,1,1]
y = np.asarray(y)

svm = SVC(C=0.5, kernel='linear')
svm.fit(X, y)
plot_decision_regions(X, y, clf=svm, legend=2)
plt.show()