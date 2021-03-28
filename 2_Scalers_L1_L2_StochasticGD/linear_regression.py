import numpy as np
class linear_regression:
    def __init__(self, eta = 0.9, max_iter = 1e4, min_weight_dist = 1e-8):
        self._eta = eta
        self.max_iter = max_iter
        self.min_weight_dist = min_weight_dist
    @staticmethod
    def mserror(X, w, c0, y_real):
        y = X.dot(w.T)+c0
        return np.sum((y - y_real)**2) / y_real.shape[0]
    @staticmethod
    def mserror_grad(X, w, c0, y_real):
        delta=(X.dot(w.T)+c0-y_real)
        return 2*delta.T.dot(X)/y_real.shape[0], np.sum(2*delta)/y_real.shape[0]
    def _calculate_eta(self, X, Y):
        gr_w, gr_c=self.mserror_grad(X, np.zeros((1, X.shape[1])), 0, Y)
        return self._eta/np.sqrt(np.linalg.norm(gr_w)**2+(gr_c)**2)
    def fit(self, X, Y):
        iter_num = 0
        weight_dist = np.inf
        w = np.zeros((1, X.shape[1]))
        c=0
        eta=self._calculate_eta(X, Y)
        while weight_dist > self.min_weight_dist and iter_num < self.max_iter:
            gr_w, gr_c=self.mserror_grad(X, w, c, Y)
            new_w = w - 2 * eta * gr_w
            new_c = c - 2 * eta * gr_c
            weight_dist = np.sqrt(np.linalg.norm(new_w - w)**2+(new_c - c)**2)
            iter_num += 1
            w = new_w
            c = new_c
        self.w=w
        self.c=c
    def predict(self, X):
        return X.dot(self.w.T)+self.c
    def test(self, X, Y):
        return self.mserror(X, self.w, self.c, Y)
