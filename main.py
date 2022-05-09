import numpy as np

class QAP:
    def __init__(self, assignments, distances,
                 flows, threshold=None):
        self.A = np.matrix(assignments)
        self.D = np.matrix(distances)
        self.F = np.matrix(flows)
        self.T = [[None, None, None], [None, None], [None]]
        self.L = self.costs()
        self.MAX = (self.L[0][0] + 1) if threshold is None \
            else threshold
        self.establishThreshold(self.MAX)

    def costs(self):
        L = []
        N = self.A.shape[0]
        for i in range(1, N):
            for j in range(i):
                for k in range(1, N):
                    for l in range(k):
                        L.append((self.D[i, j] * self.F[k, l],
                                  (i, j, k, l)))
        for i in range(N):
            for j in range(N):
                L.append((self.A[i, j], (i, j)))
        L.sort(reverse=True)
        return L

    def establishThreshold(self, th):
        self.T[0][0] = self.A < th
        self.T[0][2] = self.D < th
        self.T[2][0] = self.F < th
        self.T[0][1] = self.T[0][0] * self.T[2][0].transpose()
        self.T[1][0] = self.T[0][2].transpose() * self.T[0][0]
        self.T[1][1] = self.T[0][2].transpose() * self.T[0][1]
        self.sharp()

    def sharp(self):
        for i in range(4):
            self._sharp(list(range(i)) + list(range(i + 1, 4)))

    def _sharp(self, L3):
        [I, J, K] = L3
        self.T[3 - K][I] &= self.T[3 - J][I] * self.T[3 - K][J]
        self.T[3 - J][I] &= self.T[3 - K][I] * self.T[3 - K][J].transpose()
        self.T[3 - K][J] &= self.T[3 - J][I].transpose() * self.T[3 - K][I]

    @staticmethod
    def notPermutable(aMatrix):
        'Returns if aMatrix contains a permutation'
        return aMatrix.sum(axis=0).prod() \
               * aMatrix.sum(axis=1).prod() == 0

    @staticmethod
    def oneSwap(aMatrix):
        'Returns if aMatrix contains exactly one permutation'
        return aMatrix.sum(axis=0).prod() == 1 \
               and aMatrix.sum(axis=1).prod() == 1

    def noSolution(self):
        return QAP.notPermutable(self.T[0][0])

    def gotOneSolution(self):
        return QAP.oneSwap(self.T[0][0])


    def _break(self, item):
        if len(item) == 2:
            i, j = item
            self.T[0][0][i, j] = False
        elif len(item) == 4:
            i, j, k, l = item
            if self.T[0][2][i, j] > self.T[2][0][k, l]:
                self.T[0][2][i, j] = False
                self.T[0][2][j, i] = False
            else:
                self.T[2][0][k, l] = False
                self.T[2][0][l, k] = False

    def _saveTable(self):
        R = []
        for row in self.T:
            R.append([m.copy() for m in row])
        return R

    def optimize(self):
        saved = []
        for X in self.L:
            saveT = self._saveTable()
            self._break(X[1])
            self.sharp()
            if self.noSolution():
                saved.append(X)
                self.T = saveT
            elif self.gotOneSolution():
                return True
        return False

    def listing(self):
        A = {}
        X, Y = self.T[0][0].nonzero()
        for i, x in enumerate(X):
            A[x] = Y[i]
        return A

    def evaluate(self, S):
        cost = 0
        for i, site in S.items():
            cost += self.A[site, i]
        for i in range(1, len(S)):
            for j in range(i):
                cost += self.F[i, j] * self.D[S[i], S[j]]
        return cost

def test1(A, D, F):
    qap = QAP(A, D, F)
    if qap.optimize():
        result = qap.listing()
        return result, qap.evaluate(result)
    return None, None

def main():
    print(test1([[9, 51, 3], [2, 4, 1], [6, 22, 7]],[[0, 70, 2], [7, 0, 43], [2, 41, 0]],[[0, 31, 6], [3, 0, 42], [6, 4, 0]]))

if __name__ == '__main__':
    main()