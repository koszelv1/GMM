import numpy as np
class GMM:

    def __init__(self, n_components, max_iter = 100, comp_names=None):
    #inicjujemy model ustalając ilość grup na które dzielimy dane,
    #maksymalną ilość iteracji wykonanych w poszukiwaniu grupy danych
    #oraz, opcjonalnie, nazwy komponentów
        self.n_componets = n_components
        self.max_iter = max_iter
        if comp_names == None:
            self.comp_names = [f"comp: {index}" for index in range(self.n_componets)]
        else:
            self.comp_names = comp_names
        self.pi = [1/self.n_componets for comp in range(self.n_componets)]


    def multivariate_normal(self, X, mean_vector, covariance_matrix):
    #Obliczamy funkcję gęstości prawdopodobieństwa dla wielowymiarowego rozkładu normalnego
    #parametr X to wektor wiersza dla którego liczymy rozkład
    #parametr mean_vector to wektor wiersza zawierający średnie dla każdej kolumny
    #parametr covariance_matrix to macierz zawierająca konwariancje

        if np.linalg.det(covariance_matrix) == 0:
            return 1e-20    #nie chcąc dzielić przez zero, zwracamy bardzo małą wartość
        else:
            return (2 * np.pi) ** (-len(X) / 2) * np.linalg.det(covariance_matrix) ** (-1 / 2) * np.exp(
                -np.dot(np.dot((X - mean_vector).T, np.linalg.inv(covariance_matrix)), (X - mean_vector)) / 2)


    def fit(self, X):
    #zbieramy zestaw danych i używamy algorytmu Expectation-Maximization do szacowania parametrów modelu
    #parametr x odpowiada zestawowi danych który "fitujemy"

        new_X = np.array_split(X, self.n_componets)
        self.mean_vector = [np.mean(x, axis=0) for x in new_X]
        self.covariance_matrixes = [np.cov(x.T) for x in new_X]
        del new_X
        for iteration in range(self.max_iter):
            self.r = np.zeros((len(X), self.n_componets))
            for n in range(len(X)):
                for k in range(self.n_componets):
                    self.r[n][k] = self.pi[k] * self.multivariate_normal(X[n], self.mean_vector[k], self.covariance_matrixes[k])
                    self.r[n][k] /= sum([self.pi[j]*self.multivariate_normal(X[n], self.mean_vector[j], self.covariance_matrixes[j]) for j in range(self.n_componets)])
            N = np.sum(self.r, axis=0)
            self.mean_vector = np.zeros((self.n_componets, len(X[0])))
            for k in range(self.n_componets):
                for n in range(len(X)):
                    self.mean_vector[k] += self.r[n][k] * X[n]
            self.mean_vector = [1/N[k]*self.mean_vector[k] for k in range(self.n_componets)]
            self.covariance_matrixes = [np.zeros((len(X[0]), len(X[0]))) for k in range(self.n_componets)]
            for k in range(self.n_componets):
                self.covariance_matrixes[k] = np.cov(X.T, aweights=(self.r[:, k]), ddof=0)
            self.covariance_matrixes = [1/N[k]*self.covariance_matrixes[k] for k in range(self.n_componets)]
            # Updating the pi list
            self.pi = [N[k]/len(X) for k in range(self.n_componets)]


    def predict(self, X):
    #zbiera nowe dane i przypisuje je do komponentu z największą gęstością prawdopodobieństwa
    #parametr x odpowiada za zestaw danych na których próbujemy przewidzieć odpowiednią grupę
        probas = []
        for n in range(len(X)):
            probas.append([self.multivariate_normal(X[n], self.mean_vector[k], self.covariance_matrixes[k])
                           for k in range(self.n_componets)])
        cluster = []
        for proba in probas:
            cluster.append(self.comp_names[proba.index(max(proba))])
        return cluster