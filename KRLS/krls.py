import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

class KRLS:
    def __init__(self, x_dim, criterion):
        self.x_dim = x_dim
        self.criterion = criterion
        self.dictionary = np.zeros((0, x_dim))
        self.K_inv = None
        self.P_inv = None
        self.params = None
    
    def feature(self, x, sigma=1.0):
        if len(self.dictionary) == 0:
            return np.array([])
        diff = self.dictionary - x
        return np.exp(-np.sum(diff**2, axis=1) / (2 * sigma**2))

    def dictionary_manage(self, x, y, sigma=1.0):
        if len(self.dictionary) == 0:
            self.dictionary = np.append(self.dictionary, x.reshape(1, -1), axis=0)
            self.K_inv = np.ones((1, 1))
            self.P_inv = np.ones((1, 1))
            self.params = np.array([y])
        else:
            beta = self.feature(x, sigma)
            yhat = np.dot(self.params, beta)
            e = y - yhat
            
            a = np.dot(self.K_inv, beta)
            delta = 1 - np.dot(beta, a)
            
            if delta > self.criterion:
                self.dictionary = np.append(self.dictionary, x.reshape(1, -1), axis=0)
                term = np.outer(a, a)
                new_K_inv = (delta * self.K_inv + term) / delta
                new_row = np.append(-a, 1).reshape(1, -1)
                new_col = np.append(-a, 1).reshape(-1, 1)
                self.K_inv = np.block([[new_K_inv, new_col[:-1]], 
                                      [new_row[:, :-1], new_row[:, -1:]]])
                m = len(self.P_inv)
                self.P_inv = np.block([[self.P_inv, np.zeros((m, 1))], 
                                      [np.zeros((1, m)), 1]])
                self.params = np.append(self.params - a * e / delta, e / delta)
            else:
                Pa = np.dot(self.P_inv, a)
                q = Pa / (1 + np.dot(a, Pa))
                self.P_inv -= np.outer(q, Pa)
                self.params += e * np.dot(self.K_inv, q)
        
    def predict(self, x, sigma=1.0):
        if len(self.dictionary) == 0:
            return 0
        beta = self.feature(x, sigma)
        return np.dot(self.params, beta)

# 4.1. Точные данные
def exact_data_regression():
    np.random.seed(42)
    l = 200
    X = np.linspace(0, 10, l).reshape(-1, 1)
    y = np.sin(X).ravel()
    
    krls = KRLS(x_dim=1, criterion=0.5)
    
    for xi, yi in zip(X, y):
        krls.dictionary_manage(xi, yi, sigma=0.5)
    
    X_test = np.linspace(0, 10, 1000).reshape(-1, 1)
    y_pred = np.array([krls.predict(xi, sigma=0.5) for xi in X_test])
    y_true = np.sin(X_test).ravel()
    
    plt.figure(figsize=(12, 6))
    plt.plot(X_test, y_true, 'r:', label='True function: sin(x)')
    plt.plot(X, y, 'r.', markersize=3, alpha=0.5, label='Observations')
    plt.plot(krls.dictionary, krls.params, 'k*', markersize=10, label='Dictionary vectors')
    plt.plot(X_test, y_pred, 'b-', label='KRLS prediction')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Exact sin(x) regression (l=200) | Dictionary size: {}'.format(len(krls.dictionary)))
    plt.legend()
    plt.grid(True)
    plt.show()

# 4.2. Зашумленные данные
def noisy_data_regression():
    np.random.seed(42)
    l = 200
    X = np.linspace(0, 10, l).reshape(-1, 1)
    y = np.sin(X).ravel() + 0.1 * np.random.randn(l)
    
    krls = KRLS(x_dim=1, criterion=0.5)
    
    for xi, yi in zip(X, y):
        krls.dictionary_manage(xi, yi, sigma=0.8)
    
    X_test = np.linspace(0, 10, 1000).reshape(-1, 1)
    y_pred = np.array([krls.predict(xi, sigma=0.8) for xi in X_test])
    y_true = np.sin(X_test).ravel()
    
    plt.figure(figsize=(12, 6))
    plt.plot(X_test, y_true, 'r:', label='True function: sin(x)')
    plt.plot(X, y, 'r.', markersize=3, alpha=0.5, label='Noisy observations')
    plt.plot(krls.dictionary, krls.params, 'k*', markersize=10, label='Dictionary vectors')
    plt.plot(X_test, y_pred, 'b-', label='KRLS prediction')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Noisy sin(x) regression (l=200) | Dictionary size: {}'.format(len(krls.dictionary)))
    plt.legend()
    plt.grid(True)
    plt.show()

# 4.3. Исследование зависимости ошибки от размера словаря
def dictionary_size_study():
    np.random.seed(42)
    l = 200
    X = np.linspace(0, 10, l).reshape(-1, 1)
    y = np.sin(X).ravel() + 0.1 * np.random.randn(l)
    X_test = np.linspace(0, 10, 1000).reshape(-1, 1)
    y_true = np.sin(X_test).ravel()
    
    # Увеличим количество точек и изменим диапазон
    criteria = np.concatenate([
        np.linspace(0.001, 0.01, 10),  # Детализация для малых ε0
        np.linspace(0.02, 0.5, 30),    # Основной диапазон
        np.linspace(0.6, 1.0, 10)      # Крупные значения
    ])
    
    dict_sizes = []
    mses = []
    
    # Увеличим sigma для более плавных изменений
    sigma = 0.1
    
    for criterion in criteria:
        krls = KRLS(x_dim=1, criterion=criterion)
        for xi, yi in zip(X, y):
            krls.dictionary_manage(xi, yi, sigma=sigma)
            
        y_pred = np.array([krls.predict(xi, sigma=sigma) for xi in X_test])
        mse = mean_squared_error(y_true, y_pred)
        
        dict_sizes.append(len(krls.dictionary))
        mses.append(mse)
        print(f"ε0={criterion:.4f} | Dict size={len(krls.dictionary):3d} | MSE={mse:.6f}")
    
    # Построение графика
    plt.figure(figsize=(14, 7))
    
    # Основной график
    line, = plt.plot(dict_sizes, mses, 'b-', linewidth=2, marker='o', 
                    markersize=6, markevery=5, label='MSE')
    
    # Оптимальная точка
    opt_idx = np.argmin(mses)
    plt.scatter(dict_sizes[opt_idx], mses[opt_idx], c='red', s=200,
               label=f'Optimal (size={dict_sizes[opt_idx]}, MSE={mses[opt_idx]:.4f})')
    
    # Аннотации для ключевых точек
    step = len(criteria) // 10  # 10 равномерных отметок
    for i in range(0, len(criteria), step):
        plt.annotate(f'ε0={criteria[i]:.3f}\n({dict_sizes[i]} vec)',
                    (dict_sizes[i], mses[i]),
                    textcoords="offset points",
                    xytext=(10, 10 if i%2==0 else -20),
                    ha='center',
                    bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8),
                    fontsize=9)
    
    plt.xlabel('Dictionary size (number of support vectors)', fontsize=12)
    plt.ylabel('Mean Squared Error (MSE)', fontsize=12)
    plt.title(f'Dependence of MSE on dictionary size (σ={sigma})', fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(fontsize=12)
    
    # Автоматическое масштабирование с запасом
    plt.xlim([0, max(dict_sizes)*1.1])
    y_min, y_max = min(mses), max(mses)
    plt.ylim([y_min - 0.1*(y_max-y_min), y_max + 0.1*(y_max-y_min)])
    
    plt.tight_layout()
    plt.show()

# Запуск всех экспериментов
print("4.1. Регрессия по точным данным")
exact_data_regression()

print("\n4.2. Регрессия по зашумленным данным")
noisy_data_regression()

print("\n4.3. Исследование зависимости MSE от размера словаря")
dictionary_size_study()