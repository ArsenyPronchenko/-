import numpy as np
import pandas as pd

np.set_printoptions(precision=4, suppress=True)

'''
F = 3x1 + 4x2 -> max
x1 + 2x2 <= 12
2x1 + x2 <= 12
x1 <= 5
x2 <= 5
x1 >= 0, x2 >= 0
'''

A = np.array([
    [1, 2],
    [2, 1],
    [1, 0],
    [0, 1]
], dtype=float)

b = np.array([12, 12, 5, 5], dtype=float)
c = np.array([3, 4], dtype=float)

print("A =")
print(A)
print("\nb =", b)
print("c =", c)


def make_tableau(A, b, c):
    m, n = A.shape
    tableau = np.zeros((m + 1, n + m + 1), dtype=float)
    tableau[:m, :n] = A
    tableau[:m, n:n + m] = np.eye(m)
    tableau[:m, -1] = b
    tableau[-1, :n] = -c
    basis = list(range(n, n + m))
    return tableau, basis


def tableau_df(tableau, n_original):
    m = tableau.shape[0] - 1
    names = [f"x{i + 1}" for i in range(n_original)]
    names += [f"s{i + 1}" for i in range(m)]
    names += ["b"]
    index = [f"R{i + 1}" for i in range(m)] + ["F"]
    return pd.DataFrame(tableau, columns=names, index=index)


def pivot(tableau, pivot_row, pivot_col):
    """Пересчёт симплекс-таблицы относительно разрешающего элемента."""
    tableau = tableau.copy()
    pivot_value = tableau[pivot_row, pivot_col]
    tableau[pivot_row] = tableau[pivot_row] / pivot_value

    for i in range(tableau.shape[0]):
        if i != pivot_row:
            tableau[i] = tableau[i] - tableau[i, pivot_col] * tableau[pivot_row]

    return tableau


def choose_pivot(tableau, eps=1e-10):
    """Выбор разрешающего столбца и строки для задачи на максимум."""
    reduced = tableau[-1, :-1]

    # Критерий оптимальности
    if np.all(reduced >= -eps):
        return None, None

    # Наиболее отрицательная оценка
    pivot_col = int(np.argmin(reduced))

    column = tableau[:-1, pivot_col]
    rhs = tableau[:-1, -1]

    positive = column > eps
    if not np.any(positive):
        raise ValueError("Целевая функция не ограничена сверху.")

    ratios = np.full_like(rhs, np.inf, dtype=float)
    ratios[positive] = rhs[positive] / column[positive]
    pivot_row = int(np.argmin(ratios))

    return pivot_row, pivot_col


def simplex_max(A, b, c, verbose=True, max_iter=100, eps=1e-10):
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)

    if np.any(b < -eps):
        raise ValueError(
            "В этой реализации требуется начальный допустимый базис: b >= 0."
        )

    tableau, basis = make_tableau(A, b, c)
    n = len(c)

    def var_name(index):
        if index < n:
            return f"x{index + 1}"
        else:
            return f"s{index - n + 1}"

    if verbose:
        print("\nНачальная таблица")
        print(tableau_df(tableau, n))

    for iteration in range(1, max_iter + 1):
        pivot_row, pivot_col = choose_pivot(tableau, eps)

        if pivot_row is None:
            break

        leaving = basis[pivot_row]
        basis[pivot_row] = pivot_col

        if verbose:
            print(
                f"\nИтерация {iteration}: "
                f"входит {var_name(pivot_col)}, "
                f"выходит {var_name(leaving)}"
            )

        tableau = pivot(tableau, pivot_row, pivot_col)

        if verbose:
            print(tableau_df(tableau, n))

    else:
        raise RuntimeError("Превышено максимально допустимое число итераций.")

    x_full = np.zeros(tableau.shape[1] - 1)

    for row, var_index in enumerate(basis):
        x_full[var_index] = tableau[row, -1]

    x = x_full[:n]
    f = float(tableau[-1, -1])

    return {
        "x": x,
        "f": f,
        "basis": basis,
        "tableau": tableau
    }


result = simplex_max(A, b, c, verbose=True)

print("Оптимальный план x* =", result["x"])
print("Максимальное значение целевой функции Fmax =", result["f"])

x = result["x"]

print("\nПроверка Ax <= b:")
print(A @ x)
print("<=")
print(b)

print("\nF(x) =", c @ x)