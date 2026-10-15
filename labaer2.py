import numpy as np
import pandas as pd

np.set_printoptions(precision=4, suppress=True)

'''
Данные задачи
    z = -7x1 + 5x3 + 2x4 - 9x5 -> extr
Ограничения:
    x1 + 9x2 - 7x4 + x5 = 8
    3x1 - 8x2 + 4x3 + 2x4 + 4x5 = 7
    -8x1 + 5x2 + x3 + 2x4 - 6x5 = 7
    xi >= 0, i=1..5
'''

A = np.array([
    [ 1,  9,  0, -7,  1],
    [ 3, -8,  4,  2,  4],
    [-8,  5,  1,  2, -6]
], dtype=float)

b = np.array([8, 7, 7], dtype=float)

# Коэффициенты целевой функции: z = -7x1 + 0x2 + 5x3 + 2x4 - 9x5
c = np.array([-7, 0, 5, 2, -9], dtype=float)

print("A =")
print(A)
print("\nb =", b)
print("c =", c)

# Функции М-метода
def make_m_tableau(A, b, c, M=1_000_000):
    m, n = A.shape
    # x-переменные + искусственные R-переменные + столбец b
    tableau = np.zeros((m + 1, n + m + 1), dtype=float)
    tableau[:m, :n] = A
    tableau[:m, n:n+m] = np.eye(m)
    tableau[:m, -1] = b

    # Для max: строка оценок содержит -c.
    # Искусственные переменные имеют коэффициент -M в F,
    # поэтому до приведения к базисному виду их оценки равны +M.
    tableau[-1, :n] = -c
    tableau[-1, n:n+m] = M

    basis = list(range(n, n + m))

    # Приводим строку F к базисному виду:
    # коэффициенты при базисных искусственных переменных должны стать нулевыми.
    for i in range(m):
        tableau[-1] -= M * tableau[i]

    return tableau, basis


def m_tableau_df(tableau, n_original):
    m = tableau.shape[0] - 1
    names = [f"x{i+1}" for i in range(n_original)]
    names += [f"R{i+1}" for i in range(m)]
    names += ["b"]
    index = [f"R{i+1}" for i in range(m)] + ["F"]
    return pd.DataFrame(tableau, columns=names, index=index)


def pivot(tableau, pivot_row, pivot_col):
    tableau = tableau.copy()
    pivot_value = tableau[pivot_row, pivot_col]
    tableau[pivot_row] /= pivot_value

    for i in range(tableau.shape[0]):
        if i != pivot_row:
            tableau[i] -= tableau[i, pivot_col] * tableau[pivot_row]

    return tableau


def choose_pivot(tableau, eps=1e-10):
    reduced = tableau[-1, :-1]

    if np.all(reduced >= -eps):
        return None, None

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


def m_method_max(A, b, c, M=1_000_000, verbose=True, max_iter=100, eps=1e-9):
    """Решение задачи F = c @ x -> max М-методом."""
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)

    if np.any(b < -eps):
        raise ValueError(
            "В данной учебной реализации требуется b >= 0. "
            "Если b_i < 0, соответствующее равенство сначала умножьте на -1."
        )

    tableau, basis = make_m_tableau(A, b, c, M)
    n = len(c)
    m = len(b)

    if verbose:
        print("\nНачальная таблица М-метода")
        print(m_tableau_df(tableau, n))

    for iteration in range(1, max_iter + 1):
        pivot_row, pivot_col = choose_pivot(tableau, eps)

        if pivot_row is None:
            break

        basis[pivot_row] = pivot_col
        tableau = pivot(tableau, pivot_row, pivot_col)

        if verbose:
            print(f"\nИтерация {iteration}")
            print(m_tableau_df(tableau, n))
    else:
        raise RuntimeError("Превышено максимальное число итераций.")

    # В допустимом решении искусственные переменные
    # не должны оставаться положительными.
    for row, var_index in enumerate(basis):
        if n <= var_index < n + m and tableau[row, -1] > eps:
            raise ValueError("Допустимое решение исходной задачи отсутствует.")

    x = np.zeros(n)
    for row, var_index in enumerate(basis):
        if var_index < n:
            x[var_index] = tableau[row, -1]

    f = float(c @ x)

    return {"x": x, "f": f, "basis": basis, "tableau": tableau}


def m_method_min(A, b, c, M=1_000_000, verbose=True):
    """Решение задачи F = c @ x -> min через максимизацию -F."""
    c = np.asarray(c, dtype=float)

    result = m_method_max(
        A, b, -c,
        M=M,
        verbose=verbose
    )

    # План остаётся тем же, но значение считаем
    # для исходной целевой функции F = c @ x.
    result["f"] = float(c @ result["x"])
    return result


print("ПОИСК МАКСИМУМА")

try:
    res_max = m_method_max(A, b, c, verbose=False)
    print("Оптимальный план x* =", res_max["x"])
    print("Максимум z_max =", res_max["f"])
    print("Проверка Ax = b:")
    print(A @ res_max["x"])
    print("=")
    print(b)
except Exception as e:
    print("Ошибка при поиске максимума:", e)

print("ПОИСК МИНИМУМА")

try:
    res_min = m_method_min(A, b, c, verbose=False)
    print("Оптимальный план x* =", res_min["x"])
    print("Минимум z_min =", res_min["f"])
    print("Проверка Ax = b:")
    print(A @ res_min["x"])
    print("=")
    print(b)
except Exception as e:
    print("Минимум не существует или ошибка:", e)