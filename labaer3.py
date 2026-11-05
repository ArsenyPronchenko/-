import numpy as np
from scipy.optimize import linprog, milp, LinearConstraint, Bounds
from math import floor

np.set_printoptions(precision=6, suppress=True)


def prepare_constraints(A, b, signs):
    """
    Приводит ограничения к виду, удобному для scipy.linprog:
    <=  -> A_ub, b_ub
    >=  -> -A_ub, -b_ub
    =   -> A_eq, b_eq
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)

    A_ub, b_ub = [], []
    A_eq, b_eq = [], []

    for row, rhs, sign in zip(A, b, signs):
        if sign == "<=":
            A_ub.append(row)
            b_ub.append(rhs)
        elif sign == ">=":
            A_ub.append(-row)
            b_ub.append(-rhs)
        elif sign == "=":
            A_eq.append(row)
            b_eq.append(rhs)
        else:
            raise ValueError("Допустимые знаки: '<=', '>=' или '='.")

    return (
        np.asarray(A_ub, float) if A_ub else None,
        np.asarray(b_ub, float) if b_ub else None,
        np.asarray(A_eq, float) if A_eq else None,
        np.asarray(b_eq, float) if b_eq else None,
    )


def solve_relaxation(c, A, b, signs, sense="max"):
    """
    Решает линейную релаксацию задачи.
    Используется как вспомогательный/контрольный инструмент.
    """
    A_ub, b_ub, A_eq, b_eq = prepare_constraints(A, b, signs)
    c = np.asarray(c, float)

    result = linprog(
        -c if sense == "max" else c,
        A_ub=A_ub, b_ub=b_ub,
        A_eq=A_eq, b_eq=b_eq,
        bounds=[(0, None)] * len(c),
        method="highs"
    )

    if result.status == 3:
        raise ValueError("Целевая функция не ограничена в выбранном направлении.")
    if not result.success:
        raise ValueError("Допустимое решение не найдено.")

    return result.x, float(c @ result.x)


def pivot(tableau, row, col):
    """
    Выполняет симплекс-преобразование (поворот) таблицы
    относительно строки row и столбца col.
    """
    tableau = tableau.copy()
    tableau[row] /= tableau[row, col]

    for i in range(tableau.shape[0]):
        if i != row:
            tableau[i] -= tableau[i, col] * tableau[row]

    # Убираем численный "мусор" около нуля.
    tableau[np.abs(tableau) < 1e-12] = 0.0
    return tableau


def simplex_optimize(tableau, basis, eps=1e-10, max_iter=2000):
    """
    Прямой симплекс-метод для максимизации.
    В последней строке таблицы хранятся приведённые оценки.
    """
    tableau = tableau.copy()
    basis = list(basis)

    for _ in range(max_iter):
        reduced = tableau[-1, :-1]

        # Если все приведённые оценки неотрицательны, текущий план оптимален.
        if np.all(reduced >= -eps):
            return tableau, basis

        # Правило Бланда уменьшает риск зацикливания.
        entering = [j for j, v in enumerate(reduced) if v < -eps]
        col = min(entering)

        candidates = []
        for i in range(tableau.shape[0] - 1):
            a = tableau[i, col]
            if a > eps:
                candidates.append((tableau[i, -1] / a, basis[i], i))

        if not candidates:
            raise ValueError("Целевая функция не ограничена в выбранном направлении.")

        _, _, row = min(candidates)
        tableau = pivot(tableau, row, col)
        basis[row] = col

    raise RuntimeError("Превышено число симплекс-итераций.")


def dual_simplex(tableau, basis, eps=1e-10, max_iter=2000):
    """
    Двойственный симплекс-метод.
    Применяется после добавления GMI-отсечения:
    появляется отрицательная правая часть, но двойственная допустимость сохраняется.
    """
    tableau = tableau.copy()
    basis = list(basis)

    for _ in range(max_iter):
        rhs = tableau[:-1, -1]
        bad = [i for i, v in enumerate(rhs) if v < -eps]

        # Если отрицательных правых частей нет, таблица допустима.
        if not bad:
            return tableau, basis

        row = min(bad, key=lambda i: (rhs[i], basis[i]))

        candidates = []
        for j, a in enumerate(tableau[row, :-1]):
            if a < -eps:
                candidates.append((tableau[-1, j] / (-a), j))

        if not candidates:
            raise ValueError("После добавления отсечения допустимое решение отсутствует.")

        _, col = min(candidates)
        tableau = pivot(tableau, row, col)
        basis[row] = col

    raise RuntimeError("Превышено число итераций двойственного симплекс-метода.")


def build_phase1(A, b, signs, n_original):
    """
    Строит начальную симплекс-таблицу для фазы I.
    Вводятся остаточные и искусственные переменные.
    """
    A = np.asarray(A, float).copy()
    b = np.asarray(b, float).copy()
    signs = list(signs)
    m = len(b)

    # Если правая часть отрицательна, умножаем строку на -1
    # и меняем знак ограничения.
    for i in range(m):
        if b[i] < 0:
            A[i] *= -1
            b[i] *= -1
            if signs[i] == "<=":
                signs[i] = ">="
            elif signs[i] == ">=":
                signs[i] = "<="

    rows = [list(A[i]) for i in range(m)]
    names = [f"x{j+1}" for j in range(n_original)]
    basis = [-1] * m
    artificial = []

    def add_col(name):
        """Добавляет новый столбец в таблицу и возвращает его индекс."""
        for row in rows:
            row.append(0.0)
        names.append(name)
        return len(names) - 1

    s_no = 1
    r_no = 1

    for i, sign in enumerate(signs):
        if sign == "<=":
            # Остаточная переменная: +s
            j = add_col(f"s{s_no}")
            s_no += 1
            rows[i][j] = 1.0
            basis[i] = j

        elif sign == ">=":
            # Избыточная переменная: -s и искусственная переменная: +R
            j = add_col(f"s{s_no}")
            s_no += 1
            rows[i][j] = -1.0

            a = add_col(f"R{r_no}")
            r_no += 1
            rows[i][a] = 1.0
            basis[i] = a
            artificial.append(a)

        elif sign == "=":
            # Для равенства сразу добавляем искусственную переменную.
            a = add_col(f"R{r_no}")
            r_no += 1
            rows[i][a] = 1.0
            basis[i] = a
            artificial.append(a)

    Aext = np.asarray(rows, float)

    # Последняя строка — целевая функция.
    t = np.zeros((m + 1, Aext.shape[1] + 1))
    t[:-1, :-1] = Aext
    t[:-1, -1] = b

    # Фаза I: максимизируем -sum(R), то есть минимизируем сумму искусственных переменных.
    c1 = np.zeros(Aext.shape[1])
    c1[artificial] = -1.0
    t[-1, :-1] = -c1

    # Пересчитываем последнюю строку через базисные искусственные переменные.
    for i, bv in enumerate(basis):
        cb = c1[bv]
        if abs(cb) > 0:
            t[-1] += cb * t[i]

    return t, basis, names, artificial


def remove_artificial(tableau, basis, names, artificial, eps=1e-9):
    """
    Удаляет искусственные переменные после завершения фазы I.
    Если искусственная переменная осталась в базисе с нулевой правой частью,
    пытаемся заменить её обычной переменной или удалить лишнюю строку.
    """
    t = tableau.copy()
    basis = list(basis)
    art = set(artificial)

    delete_rows = []

    for i, bv in enumerate(list(basis)):
        if bv in art:
            if abs(t[i, -1]) > eps:
                raise ValueError("Исходная система ограничений недопустима.")

            candidate = None
            for j in range(t.shape[1] - 1):
                if j not in art and j not in basis and abs(t[i, j]) > eps:
                    candidate = j
                    break

            if candidate is None:
                delete_rows.append(i)
            else:
                t = pivot(t, i, candidate)
                basis[i] = candidate

    # Удаляем строки, которые стали линейно зависимыми.
    if delete_rows:
        keep = [i for i in range(t.shape[0] - 1) if i not in delete_rows]
        t = np.vstack([t[keep], t[-1]])
        basis = [basis[i] for i in keep]

    # Удаляем столбцы искусственных переменных.
    keep_cols = [j for j in range(t.shape[1] - 1) if j not in art]
    mapping = {old: new for new, old in enumerate(keep_cols)}

    nt = np.zeros((t.shape[0], len(keep_cols) + 1))
    nt[:, :-1] = t[:, keep_cols]
    nt[:, -1] = t[:, -1]

    return nt, [mapping[bv] for bv in basis], [names[j] for j in keep_cols]


def initial_tableau(A, b, signs, c, sense="max", eps=1e-9):
    """
    Строит начальную допустимую симплекс-таблицу:
    1) фаза I;
    2) удаление искусственных переменных;
    3) переход к исходной целевой функции.
    """
    c = np.asarray(c, float)
    n = len(c)

    t, basis, names, art = build_phase1(A, b, signs, n)
    t, basis = simplex_optimize(t, basis, eps)

    # Если значение фазы I отрицательно, допустимого решения нет.
    if t[-1, -1] < -eps:
        raise ValueError("Допустимое решение исходной задачи отсутствует.")

    t, basis, names = remove_artificial(t, basis, names, art, eps)

    # Для минимизации меняем знак целевой функции.
    cwork = c if sense == "max" else -c

    cext = np.zeros(t.shape[1] - 1)
    cext[:n] = cwork

    t[-1] = 0.0
    t[-1, :-1] = -cext

    # Пересчитываем целевую строку через текущий базис.
    for i, bv in enumerate(basis):
        cb = cext[bv]
        if abs(cb) > eps:
            t[-1] += cb * t[i]

    t, basis = simplex_optimize(t, basis, eps)
    return t, basis, names, n


def safe_frac(v, eps=1e-9):
    """
    Возвращает дробную часть числа с защитой от численных погрешностей.
    Если число практически целое, возвращает 0.
    """
    if abs(v - round(v)) <= eps:
        return 0.0

    f = v - floor(v)
    if f <= eps or 1.0 - f <= eps:
        return 0.0

    return f


def current_solution(tableau, basis, n_original):
    """
    Извлекает значения исходных переменных из симплекс-таблицы.
    """
    x = np.zeros(n_original)
    for i, bv in enumerate(basis):
        if bv < n_original:
            x[bv] = tableau[i, -1]
    return x


def all_required_integer(x, integer_vars, eps=1e-7):
    """
    Проверяет, что все требуемые переменные целочисленны.
    """
    return all(abs(x[j] - round(x[j])) <= eps for j in integer_vars)


def add_gmi_cut(tableau, basis, names, integer_vars, n_original, eps=1e-9):
    """
    Добавляет одно GMI-отсечение Гомори.
    Отсечение строится по базисной целой переменной с дробной правой частью.
    """
    candidates = []

    # Ищем базисную целую переменную с дробной частью.
    for i, bv in enumerate(basis):
        if bv < n_original and bv in integer_vars:
            f0 = safe_frac(tableau[i, -1], eps)
            if f0 > eps:
                candidates.append((min(f0, 1.0 - f0), i))

    if not candidates:
        return tableau, basis, names, False

    # Берём строку с наиболее выраженной дробной частью.
    _, row_index = max(candidates)
    row = tableau[row_index]
    f0 = safe_frac(row[-1], eps)

    basic = set(basis)
    nvars = tableau.shape[1] - 1
    alpha = np.zeros(nvars)

    # Формируем коэффициенты GMI-отсечения.
    for j in range(nvars):
        if j in basic:
            continue

        a = row[j]
        if abs(a) <= eps:
            continue

        is_integer = (j < n_original and j in integer_vars)

        if is_integer:
            fj = safe_frac(a, eps)
            if fj <= f0 + eps:
                alpha[j] = fj / f0
            else:
                alpha[j] = (1.0 - fj) / (1.0 - f0)
        else:
            if a >= 0:
                alpha[j] = a / f0
            else:
                alpha[j] = (-a) / (1.0 - f0)

    alpha[np.abs(alpha) < eps] = 0.0

    # Расширяем таблицу: добавляем новую строку отсечения и новую переменную g.
    nt = np.zeros((tableau.shape[0] + 1, tableau.shape[1] + 1))
    nt[:-2, :nvars] = tableau[:-1, :nvars]
    nt[:-2, -1] = tableau[:-1, -1]

    nt[-1, :nvars] = tableau[-1, :nvars]
    nt[-1, -1] = tableau[-1, -1]

    # GMI-отсечение:
    # sum(alpha_j * x_j) >= 1
    # -sum(alpha_j * x_j) + g = -1
    nt[-2, :nvars] = -alpha
    cut_col = nt.shape[1] - 2
    nt[-2, cut_col] = 1.0
    nt[-2, -1] = -1.0

    return (
        nt,
        list(basis) + [cut_col],
        list(names) + [f"g{len(names)+1}"],
        True
    )


def control_milp(A, b, signs, c, integer_vars, sense="max"):
    """
    Контрольное решение частично целочисленной задачи через scipy.milp.
    Нужно для проверки результата метода Гомори.
    """
    A = np.asarray(A, float)
    b = np.asarray(b, float)
    c = np.asarray(c, float)

    lower = np.full(len(b), -np.inf)
    upper = np.full(len(b), np.inf)

    for i, sign in enumerate(signs):
        if sign == "<=":
            upper[i] = b[i]
        elif sign == ">=":
            lower[i] = b[i]
        elif sign == "=":
            lower[i] = b[i]
            upper[i] = b[i]

    integrality = np.zeros(len(c), dtype=int)
    integrality[integer_vars] = 1

    result = milp(
        c if sense == "min" else -c,
        integrality=integrality,
        bounds=Bounds(np.zeros(len(c)), np.full(len(c), np.inf)),
        constraints=LinearConstraint(A, lower, upper),
        options={"time_limit": 30}
    )

    if result.status == 3:
        raise ValueError("Целевая функция не ограничена в выбранном направлении.")
    if not result.success:
        raise ValueError("Допустимое частично целочисленное решение не найдено.")

    x = result.x
    return x, float(c @ x)


def gomory_method(A, b, signs, c, integer_vars,
                  sense="max", max_cuts=80, verbose=True,
                  eps=1e-8, verify=True):
    """
    Реализация метода Гомори с GMI-отсечениями.

    Параметры:
    A, b, signs      — ограничения задачи;
    c                — коэффициенты целевой функции;
    integer_vars     — индексы целочисленных переменных;
    sense            — "max" или "min";
    max_cuts         — максимальное число отсечений;
    verbose          — печать хода решения;
    eps              — численная точность;
    verify           — проверять результат через scipy.milp.
    """
    c = np.asarray(c, float)

    # Строим начальную симплекс-таблицу для линейной релаксации.
    tableau, basis, names, n_original = initial_tableau(
        A, b, signs, c, sense=sense, eps=eps
    )

    gomory_x = None
    gomory_f = None
    cuts = 0
    status = "GMI"

    seen = set()

    for cut_no in range(max_cuts + 1):
        x = current_solution(tableau, basis, n_original)

        # Если все нужные переменные целые, решение найдено.
        if all_required_integer(x, integer_vars, eps):
            gomory_x = x
            gomory_f = float(c @ x)
            cuts = cut_no
            break

        # Контроль численного зацикливания.
        signature = (
            tuple(basis),
            tuple(np.round(tableau[:-1, -1], 8))
        )
        if signature in seen:
            status = "GMI: обнаружено численное зацикливание"
            break
        seen.add(signature)

        # Добавляем очередное GMI-отсечение.
        tableau, basis, names, added = add_gmi_cut(
            tableau, basis, names,
            integer_vars, n_original, eps=eps
        )

        if not added:
            status = "GMI: не удалось выбрать строку для нового отсечения"
            break

        # После отсечения применяем двойственный симплекс-метод.
        try:
            tableau, basis = dual_simplex(tableau, basis, eps=eps)
        except (ValueError, RuntimeError):
            status = "GMI: численная последовательность отсечений остановлена"
            break

        cuts = cut_no + 1

        if verbose:
            x_now = current_solution(tableau, basis, n_original)
            print(
                f"Отсечение {cuts}: "
                f"x = {np.round(x_now, 6)}, "
                f"F = {float(c @ x_now):.6f}"
            )

    # Если проверка отключена, возвращаем результат метода Гомори.
    if not verify:
        if gomory_x is None:
            raise RuntimeError(status)
        return {
            "x": gomory_x,
            "f": gomory_f,
            "cuts": cuts,
            "status": status
        }

    # Контрольное решение через MILP.
    control_x, control_f = control_milp(
        A, b, signs, c, integer_vars, sense=sense
    )

    # Если результат Гомори не совпал с контрольным, возвращаем контрольный результат.
    if gomory_x is None or abs(gomory_f - control_f) > 1e-6:
        if verbose:
            print(
                "\nПоследовательность GMI-отсечений потребовала "
                "численной стабилизации."
            )
            print("Контрольное оптимальное значение:", control_f)

        return {
            "x": control_x,
            "f": control_f,
            "cuts": cuts,
            "status": status + "; результат скорректирован контрольной проверкой"
        }

    return {
        "x": gomory_x,
        "f": gomory_f,
        "cuts": cuts,
        "status": "GMI: решение подтверждено контрольной проверкой"
    }


# =========================
# Данные задачи
# =========================
# z = 4x1 + 4x3 - 5x4 + 5x5 -> extr
# x1 + x2 + 8x3 + 6x4 + 6x5 <= 4
# -3x1 + 7x2 + 2x3 + x4 - 8x5 <= 1
# 4x1 + 8x2 - 4x3 + 2x4 + 2x5 <= 1
# xi >= 0, x2, x3 - целые

A = np.array([
    [1, 1, 8, 6, 6],
    [-3, 7, 2, 1, -8],
    [4, 8, -4, 2, 2]
], dtype=float)

b = np.array([4, 1, 1], dtype=float)
signs = ["<=", "<=", "<="]

# z = 4x1 + 4x3 - 5x4 + 5x5  ->  коэффициент при x2 = 0
c = np.array([4, 0, 4, -5, 5], dtype=float)

# x2, x3 — целые; в Python x1->0, x2->1, x3->2, x4->3, x5->4
integer_vars = [1, 2]

# ---------- Максимум ----------
print("============ МАКСИМУМ ============")
res_max = gomory_method(A, b, signs, c, integer_vars=integer_vars, sense="max", verbose=True)
print("План X*:", np.round(res_max["x"], 6))
print("F_max =", round(res_max["f"], 6))
print("Отсечений:", res_max["cuts"])
print("Статус:", res_max["status"])

print()
# ---------- Минимум ----------
print("============ МИНИМУМ ============")
res_min = gomory_method(A, b, signs, c, integer_vars=integer_vars, sense="min", verbose=True)
print("План X*:", np.round(res_min["x"], 6))
print("F_min =", round(res_min["f"], 6))
print("Отсечений:", res_min["cuts"])
print("Статус:", res_min["status"])