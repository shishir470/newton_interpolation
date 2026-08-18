"""
Newton's Forward & Backward Difference Interpolation
=====================================================

Pure-Python implementation (no external dependencies) for interpolating
(or extrapolating) a value y(x) from a table of equally-spaced data
points (x0, y0), (x1, y1), ..., (xn, yn).

Two classical methods are implemented:

  * Newton's FORWARD difference formula  -> best when x is near the
    START of the table.
  * Newton's BACKWARD difference formula -> best when x is near the
    END of the table.

Both formulas require the x-values to be equally spaced (step h is
constant). The module builds the finite-difference table once and
derives both formulas from it, since they are just two different
readings (top diagonal vs. bottom diagonal) of the same table.

Run this file directly for an interactive command-line session:
    python3 newton_interpolation.py
"""

from __future__ import annotations
from typing import List, Sequence, Tuple


# --------------------------------------------------------------------------
# Core numerical routines
# --------------------------------------------------------------------------

def check_equally_spaced(x: Sequence[float], tol: float = 1e-6) -> float:
    """Return the common step h if x is equally spaced, else raise ValueError."""
    if len(x) < 2:
        raise ValueError("Need at least 2 data points.")
    h = x[1] - x[0]
    if abs(h) < tol:
        raise ValueError("Step size is ~0; x-values must be distinct.")
    for i in range(1, len(x)):
        step = x[i] - x[i - 1]
        if abs(step - h) > tol:
            raise ValueError(
                f"x-values are not equally spaced: step between x[{i-1}]={x[i-1]} "
                f"and x[{i}]={x[i]} is {step}, expected {h}."
            )
    return h


def difference_table(y: Sequence[float]) -> List[List[float]]:
    """
    Build the full finite-difference table.

    Returns a list of columns: table[0] = y itself (0th differences),
    table[1] = first differences, table[2] = second differences, etc.
    table[k] has (n + 1 - k) entries.
    """
    n = len(y)
    table: List[List[float]] = [list(y)]
    for k in range(1, n):
        prev = table[k - 1]
        col = [prev[i + 1] - prev[i] for i in range(len(prev) - 1)]
        table.append(col)
    return table


def _poly_product(p: float, k: int, sign: int) -> float:
    """p * (p + sign*1) * (p + sign*2) * ... * (p + sign*(k-1))"""
    result = p
    for i in range(1, k):
        result *= (p + sign * i)
    return result


def factorial(n: int) -> int:
    r = 1
    for i in range(2, n + 1):
        r *= i
    return r


def newton_forward(x: Sequence[float], y: Sequence[float], x_target: float
                    ) -> Tuple[float, List[float], float, List[List[float]]]:
    """
    Newton's forward difference interpolation.

    Returns: (result, term_values, p, difference_table)
      term_values[k] is the k-th term of the series (k = 0..n), so that
      result == sum(term_values).
    """
    h = check_equally_spaced(x)
    table = difference_table(y)
    n = len(y) - 1
    p = (x_target - x[0]) / h

    terms = []
    total = 0.0
    for k in range(n + 1):
        diff0k = table[k][0]  # Delta^k y0 -- top of column k
        coeff = _poly_product(p, k, sign=-1) / factorial(k) if k > 0 else 1.0
        term = coeff * diff0k
        terms.append(term)
        total += term
    return total, terms, p, table


def newton_backward(x: Sequence[float], y: Sequence[float], x_target: float
                     ) -> Tuple[float, List[float], float, List[List[float]]]:
    """
    Newton's backward difference interpolation.

    Returns: (result, term_values, p, difference_table)
    """
    h = check_equally_spaced(x)
    table = difference_table(y)
    n = len(y) - 1
    p = (x_target - x[n]) / h

    terms = []
    total = 0.0
    for k in range(n + 1):
        # nabla^k y_n equals the LAST entry of column k in the forward table
        diffnk = table[k][-1]
        coeff = _poly_product(p, k, sign=+1) / factorial(k) if k > 0 else 1.0
        term = coeff * diffnk
        terms.append(term)
        total += term
    return total, terms, p, table


def choose_method(x: Sequence[float], x_target: float) -> str:
    """Heuristic: use forward if x_target is closer to the start, else backward."""
    mid = (x[0] + x[-1]) / 2
    return "forward" if x_target <= mid else "backward"


def format_table(table: List[List[float]], x: Sequence[float]) -> str:
    """Pretty-print the difference table for terminal display."""
    n = len(table)
    headers = ["x", "y"] + [f"D^{k}y" for k in range(1, n)]
    rows = []
    for i in range(len(x)):
        row = [f"{x[i]:.6g}"]
        for k in range(n):
            if i < len(table[k]):
                row.append(f"{table[k][i]:.6g}")
            else:
                row.append("")
        rows.append(row)
    widths = [max(len(headers[c]), *(len(r[c]) for r in rows)) + 2 for c in range(len(headers))]
    lines = ["".join(h.ljust(w) for h, w in zip(headers, widths))]
    for r in rows:
        lines.append("".join(c.ljust(w) for c, w in zip(r, widths)))
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Interactive CLI
# --------------------------------------------------------------------------

def _read_float_list(prompt: str) -> List[float]:
    raw = input(prompt)
    parts = raw.replace(",", " ").split()
    return [float(p) for p in parts]


def main() -> None:
    print("=" * 60)
    print(" Newton's Forward / Backward Difference Interpolation")
    print("=" * 60)
    x = _read_float_list("Enter x values (space or comma separated): ")
    y = _read_float_list("Enter y values (space or comma separated): ")
    if len(x) != len(y):
        print(f"Error: {len(x)} x-values but {len(y)} y-values were given.")
        return
    x_target = float(input("Enter the x value to interpolate: "))

    method = input("Method [forward/backward/auto] (default auto): ").strip().lower()
    if method not in ("forward", "backward"):
        method = choose_method(x, x_target)
        print(f"-> Auto-selected: {method} difference method")

    try:
        if method == "forward":
            result, terms, p, table = newton_forward(x, y, x_target)
        else:
            result, terms, p, table = newton_backward(x, y, x_target)
    except ValueError as e:
        print(f"Error: {e}")
        return

    print("\nFinite difference table:")
    print(format_table(table, x))

    print(f"\np = {p:.6f}")
    print("Term-by-term contribution:")
    running = 0.0
    for k, t in enumerate(terms):
        running += t
        print(f"  term {k}: {t:.6f}   (running total: {running:.6f})")

    print(f"\n==> Interpolated value at x = {x_target}:  y = {result:.6f}")


if __name__ == "__main__":
    main()
