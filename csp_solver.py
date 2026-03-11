#!/usr/bin/env python3
"""csp_solver.py — Constraint Satisfaction Problem solver.

Implements arc consistency (AC-3), backtracking with MRV heuristic,
forward checking, and constraint propagation. Solves Sudoku, map
coloring, N-Queens, and custom CSPs.

One file. Zero deps. Does one thing well.
"""

import sys
from copy import deepcopy
from typing import Callable


class CSP:
    """Generic CSP with variables, domains, and constraints."""

    def __init__(self):
        self.variables: list[str] = []
        self.domains: dict[str, list] = {}
        self.constraints: list[tuple[list[str], Callable]] = []
        self.neighbors: dict[str, set[str]] = {}

    def add_variable(self, name: str, domain: list):
        self.variables.append(name)
        self.domains[name] = list(domain)
        self.neighbors[name] = set()

    def add_constraint(self, variables: list[str], check: Callable):
        self.constraints.append((variables, check))
        for v in variables:
            for u in variables:
                if u != v:
                    self.neighbors[v].add(u)

    def is_consistent(self, assignment: dict) -> bool:
        for variables, check in self.constraints:
            vals = [assignment.get(v) for v in variables]
            if all(v is not None for v in vals):
                if not check(*vals):
                    return False
        return True


def ac3(csp: CSP, domains: dict[str, list]) -> bool:
    """Arc consistency - prune domains."""
    queue = [(xi, xj) for xi in csp.variables for xj in csp.neighbors.get(xi, set())]
    while queue:
        xi, xj = queue.pop(0)
        if revise(csp, domains, xi, xj):
            if not domains[xi]:
                return False
            for xk in csp.neighbors.get(xi, set()) - {xj}:
                queue.append((xk, xi))
    return True


def revise(csp: CSP, domains: dict, xi: str, xj: str) -> bool:
    revised = False
    for x in domains[xi][:]:
        # Check if any value in xj's domain satisfies constraints with x
        satisfies = False
        for y in domains[xj]:
            assignment = {xi: x, xj: y}
            ok = True
            for variables, check in csp.constraints:
                if xi in variables and xj in variables:
                    vals = [assignment.get(v) for v in variables]
                    if all(v is not None for v in vals):
                        if not check(*vals):
                            ok = False
                            break
            if ok:
                satisfies = True
                break
        if not satisfies:
            domains[xi].remove(x)
            revised = True
    return revised


def solve(csp: CSP) -> dict | None:
    """Backtracking search with MRV + forward checking."""
    domains = deepcopy(csp.domains)
    if not ac3(csp, domains):
        return None
    return _backtrack({}, csp, domains)


def _backtrack(assignment: dict, csp: CSP, domains: dict) -> dict | None:
    if len(assignment) == len(csp.variables):
        return assignment

    # MRV: pick variable with fewest remaining values
    unassigned = [v for v in csp.variables if v not in assignment]
    var = min(unassigned, key=lambda v: len(domains[v]))

    for value in domains[var]:
        assignment[var] = value
        if csp.is_consistent(assignment):
            # Forward check
            saved = {}
            ok = True
            for neighbor in csp.neighbors.get(var, set()):
                if neighbor not in assignment:
                    saved[neighbor] = list(domains[neighbor])
                    domains[neighbor] = [v for v in domains[neighbor]
                                         if _check_pair(csp, var, value, neighbor, v)]
                    if not domains[neighbor]:
                        ok = False
                        break
            if ok:
                result = _backtrack(assignment, csp, domains)
                if result is not None:
                    return result
            # Restore
            for k, v in saved.items():
                domains[k] = v
        del assignment[var]
    return None


def _check_pair(csp, v1, val1, v2, val2) -> bool:
    a = {v1: val1, v2: val2}
    for variables, check in csp.constraints:
        if v1 in variables and v2 in variables:
            vals = [a.get(v) for v in variables]
            if all(v is not None for v in vals):
                if not check(*vals):
                    return False
    return True


# ─── Preset Problems ───

def sudoku(grid: list[list[int]]) -> list[list[int]] | None:
    """Solve a 9x9 Sudoku."""
    csp = CSP()
    for r in range(9):
        for c in range(9):
            name = f'R{r}C{c}'
            domain = [grid[r][c]] if grid[r][c] != 0 else list(range(1, 10))
            csp.add_variable(name, domain)

    neq = lambda a, b: a != b
    # Row constraints
    for r in range(9):
        for i in range(9):
            for j in range(i + 1, 9):
                csp.add_constraint([f'R{r}C{i}', f'R{r}C{j}'], neq)
    # Column constraints
    for c in range(9):
        for i in range(9):
            for j in range(i + 1, 9):
                csp.add_constraint([f'R{i}C{c}', f'R{j}C{c}'], neq)
    # Box constraints
    for br in range(3):
        for bc in range(3):
            cells = [f'R{br*3+r}C{bc*3+c}' for r in range(3) for c in range(3)]
            for i in range(len(cells)):
                for j in range(i + 1, len(cells)):
                    csp.add_constraint([cells[i], cells[j]], neq)

    result = solve(csp)
    if result is None:
        return None
    return [[result[f'R{r}C{c}'] for c in range(9)] for r in range(9)]


def map_coloring(adjacency: dict[str, list[str]], colors: list[str]) -> dict[str, str] | None:
    """Color a map so no adjacent regions share a color."""
    csp = CSP()
    for region in adjacency:
        csp.add_variable(region, list(colors))
    for region, neighbors in adjacency.items():
        for neighbor in neighbors:
            csp.add_constraint([region, neighbor], lambda a, b: a != b)
    return solve(csp)


def demo():
    print("=== CSP Solver ===\n")

    # Map coloring: Australia
    print("Map Coloring (Australia):")
    adj = {
        'WA': ['NT', 'SA'], 'NT': ['WA', 'SA', 'Q'],
        'SA': ['WA', 'NT', 'Q', 'NSW', 'V'], 'Q': ['NT', 'SA', 'NSW'],
        'NSW': ['Q', 'SA', 'V'], 'V': ['SA', 'NSW'], 'T': []
    }
    result = map_coloring(adj, ['red', 'green', 'blue'])
    if result:
        for region, color in sorted(result.items()):
            print(f"  {region}: {color}")

    # Sudoku
    print("\nSudoku:")
    puzzle = [
        [5,3,0,0,7,0,0,0,0],
        [6,0,0,1,9,5,0,0,0],
        [0,9,8,0,0,0,0,6,0],
        [8,0,0,0,6,0,0,0,3],
        [4,0,0,8,0,3,0,0,1],
        [7,0,0,0,2,0,0,0,6],
        [0,6,0,0,0,0,2,8,0],
        [0,0,0,4,1,9,0,0,5],
        [0,0,0,0,8,0,0,7,9],
    ]
    solution = sudoku(puzzle)
    if solution:
        for row in solution:
            print(f"  {' '.join(str(x) for x in row)}")


if __name__ == '__main__':
    if '--test' in sys.argv:
        # Map coloring
        adj = {'A': ['B', 'C'], 'B': ['A', 'C'], 'C': ['A', 'B']}
        r = map_coloring(adj, ['r', 'g', 'b'])
        assert r and r['A'] != r['B'] and r['A'] != r['C'] and r['B'] != r['C']
        # Sudoku
        puzzle = [[5,3,0,0,7,0,0,0,0],[6,0,0,1,9,5,0,0,0],[0,9,8,0,0,0,0,6,0],
                  [8,0,0,0,6,0,0,0,3],[4,0,0,8,0,3,0,0,1],[7,0,0,0,2,0,0,0,6],
                  [0,6,0,0,0,0,2,8,0],[0,0,0,4,1,9,0,0,5],[0,0,0,0,8,0,0,7,9]]
        s = sudoku(puzzle)
        assert s and all(sum(row) == 45 for row in s)
        print("All tests passed ✓")
    else:
        demo()
