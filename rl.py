"""GridWorld 环境与命令行入口，仅用标准库。"""

import json
import math
import sys

_ACTIONS = {
    "U": (-1, 0),
    "R": (0, 1),
    "D": (1, 0),
    "L": (0, -1),
}

_VALID_CELLS = frozenset("S.G#")


class GridWorld:
    """确定性网格世界。S 起点，G 终点，# 墙，. 空地。"""

    def __init__(self, rows):
        if not isinstance(rows, tuple):
            raise TypeError("rows must be a tuple of str")
        for row in rows:
            if not isinstance(row, str):
                raise TypeError("every row must be a str")
        if not rows:
            raise ValueError("rows must be non-empty")
        width = len(rows[0])
        starts = []
        goals = 0
        for r, row in enumerate(rows):
            if len(row) != width:
                raise ValueError("rows must have equal width")
            for c, cell in enumerate(row):
                if cell not in _VALID_CELLS:
                    raise ValueError("invalid cell %r" % cell)
                if cell == "S":
                    starts.append((r, c))
                elif cell == "G":
                    goals += 1
        if len(starts) != 1:
            raise ValueError("exactly one S required")
        if goals < 1:
            raise ValueError("at least one G required")
        self.rows = rows
        self._start = starts[0]
        self.state = self._start
        self.done = False

    def reset(self):
        self.state = self._start
        self.done = False
        return self.state

    def _cell(self, state):
        return self.rows[state[0]][state[1]]

    def _in_bounds(self, state):
        r, c = state
        return 0 <= r < len(self.rows) and 0 <= c < len(self.rows[0])

    def _move(self, state, action):
        """纯查询：返回 (new_state, reward, done)。"""
        if self._cell(state) == "G":
            return state, 0, True
        dr, dc = _ACTIONS[action]
        nxt = (state[0] + dr, state[1] + dc)
        if not self._in_bounds(nxt) or self._cell(nxt) == "#":
            return state, -1, False
        if self._cell(nxt) == "G":
            return nxt, 0, True
        return nxt, -1, False

    @staticmethod
    def _check_action(action):
        if not isinstance(action, str):
            raise TypeError("action must be a str")
        if action not in _ACTIONS:
            raise ValueError("action must be one of U, R, D, L")

    def step(self, action):
        self._check_action(action)
        if self.done:
            raise RuntimeError("episode is done; call reset()")
        self.state, reward, self.done = self._move(self.state, action)
        return self.state, reward, self.done

    def transition(self, state, action):
        if (
            not isinstance(state, tuple)
            or len(state) != 2
            or any(isinstance(v, bool) or not isinstance(v, int) for v in state)
        ):
            raise TypeError("state must be a tuple of two ints")
        if not self._in_bounds(state) or self._cell(state) == "#":
            raise ValueError("state out of bounds or inside a wall")
        self._check_action(action)
        return self._move(state, action)


def _reachable_cells(env):
    """从起点 S 四向避墙可达的格子（含起点）。"""
    start = env._start
    seen = {start}
    stack = [start]
    while stack:
        state = stack.pop()
        if env._cell(state) == "G":
            continue
        for action in _ACTIONS:
            nxt, _, _ = env._move(state, action)
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return seen


def value_iteration(env, gamma=0.9, tolerance=1e-9, max_iterations=10000):
    """同步值迭代，返回 (values, iterations)。"""
    if not isinstance(env, GridWorld):
        raise TypeError("env must be a GridWorld")
    if isinstance(gamma, bool) or not isinstance(gamma, (int, float)):
        raise TypeError("gamma must be an int or float")
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)):
        raise TypeError("tolerance must be an int or float")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
        raise TypeError("max_iterations must be an int")
    gamma = float(gamma)
    tolerance = float(tolerance)
    if not math.isfinite(gamma) or not 0.0 <= gamma < 1.0:
        raise ValueError("gamma must be in [0, 1)")
    if not (math.isfinite(tolerance) and tolerance > 0.0):
        raise ValueError("tolerance must be in (0, +inf)")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be a positive int")

    cells = _reachable_cells(env)
    values = {cell: 0.0 for cell in cells}
    actions = ("U", "R", "D", "L")
    iterations = 0
    for _ in range(max_iterations):
        iterations += 1
        new_values = {}
        delta = 0.0
        for state in cells:
            if env._cell(state) == "G":
                new_values[state] = 0.0
                continue
            best = None
            for action in actions:
                nxt, reward, _ = env.transition(state, action)
                candidate = reward + gamma * values[nxt]
                if best is None or candidate > best:
                    best = candidate
            new_values[state] = best
            change = abs(best - values[state])
            if change > delta:
                delta = change
        values = new_values
        if delta <= tolerance:
            return values, iterations
    raise RuntimeError("value iteration did not converge within max_iterations")


def _format_value(value):
    if abs(value) < 0.5e-12:
        value = 0.0
    return format(value, ".12f")


def _run(argv):
    if len(argv) == 4 and argv[1] == "run":
        try:
            env = GridWorld(tuple(argv[2].split("/")))
            actions = argv[3]
            if not actions:
                raise ValueError("actions must be non-empty")
            out = []
            for action in actions:
                state, reward, done = env.step(action)
                out.append([state[0], state[1], reward, done])
        except (TypeError, ValueError, RuntimeError):
            return 2
        sys.stdout.write(json.dumps(out, separators=(",", ":")) + "\n")
        return 0
    if len(argv) == 3 and argv[1] == "value":
        try:
            env = GridWorld(tuple(argv[2].split("/")))
            values, iterations = value_iteration(env)
        except (TypeError, ValueError, RuntimeError):
            return 2
        ordered = [
            [r, c, _format_value(values[(r, c)])]
            for r, c in sorted(values)
        ]
        payload = {"iterations": iterations, "values": ordered}
        sys.stdout.write(json.dumps(payload, separators=(",", ":")) + "\n")
        return 0
    return 2


if __name__ == "__main__":
    try:
        code = _run(sys.argv)
    except Exception:
        code = 2
    if code != 0:
        sys.stderr.write("error\n")
    sys.exit(code)
