"""GridWorld 环境与命令行入口，仅用标准库。"""

import json
import math
import random
import sys

_ACTIONS = {
    "U": (-1, 0),
    "R": (0, 1),
    "D": (1, 0),
    "L": (0, -1),
}

_VALID_CELLS = frozenset("S.G#")


def _is_finite_number(value):
    """有限性判定，且不对大整数做 float 转换（避免 OverflowError）。"""
    return isinstance(value, int) or (
        isinstance(value, float) and math.isfinite(value)
    )


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
    """从 S 出发四向移动可到达的非墙格坐标。"""
    start = env._start
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in _ACTIONS.values():
            nxt = (r + dr, c + dc)
            if (
                nxt not in seen
                and env._in_bounds(nxt)
                and env._cell(nxt) != "#"
            ):
                seen.add(nxt)
                stack.append(nxt)
    return seen


def value_iteration(env, gamma=0.9, tolerance=1e-9, max_iterations=10000):
    """同步值迭代，返回 (values, iterations)。

    values 为 {(row, col): float}，键为从 S 四向避墙可达的格子；
    iterations 为实际迭代轮数。
    """
    if not isinstance(env, GridWorld):
        raise TypeError("env must be a GridWorld")
    if isinstance(gamma, bool) or not isinstance(gamma, (int, float)):
        raise TypeError("gamma must be an int or float")
    if not _is_finite_number(gamma) or gamma < 0 or gamma >= 1:
        raise ValueError("gamma must be finite and in [0, 1)")
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)):
        raise TypeError("tolerance must be an int or float")
    if not _is_finite_number(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and in (0, +inf)")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
        raise TypeError("max_iterations must be an int")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")

    states = sorted(_reachable_cells(env))
    values = {state: 0.0 for state in states}

    iterations = 0
    while iterations < max_iterations:
        new_values = {}
        for state in states:
            if env._cell(state) == "G":
                new_values[state] = 0.0
            else:
                best = None
                for action in _ACTIONS:
                    nxt, reward, _done = env.transition(state, action)
                    candidate = reward + gamma * values[nxt]
                    if best is None or candidate > best:
                        best = candidate
                new_values[state] = best
        iterations += 1
        delta = max(
            abs(new_values[state] - values[state]) for state in states
        )
        values = new_values
        if delta <= tolerance:
            return values, iterations
    raise RuntimeError("value iteration did not converge")


def q_learning(
    env,
    episodes=500,
    alpha=0.5,
    gamma=0.9,
    epsilon=0.1,
    seed=0,
    epsilon_end=None,
):
    """Q-learning，返回 {((row, col), action): float}。

    Q 覆盖从 S 可达的非 G 格与 U/R/D/L 的全部组合，初始为 0.0。
    每回合 reset，单回合最多 1000 步；全部随机性来自一个
    random.Random(seed)。

    epsilon_end 为 None 时各回合探索率恒为 epsilon；否则第 e 回合
    （0 起）的探索率在 epsilon 与 epsilon_end 间线性变化，
    episodes 为 1 时恒为 epsilon。
    """
    if not isinstance(env, GridWorld):
        raise TypeError("env must be a GridWorld")
    if isinstance(episodes, bool) or not isinstance(episodes, int):
        raise TypeError("episodes must be an int")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an int")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise TypeError("alpha must be an int or float")
    if isinstance(gamma, bool) or not isinstance(gamma, (int, float)):
        raise TypeError("gamma must be an int or float")
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)):
        raise TypeError("epsilon must be an int or float")
    if epsilon_end is not None and (
        isinstance(epsilon_end, bool)
        or not isinstance(epsilon_end, (int, float))
    ):
        raise TypeError("epsilon_end must be None, an int or float")
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if not _is_finite_number(alpha) or alpha <= 0 or alpha > 1:
        raise ValueError("alpha must be finite and in (0, 1]")
    if not _is_finite_number(gamma) or gamma < 0 or gamma >= 1:
        raise ValueError("gamma must be finite and in [0, 1)")
    if not _is_finite_number(epsilon) or epsilon < 0 or epsilon > 1:
        raise ValueError("epsilon must be finite and in [0, 1]")
    if epsilon_end is not None and (
        not _is_finite_number(epsilon_end)
        or epsilon_end < 0
        or epsilon_end > 1
    ):
        raise ValueError("epsilon_end must be finite and in [0, 1]")

    actions = tuple(_ACTIONS)  # U, R, D, L
    q = {
        (state, action): 0.0
        for state in sorted(_reachable_cells(env))
        if env._cell(state) != "G"
        for action in actions
    }
    rng = random.Random(seed)

    for episode in range(episodes):
        if epsilon_end is None or episodes == 1:
            rate = epsilon
        else:
            rate = epsilon + (epsilon_end - epsilon) * episode / (
                episodes - 1
            )
        state = env.reset()
        for _ in range(1000):
            if rng.random() < rate:
                action = actions[rng.randrange(4)]
            else:
                action = max(actions, key=lambda a: q[(state, a)])
            next_state, reward, done = env.step(action)
            if done:
                target = reward
            else:
                target = reward + gamma * max(
                    q[(next_state, a)] for a in actions
                )
            key = (state, action)
            q[key] += alpha * (target - q[key])
            if done:
                break
            state = next_state
    return q


def _format_value(value):
    if abs(value) < 0.5e-12:
        value = 0.0
    return format(value, ".12f")


def _run(argv):
    if len(argv) < 2:
        return 2
    command = argv[1]
    if command == "run" and len(argv) == 4:
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
    if command == "value" and len(argv) == 3:
        try:
            env = GridWorld(tuple(argv[2].split("/")))
            values, iterations = value_iteration(env)
            ordered = [
                [r, c, _format_value(values[(r, c)])]
                for (r, c) in sorted(values)
            ]
            out = {"iterations": iterations, "values": ordered}
        except (TypeError, ValueError, RuntimeError):
            return 2
        sys.stdout.write(json.dumps(out, separators=(",", ":")) + "\n")
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
