"""GridWorld 环境与命令行入口，仅用标准库。"""

import hashlib
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
) -> dict:
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


def sarsa_lambda(
    env,
    episodes=500,
    alpha=0.5,
    gamma=0.9,
    epsilon=0.1,
    lambda_=0.9,
    seed=0,
    epsilon_end=None,
    decay_steps=None,
) -> dict:
    """SARSA(λ)，返回 {((row, col), action): float}。

    Q 覆盖从 S 可达的非 G 格与 U/R/D/L 的全部组合，初始为 0.0。
    每回合 reset，资格迹 E 清零，先选动作；单回合最多 1000 步。
    全部随机性来自一个 random.Random(seed)。

    epsilon_end 为 None 时 decay_steps 也须为 None，各回合探索率恒为
    epsilon；否则第 e 回合（0 起）e>=decay_steps 时探索率为
    epsilon_end，之前令 t=e/decay_steps，探索率为
    epsilon_end+(epsilon-epsilon_end)*(1+cos(pi*t))/2。
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
    if isinstance(lambda_, bool) or not isinstance(lambda_, (int, float)):
        raise TypeError("lambda_ must be an int or float")
    if epsilon_end is not None and (
        isinstance(epsilon_end, bool)
        or not isinstance(epsilon_end, (int, float))
    ):
        raise TypeError("epsilon_end must be None, an int or float")
    if decay_steps is not None and (
        isinstance(decay_steps, bool) or not isinstance(decay_steps, int)
    ):
        raise TypeError("decay_steps must be None or an int")
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if not _is_finite_number(alpha) or alpha <= 0 or alpha > 1:
        raise ValueError("alpha must be finite and in (0, 1]")
    if not _is_finite_number(gamma) or gamma < 0 or gamma >= 1:
        raise ValueError("gamma must be finite and in [0, 1)")
    if not _is_finite_number(epsilon) or epsilon < 0 or epsilon > 1:
        raise ValueError("epsilon must be finite and in [0, 1]")
    if not _is_finite_number(lambda_) or lambda_ < 0 or lambda_ > 1:
        raise ValueError("lambda_ must be finite and in [0, 1]")
    if epsilon_end is None:
        if decay_steps is not None:
            raise ValueError(
                "decay_steps must be None when epsilon_end is None"
            )
    else:
        if not _is_finite_number(epsilon_end) or (
            epsilon_end < 0 or epsilon_end > 1
        ):
            raise ValueError("epsilon_end must be finite and in [0, 1]")
        if decay_steps is None or decay_steps <= 0:
            raise ValueError("decay_steps must be positive")

    actions = tuple(_ACTIONS)  # U, R, D, L
    states = sorted(
        state
        for state in _reachable_cells(env)
        if env._cell(state) != "G"
    )
    q = {(state, action): 0.0 for state in states for action in actions}
    rng = random.Random(seed)

    def choose_action(state, rate):
        if rng.random() < rate:
            return actions[rng.randrange(4)]
        return max(actions, key=lambda a: q[(state, a)])

    for episode in range(episodes):
        if epsilon_end is None:
            rate = epsilon
        elif episode >= decay_steps:
            rate = epsilon_end
        else:
            t = episode / decay_steps
            rate = epsilon_end + (epsilon - epsilon_end) * (
                1 + math.cos(math.pi * t)
            ) / 2
        state = env.reset()
        eligibility = {key: 0.0 for key in q}
        action = choose_action(state, rate)
        for _ in range(1000):
            next_state, reward, done = env.step(action)
            if done:
                delta = reward - q[(state, action)]
            else:
                next_action = choose_action(next_state, rate)
                delta = (
                    reward
                    + gamma * q[(next_state, next_action)]
                    - q[(state, action)]
                )
            eligibility[(state, action)] += 1
            for key in q:
                q[key] += alpha * delta * eligibility[key]
                eligibility[key] *= gamma * lambda_
            if done:
                break
            state, action = next_state, next_action
    return q


def reinforce(env, episodes=500, alpha=0.05, gamma=0.9, seed=0) -> dict:
    """REINFORCE（带平均回报基线），返回 {((row, col), action): float}。

    H 覆盖从 S 可达的非 G 格与 U/R/D/L 的全部组合，初始为 0.0。
    按 softmax 偏好 H 采样动作；每回合 reset，单回合最多 1000 步。
    回合末以各步回报 G 的均值为基线 B，按 A=G-B 更新 H。
    全部随机性来自一个 random.Random(seed)。
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
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if not _is_finite_number(alpha) or alpha <= 0 or alpha > 1:
        raise ValueError("alpha must be finite and in (0, 1]")
    if not _is_finite_number(gamma) or gamma < 0 or gamma >= 1:
        raise ValueError("gamma must be finite and in [0, 1)")

    actions = tuple(_ACTIONS)  # U, R, D, L
    states = sorted(
        state
        for state in _reachable_cells(env)
        if env._cell(state) != "G"
    )
    h = {(state, action): 0.0 for state in states for action in actions}
    rng = random.Random(seed)

    for _ in range(episodes):
        state = env.reset()
        trajectory = []
        for _ in range(1000):
            m = max(h[(state, a)] for a in actions)
            weights = [math.exp(h[(state, a)] - m) for a in actions]
            total = sum(weights)
            probs = [weight / total for weight in weights]
            u = rng.random()
            cumulative = 0.0
            action = "L"
            for a, p_a in zip(actions, probs):
                cumulative += p_a
                if cumulative > u:
                    action = a
                    break
            prob = dict(zip(actions, probs))
            next_state, reward, done = env.step(action)
            trajectory.append((state, action, reward, prob))
            if done:
                break
            state = next_state
        g = 0.0
        returns = [0.0] * len(trajectory)
        for t in range(len(trajectory) - 1, -1, -1):
            g = trajectory[t][2] + gamma * g
            returns[t] = g
        baseline = sum(returns) / len(trajectory)
        for (state, action, _reward, prob), g_t in zip(trajectory, returns):
            advantage = g_t - baseline
            for b in actions:
                indicator = 1.0 if b == action else 0.0
                h[(state, b)] += alpha * advantage * (indicator - prob[b])
    return h


def reinforce_v(
    env,
    episodes=500,
    alpha=0.05,
    beta=0.1,
    gamma=0.9,
    seed=0,
    max_steps=1000,
) -> dict:
    """REINFORCE（带状态值 V 基线），返回 h/v 表与逐回合统计。

    H 覆盖从 S 可达的非 G 格与 U/R/D/L 的全部组合，V 覆盖同样的格子，
    初始均为 0.0。每回合 reset，至 done 或 max_steps 步；每步按
    softmax(H[s,·]) 采样动作，仅调用一次 random()，并记录
    (s, a, r, p, V[s])。回合末自 G=0.0 逆序递推 G=r+gamma*G
    （截断、不自举），再正序以 A=G 减去所记录 V[s]，先对各 b 同时
    作 H[s,b]+=alpha*A*(I[b=a]-p[b])，再 V[s]+=beta*A。
    全部随机性来自一个 random.Random(seed)。

    返回键依次为 h、v、episodes；h 项按坐标升序为
    [r, c, hU, hR, hD, hL]，v 项为 [r, c, V]，其中数均为 float；
    episodes 项为 [steps, reward, done]，reward 为未折扣回报和。
    """
    if not isinstance(env, GridWorld):
        raise TypeError("env must be a GridWorld")
    if isinstance(episodes, bool) or not isinstance(episodes, int):
        raise TypeError("episodes must be an int")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an int")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int):
        raise TypeError("max_steps must be an int")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise TypeError("alpha must be an int or float")
    if isinstance(beta, bool) or not isinstance(beta, (int, float)):
        raise TypeError("beta must be an int or float")
    if isinstance(gamma, bool) or not isinstance(gamma, (int, float)):
        raise TypeError("gamma must be an int or float")
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if not _is_finite_number(alpha) or alpha <= 0 or alpha > 1:
        raise ValueError("alpha must be finite and in (0, 1]")
    if not _is_finite_number(beta) or beta <= 0 or beta > 1:
        raise ValueError("beta must be finite and in (0, 1]")
    if not _is_finite_number(gamma) or gamma < 0 or gamma >= 1:
        raise ValueError("gamma must be finite and in [0, 1)")

    actions = tuple(_ACTIONS)  # U, R, D, L
    states = sorted(
        state
        for state in _reachable_cells(env)
        if env._cell(state) != "G"
    )
    h = {(state, action): 0.0 for state in states for action in actions}
    v = {state: 0.0 for state in states}
    rng = random.Random(seed)
    episode_results = []

    for _ in range(episodes):
        state = env.reset()
        trajectory = []
        total_reward = 0
        done = False
        for _ in range(max_steps):
            m = max(h[(state, a)] for a in actions)
            weights = [math.exp(h[(state, a)] - m) for a in actions]
            total = sum(weights)
            probs = [weight / total for weight in weights]
            u = rng.random()
            cumulative = 0.0
            action = "L"
            for a, p_a in zip(actions, probs):
                cumulative += p_a
                if cumulative > u:
                    action = a
                    break
            prob = dict(zip(actions, probs))
            value = v[state]
            next_state, reward, done = env.step(action)
            trajectory.append((state, action, reward, prob, value))
            total_reward += reward
            if done:
                break
            state = next_state
        episode_results.append([len(trajectory), total_reward, done])

        g = 0.0
        returns = [0.0] * len(trajectory)
        for t in range(len(trajectory) - 1, -1, -1):
            g = trajectory[t][2] + gamma * g
            returns[t] = g
        for t, (st, action, _reward, prob, value) in enumerate(trajectory):
            advantage = returns[t] - value
            for b in actions:
                indicator = 1.0 if b == action else 0.0
                h[(st, b)] += alpha * advantage * (indicator - prob[b])
            v[st] += beta * advantage

    h_table = [
        [
            float(r),
            float(c),
            h[((r, c), "U")],
            h[((r, c), "R")],
            h[((r, c), "D")],
            h[((r, c), "L")],
        ]
        for (r, c) in states
    ]
    v_table = [
        [float(r), float(c), v[(r, c)]] for (r, c) in states
    ]
    return {"h": h_table, "v": v_table, "episodes": episode_results}


def actor_critic(
    env,
    episodes=500,
    alpha=0.05,
    beta=0.1,
    gamma=0.9,
    seed=0,
    max_steps=1000,
) -> dict:
    """在线一步 actor-critic，返回 h/v 表与逐回合统计。

    H 覆盖从 S 可达的非 G 格与 U/R/D/L 的全部组合，V 覆盖同样的格子，
    初始均为 0.0。每回合 reset，至 done 或 max_steps 步；每步按
    softmax(H[s,·]) 采样动作，仅调用一次 random()。step 前保留动作
    概率 p 与旧 V[s]，step 得 (s2, r, done)：done 时 δ=r-V[s]，
    否则 δ=r+gamma*V[s2]-V[s]；达到步限且未 done 的末步同样按后式
    自举。先对各 b 同时作 H[s,b]+=alpha*δ*(I[b=a]-p[b])，再作
    V[s]+=beta*δ。全部随机性来自一个 random.Random(seed)。

    返回键依次为 h、v、episodes；h 项按坐标升序为
    [r, c, hU, hR, hD, hL]，v 项为 [r, c, V]，其中数均为 float；
    episodes 项为 [steps, reward, done]，类型依次为 int/int/bool，
    reward 为未折扣回报和。
    """
    if not isinstance(env, GridWorld):
        raise TypeError("env must be a GridWorld")
    if isinstance(episodes, bool) or not isinstance(episodes, int):
        raise TypeError("episodes must be an int")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an int")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int):
        raise TypeError("max_steps must be an int")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise TypeError("alpha must be an int or float")
    if isinstance(beta, bool) or not isinstance(beta, (int, float)):
        raise TypeError("beta must be an int or float")
    if isinstance(gamma, bool) or not isinstance(gamma, (int, float)):
        raise TypeError("gamma must be an int or float")
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if not _is_finite_number(alpha) or alpha <= 0 or alpha > 1:
        raise ValueError("alpha must be finite and in (0, 1]")
    if not _is_finite_number(beta) or beta <= 0 or beta > 1:
        raise ValueError("beta must be finite and in (0, 1]")
    if not _is_finite_number(gamma) or gamma < 0 or gamma >= 1:
        raise ValueError("gamma must be finite and in [0, 1)")

    actions = tuple(_ACTIONS)  # U, R, D, L
    states = sorted(
        state
        for state in _reachable_cells(env)
        if env._cell(state) != "G"
    )
    h = {(state, action): 0.0 for state in states for action in actions}
    v = {state: 0.0 for state in states}
    rng = random.Random(seed)
    episode_results = []

    for _ in range(episodes):
        state = env.reset()
        steps = 0
        total_reward = 0
        done = False
        for _ in range(max_steps):
            m = max(h[(state, a)] for a in actions)
            weights = [math.exp(h[(state, a)] - m) for a in actions]
            total = sum(weights)
            probs = [weight / total for weight in weights]
            u = rng.random()
            cumulative = 0.0
            action = "L"
            for a, p_a in zip(actions, probs):
                cumulative += p_a
                if cumulative > u:
                    action = a
                    break
            prob = dict(zip(actions, probs))
            value = v[state]
            next_state, reward, done = env.step(action)
            steps += 1
            total_reward += reward
            if done:
                delta = reward - value
            else:
                delta = reward + gamma * v[next_state] - value
            for b in actions:
                indicator = 1.0 if b == action else 0.0
                h[(state, b)] += alpha * delta * (indicator - prob[b])
            v[state] += beta * delta
            if done:
                break
            state = next_state
        episode_results.append([steps, total_reward, done])

    h_table = [
        [
            float(r),
            float(c),
            h[((r, c), "U")],
            h[((r, c), "R")],
            h[((r, c), "D")],
            h[((r, c), "L")],
        ]
        for (r, c) in states
    ]
    v_table = [
        [float(r), float(c), v[(r, c)]] for (r, c) in states
    ]
    return {"h": h_table, "v": v_table, "episodes": episode_results}


def nstep_actor_critic(
    env,
    episodes=500,
    alpha=0.05,
    beta=0.1,
    gamma=0.9,
    n_steps=5,
    seed=0,
    max_steps=1000,
) -> dict:
    """在线 n 步 actor-critic，返回 h/v 表与逐回合统计。

    H 覆盖从 S 可达的非 G 格与 U/R/D/L 的全部组合，V 覆盖同样的格子，
    初始均为 0.0。每回合 reset，至 done 或 max_steps 步；每步按
    softmax(H[s,·]) 采样动作，仅调用一次 random()。step 前保留
    (s, a, p, v)，v 为旧 V[s]；step 得 (s2, r, done) 后再存
    (r, done, v2) 入 FIFO，done 时 v2=0.0，否则 v2 为当时 V[s2]。
    FIFO 满 n_steps 即取队头起 m=n_steps 项更新并弹头：
    G=Σ(k=0..m-1) gamma^k*r[k]+gamma^m*B，末项 done 则 B=0.0，
    否则 B 为其 v2；A=G-头项 v，按头项旧 p 先对各动作 b 同时作
    H[s,b]+=alpha*A*(I[b=a]-p[b])，再作 V[s]+=beta*A。回合因 done
    或 max_steps 结束后按头到尾冲刷 FIFO，每次 m 取当前长度；终止尾
    不自举，截断尾用所存 v2 自举。全部随机性来自一个
    random.Random(seed)。n_steps=1 时与 actor_critic 逐值等同。

    返回键依次为 h、v、episodes；h 项按坐标升序为
    [r, c, hU, hR, hD, hL]，v 项为 [r, c, V]，其中数均为 float；
    episodes 项为 [steps, reward, done]，类型依次为 int/int/bool，
    reward 为未折扣回报和。
    """
    if not isinstance(env, GridWorld):
        raise TypeError("env must be a GridWorld")
    if isinstance(episodes, bool) or not isinstance(episodes, int):
        raise TypeError("episodes must be an int")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an int")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int):
        raise TypeError("max_steps must be an int")
    if isinstance(n_steps, bool) or not isinstance(n_steps, int):
        raise TypeError("n_steps must be an int")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise TypeError("alpha must be an int or float")
    if isinstance(beta, bool) or not isinstance(beta, (int, float)):
        raise TypeError("beta must be an int or float")
    if isinstance(gamma, bool) or not isinstance(gamma, (int, float)):
        raise TypeError("gamma must be an int or float")
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if not _is_finite_number(alpha) or alpha <= 0 or alpha > 1:
        raise ValueError("alpha must be finite and in (0, 1]")
    if not _is_finite_number(beta) or beta <= 0 or beta > 1:
        raise ValueError("beta must be finite and in (0, 1]")
    if not _is_finite_number(gamma) or gamma < 0 or gamma >= 1:
        raise ValueError("gamma must be finite and in [0, 1)")

    actions = tuple(_ACTIONS)  # U, R, D, L
    states = sorted(
        state
        for state in _reachable_cells(env)
        if env._cell(state) != "G"
    )
    h = {(state, action): 0.0 for state in states for action in actions}
    v = {state: 0.0 for state in states}
    rng = random.Random(seed)
    episode_results = []

    def update(queue, m):
        """以队头起 m 项计算 n 步回报，仅更新队头状态。"""
        _s, _a, prob, value, _r, tail_done, tail_v2 = queue[m - 1]
        bootstrap = 0.0 if tail_done else tail_v2
        g = 0.0
        discount = 1.0
        for k in range(m):
            g += discount * queue[k][4]
            discount *= gamma
        g += discount * bootstrap
        head_state, head_action, head_prob, head_value = queue[0][:4]
        advantage = g - head_value
        for b in actions:
            indicator = 1.0 if b == head_action else 0.0
            h[(head_state, b)] += (
                alpha * advantage * (indicator - head_prob[b])
            )
        v[head_state] += beta * advantage

    for _ in range(episodes):
        state = env.reset()
        queue = []
        steps = 0
        total_reward = 0
        done = False
        for _ in range(max_steps):
            m = max(h[(state, a)] for a in actions)
            weights = [math.exp(h[(state, a)] - m) for a in actions]
            total = sum(weights)
            probs = [weight / total for weight in weights]
            u = rng.random()
            cumulative = 0.0
            action = "L"
            for a, p_a in zip(actions, probs):
                cumulative += p_a
                if cumulative > u:
                    action = a
                    break
            prob = dict(zip(actions, probs))
            value = v[state]
            next_state, reward, done = env.step(action)
            value_next = 0.0 if done else v[next_state]
            queue.append(
                (state, action, prob, value, reward, done, value_next)
            )
            steps += 1
            total_reward += reward
            if len(queue) == n_steps:
                update(queue, n_steps)
                queue.pop(0)
            if done:
                break
            state = next_state
        while queue:
            update(queue, len(queue))
            queue.pop(0)
        episode_results.append([steps, total_reward, done])

    h_table = [
        [
            float(r),
            float(c),
            h[((r, c), "U")],
            h[((r, c), "R")],
            h[((r, c), "D")],
            h[((r, c), "L")],
        ]
        for (r, c) in states
    ]
    v_table = [
        [float(r), float(c), v[(r, c)]] for (r, c) in states
    ]
    return {"h": h_table, "v": v_table, "episodes": episode_results}


def gae_actor_critic(
    env,
    episodes=500,
    alpha=0.05,
    beta=0.1,
    gamma=0.9,
    lambda_=0.95,
    seed=0,
    max_steps=1000,
) -> dict:
    """GAE actor-critic（整回合采样后更新），返回 h/v 表与逐回合统计。

    H 覆盖从 S 可达的非 G 格与 U/R/D/L 的全部组合，V 覆盖同样的格子，
    初始均为 0.0。每回合 reset，至 done 或 max_steps 步；每步按
    softmax(H[s,·]) 采样动作，仅调用一次 random()，并记录
    (s, a, r, p, v, v2)：v 为采样时 V[s]，done 时 v2=0，否则为当时
    V[s2]（步限截断仍自举）。回合末自 A=0 逆序递推
    delta=r+gamma*v2-v、A=delta+gamma*lambda_*A 并保存；再正序以所存
    p、A 先对各 b 同时作 H[s,b]+=alpha*A*(I[b=a]-p[b])，再作
    V[s]+=beta*A。全部随机性来自一个 random.Random(seed)。

    返回键依次为 h、v、episodes；h 项按坐标升序为
    [r, c, hU, hR, hD, hL]，v 项为 [r, c, V]，其中数均为 float；
    episodes 项为 [steps, reward, done]，类型依次为 int/int/bool，
    reward 为未折扣回报和。
    """
    if not isinstance(env, GridWorld):
        raise TypeError("env must be a GridWorld")
    if isinstance(episodes, bool) or not isinstance(episodes, int):
        raise TypeError("episodes must be an int")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an int")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int):
        raise TypeError("max_steps must be an int")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise TypeError("alpha must be an int or float")
    if isinstance(beta, bool) or not isinstance(beta, (int, float)):
        raise TypeError("beta must be an int or float")
    if isinstance(gamma, bool) or not isinstance(gamma, (int, float)):
        raise TypeError("gamma must be an int or float")
    if isinstance(lambda_, bool) or not isinstance(lambda_, (int, float)):
        raise TypeError("lambda_ must be an int or float")
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if not _is_finite_number(alpha) or alpha <= 0 or alpha > 1:
        raise ValueError("alpha must be finite and in (0, 1]")
    if not _is_finite_number(beta) or beta <= 0 or beta > 1:
        raise ValueError("beta must be finite and in (0, 1]")
    if not _is_finite_number(gamma) or gamma < 0 or gamma > 1:
        raise ValueError("gamma must be finite and in [0, 1]")
    if not _is_finite_number(lambda_) or lambda_ < 0 or lambda_ > 1:
        raise ValueError("lambda_ must be finite and in [0, 1]")

    actions = tuple(_ACTIONS)  # U, R, D, L
    states = sorted(
        state
        for state in _reachable_cells(env)
        if env._cell(state) != "G"
    )
    h = {(state, action): 0.0 for state in states for action in actions}
    v = {state: 0.0 for state in states}
    rng = random.Random(seed)
    episode_results = []

    for _ in range(episodes):
        state = env.reset()
        trajectory = []
        total_reward = 0
        done = False
        for _ in range(max_steps):
            m = max(h[(state, a)] for a in actions)
            weights = [math.exp(h[(state, a)] - m) for a in actions]
            total = sum(weights)
            probs = [weight / total for weight in weights]
            u = rng.random()
            cumulative = 0.0
            action = "L"
            for a, p_a in zip(actions, probs):
                cumulative += p_a
                if cumulative > u:
                    action = a
                    break
            prob = dict(zip(actions, probs))
            value = v[state]
            next_state, reward, done = env.step(action)
            value_next = 0.0 if done else v[next_state]
            trajectory.append(
                (state, action, reward, prob, value, value_next)
            )
            total_reward += reward
            if done:
                break
            state = next_state
        episode_results.append([len(trajectory), total_reward, done])

        advantages = [0.0] * len(trajectory)
        a_t = 0.0
        for t in range(len(trajectory) - 1, -1, -1):
            _st, _act, reward, _prob, value, value_next = trajectory[t]
            delta = reward + gamma * value_next - value
            a_t = delta + gamma * lambda_ * a_t
            advantages[t] = a_t
        for (st, action, _reward, prob, _value, _v2), adv in zip(
            trajectory, advantages
        ):
            for b in actions:
                indicator = 1.0 if b == action else 0.0
                h[(st, b)] += alpha * adv * (indicator - prob[b])
            v[st] += beta * adv

    h_table = [
        [
            float(r),
            float(c),
            h[((r, c), "U")],
            h[((r, c), "R")],
            h[((r, c), "D")],
            h[((r, c), "L")],
        ]
        for (r, c) in states
    ]
    v_table = [
        [float(r), float(c), v[(r, c)]] for (r, c) in states
    ]
    return {"h": h_table, "v": v_table, "episodes": episode_results}


def generalized_advantage_estimate(
    rewards, values, gamma=0.9, lambda_=0.95, bootstrap=0.0
) -> dict:
    """广义优势估计（GAE），返回固定键序 advantages、returns 的 dict。

    rewards、values 须为等长非空的 list/tuple，各元素为非 bool 的
    int/float；gamma、lambda_、bootstrap 为非 bool 的 int/float 标量，
    其中 gamma、lambda_ 须在 [0, 1]。所有输入先复制并转换为 float
    （转换溢出或结果非有限均抛 ValueError），不修改原序列。逆序递推
    delta=r[t]+gamma*(bootstrap if t==T-1 else v[t+1])-v[t]、
    A[t]=delta+gamma*lambda_*A[t+1]（末步之外以 v[t+1] 自举），
    returns[t]=A[t]+v[t]；两个列表均按原时序排列。
    """
    if not isinstance(rewards, (list, tuple)):
        raise TypeError("rewards must be a list or tuple")
    if not isinstance(values, (list, tuple)):
        raise TypeError("values must be a list or tuple")
    if len(rewards) == 0 or len(values) == 0:
        raise ValueError("rewards and values must be non-empty")
    if len(rewards) != len(values):
        raise ValueError("rewards and values must have equal length")

    def _to_float(item, name):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TypeError(f"{name} must contain only int or float")
        try:
            result = float(item)
        except OverflowError:
            raise ValueError(f"{name} must convert to a finite float")
        if not math.isfinite(result):
            raise ValueError(f"{name} must contain only finite numbers")
        return result

    r = [_to_float(item, "rewards") for item in rewards]
    v = [_to_float(item, "values") for item in values]

    def _to_scalar(item, name):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TypeError(f"{name} must be an int or float")
        try:
            result = float(item)
        except OverflowError:
            raise ValueError(f"{name} must convert to a finite float")
        if not math.isfinite(result):
            raise ValueError(f"{name} must be finite")
        return result

    g = _to_scalar(gamma, "gamma")
    l = _to_scalar(lambda_, "lambda_")
    b = _to_scalar(bootstrap, "bootstrap")
    if g < 0.0 or g > 1.0:
        raise ValueError("gamma must be in [0, 1]")
    if l < 0.0 or l > 1.0:
        raise ValueError("lambda_ must be in [0, 1]")

    T = len(r)
    n = 0.0
    A = [0.0] * T
    for t in range(T - 1, -1, -1):
        next_value = b if t == T - 1 else v[t + 1]
        d = r[t] + g * next_value - v[t]
        A[t] = d + g * l * n
        n = A[t]
    returns = [A[t] + v[t] for t in range(T)]
    return {"advantages": A, "returns": returns}


def ppo_clipped_surrogate(
    old_log_probs, new_log_probs, advantages, clip_epsilon=0.2
) -> dict:
    """PPO 截断代理目标，返回固定键序 ratios、objectives、mean_objective 的 dict。

    old_log_probs、new_log_probs、advantages 须为等长非空的 list/tuple，
    各元素为非 bool 的 int/float；clip_epsilon 为非 bool 的 int/float
    标量且须在 [0, 1)。所有输入先复制并转换为 float（转换溢出或结果
    非有限均抛 ValueError），不修改原序列。逐项计算
    d=n[i]-o[i]、r=exp(d)（上溢抛 ValueError，下溢为 0.0 合法）、
    c=min(max(r,1-e),1+e)、u=r*a[i]、v=c*a[i]、z=min(u,v)；d 或 u、v
    非有限均抛 ValueError。ratios、objectives 为 r、z 的 float 列表，
    mean_objective 为 sum(objectives, 0.0)/len(objectives)；求和或均值
    非有限抛 ValueError。
    """
    if not isinstance(old_log_probs, (list, tuple)):
        raise TypeError("old_log_probs must be a list or tuple")
    if not isinstance(new_log_probs, (list, tuple)):
        raise TypeError("new_log_probs must be a list or tuple")
    if not isinstance(advantages, (list, tuple)):
        raise TypeError("advantages must be a list or tuple")
    if (
        len(old_log_probs) == 0
        or len(new_log_probs) == 0
        or len(advantages) == 0
    ):
        raise ValueError(
            "old_log_probs, new_log_probs and advantages must be non-empty"
        )
    if not (
        len(old_log_probs) == len(new_log_probs) == len(advantages)
    ):
        raise ValueError(
            "old_log_probs, new_log_probs and advantages must have equal length"
        )

    def _to_float(item, name):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TypeError(f"{name} must contain only int or float")
        try:
            result = float(item)
        except OverflowError:
            raise ValueError(f"{name} must convert to a finite float")
        if not math.isfinite(result):
            raise ValueError(f"{name} must contain only finite numbers")
        return result

    o = [_to_float(item, "old_log_probs") for item in old_log_probs]
    n = [_to_float(item, "new_log_probs") for item in new_log_probs]
    a = [_to_float(item, "advantages") for item in advantages]

    if isinstance(clip_epsilon, bool) or not isinstance(
        clip_epsilon, (int, float)
    ):
        raise TypeError("clip_epsilon must be an int or float")
    try:
        e = float(clip_epsilon)
    except OverflowError:
        raise ValueError("clip_epsilon must convert to a finite float")
    if not math.isfinite(e):
        raise ValueError("clip_epsilon must be finite")
    if e < 0.0 or e >= 1.0:
        raise ValueError("clip_epsilon must be in [0, 1)")

    ratios = []
    objectives = []
    for i in range(len(o)):
        d = n[i] - o[i]
        if not math.isfinite(d):
            raise ValueError("log-prob difference must be finite")
        try:
            r = math.exp(d)
        except OverflowError:
            raise ValueError("probability ratio exp overflow")
        c = min(max(r, 1.0 - e), 1.0 + e)
        u = r * a[i]
        v = c * a[i]
        if not math.isfinite(u) or not math.isfinite(v):
            raise ValueError("clipped objective terms must be finite")
        ratios.append(r)
        objectives.append(min(u, v))
    total = sum(objectives, 0.0)
    if not math.isfinite(total):
        raise ValueError("objective sum must be finite")
    mean_objective = total / len(objectives)
    if not math.isfinite(mean_objective):
        raise ValueError("mean_objective must be finite")
    return {
        "ratios": ratios,
        "objectives": objectives,
        "mean_objective": mean_objective,
    }


def ppo_update(logits, batch, lr=0.05, c=0.2, epochs=4) -> dict:
    """PPO 多轮 logits 更新，返回固定键序 logits、objectives、probabilities 的 dict。

    logits 须为非空矩形 list，每行为非空 list 且元素为有限 float；
    batch 须为非空 list，每项为四项 list [s, a, o, A]，其中 s、a 为
    非 bool 的 int 且分别为有效行、列索引，o、A 为有限 float；lr、c
    为有限 float 且依次属于 (0, 1]、[0, 1)；epochs 为非 bool 正 int。
    类型不符抛 TypeError，其余约束不符抛 ValueError。每轮冻结 L，
    各行取 m=max(L)，算 q_j=L_j-m-log(Σexp(L_k-m))、p_j=exp(q_j)；
    按 batch 序取 q_a，以 ppo_clipped_surrogate 求 r 及更新前均值。
    若 A>0 且 r>1+c 或 A<0 且 r<1-c，该样本梯度为 0；否则行 s 列 j
    梯度为 r*A*(I[j==a]-p_j)。依样本、列序累加，除 batch 长度后同步
    L+=lr*g。新运算结果非有限均抛 ValueError。返回最终 L、各轮均值
    list、最终 softmax 二维 list，数值均为 float；不修改输入，同输入
    逐值一致。
    """
    if not isinstance(logits, list):
        raise TypeError("logits must be a list")
    if not logits:
        raise ValueError("logits must be non-empty")
    width = None
    for row in logits:
        if not isinstance(row, list):
            raise TypeError("every row of logits must be a list")
        if not row:
            raise ValueError("every row of logits must be non-empty")
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise ValueError("logits must be rectangular")
        for item in row:
            if not isinstance(item, float):
                raise TypeError("logits must contain only float")
            if not math.isfinite(item):
                raise ValueError("logits must contain only finite float")
    n_rows = len(logits)
    n_cols = width

    if not isinstance(batch, list):
        raise TypeError("batch must be a list")
    if not batch:
        raise ValueError("batch must be non-empty")
    samples = []
    for item in batch:
        if not isinstance(item, list):
            raise TypeError("every batch item must be a list")
        if len(item) != 4:
            raise ValueError("every batch item must have exactly four elements")
        s, a, o, adv = item
        if isinstance(s, bool) or not isinstance(s, int):
            raise TypeError("state index must be a non-bool int")
        if isinstance(a, bool) or not isinstance(a, int):
            raise TypeError("action index must be a non-bool int")
        if not 0 <= s < n_rows:
            raise ValueError("state index out of range")
        if not 0 <= a < n_cols:
            raise ValueError("action index out of range")
        if not isinstance(o, float):
            raise TypeError("old log-prob must be a float")
        if not math.isfinite(o):
            raise ValueError("old log-prob must be finite")
        if not isinstance(adv, float):
            raise TypeError("advantage must be a float")
        if not math.isfinite(adv):
            raise ValueError("advantage must be finite")
        samples.append((s, a, o, adv))

    if not isinstance(lr, float):
        raise TypeError("lr must be a float")
    if not math.isfinite(lr):
        raise ValueError("lr must be finite")
    if lr <= 0.0 or lr > 1.0:
        raise ValueError("lr must be in (0, 1]")
    if not isinstance(c, float):
        raise TypeError("c must be a float")
    if not math.isfinite(c):
        raise ValueError("c must be finite")
    if c < 0.0 or c >= 1.0:
        raise ValueError("c must be in [0, 1)")
    if isinstance(epochs, bool) or not isinstance(epochs, int):
        raise TypeError("epochs must be a non-bool int")
    if epochs <= 0:
        raise ValueError("epochs must be positive")

    def _log_softmax(row):
        m = max(row)
        total = 0.0
        for value in row:
            total += math.exp(value - m)
        if not math.isfinite(total):
            raise ValueError("softmax normalizer must be finite")
        log_total = math.log(total)
        q = []
        p = []
        for value in row:
            q_j = value - m - log_total
            if not math.isfinite(q_j):
                raise ValueError("log-probabilities must be finite")
            p_j = math.exp(q_j)
            if not math.isfinite(p_j):
                raise ValueError("probabilities must be finite")
            q.append(q_j)
            p.append(p_j)
        return q, p

    L = [list(row) for row in logits]
    n_batch = len(samples)
    objectives = []
    for _ in range(epochs):
        frozen = [list(row) for row in L]
        qs = []
        ps = []
        for row in frozen:
            q, p = _log_softmax(row)
            qs.append(q)
            ps.append(p)
        g = [[0.0] * n_cols for _ in range(n_rows)]
        epoch_total = 0.0
        for s, a, o, adv in samples:
            result = ppo_clipped_surrogate([o], [qs[s][a]], [adv], c)
            r = result["ratios"][0]
            epoch_total += result["mean_objective"]
            if not math.isfinite(epoch_total):
                raise ValueError("objective sum must be finite")
            if (adv > 0.0 and r > 1.0 + c) or (adv < 0.0 and r < 1.0 - c):
                continue
            row_g = g[s]
            p_row = ps[s]
            coeff = r * adv
            for j in range(n_cols):
                grad = coeff * ((1.0 if j == a else 0.0) - p_row[j])
                row_g[j] += grad
                if not math.isfinite(row_g[j]):
                    raise ValueError("gradient must be finite")
        mean_objective = epoch_total / n_batch
        if not math.isfinite(mean_objective):
            raise ValueError("epoch mean objective must be finite")
        objectives.append(mean_objective)
        for i in range(n_rows):
            for j in range(n_cols):
                g[i][j] /= n_batch
                new_value = L[i][j] + lr * g[i][j]
                if not math.isfinite(new_value):
                    raise ValueError("updated logits must be finite")
                L[i][j] = new_value

    probabilities = []
    for row in L:
        _, p = _log_softmax(row)
        probabilities.append(p)
    return {
        "logits": L,
        "objectives": objectives,
        "probabilities": probabilities,
    }


def ppo_train(
    env,
    episodes=100,
    alpha=0.05,
    gamma=0.9,
    lambda_=0.95,
    c=0.2,
    epochs=4,
    seed=0,
    max_steps=1000,
) -> dict:
    """PPO 训练（每回合 GAE 采样后多轮截断更新），返回 h 表、逐回合步表与目标值。

    从 S 可达的非 G 格按坐标升序编号为 0..n-1；H 为 n×4 的 logits 表
    （列序 U/R/D/L），V 为长度 n 的状态值表，初始均为 0.0。每回合
    reset，至 done 或 max_steps 步；每步按稳定 softmax(H[s,·]) 采样
    动作，仅调用一次 random()，并记录
    (s, a, r, old_logp, v, v2, done)：old_logp 为采样策略下所选动作的
    对数概率 H[s,a]-m-log(Σexp(H[s,·]-m))，v 为采样时 V[s]，done 时
    v2=0.0，否则为当时 V[s2]（步限截断仍自举）。回合末自 A=0.0 逆序
    递推 delta=r+gamma*v2-v、A=delta+gamma*lambda_*A；再将
    [s, a, old_logp, A] 批次传给
    ppo_update(H, batch, float(alpha), c, epochs)，以其返回 logits
    续训，随后按步序作 V[s]+=alpha*(A+v-V[s])，式中 v 为该步记录值。
    全部随机性来自一个 random.Random(seed)。

    返回键依次为 h、episodes、objectives；h 行按坐标升序为
    [r, c, lU, lR, lD, lL]，均为 float；episodes 为各回合的步列表，
    步项为 [r, c, action, reward, done]，action 为 U/R/D/L 字符串；
    objectives 为各回合 ppo_update 返回的 objectives。新运算结果非有限
    均抛 ValueError。
    """
    if not isinstance(env, GridWorld):
        raise TypeError("env must be a GridWorld")
    if isinstance(episodes, bool) or not isinstance(episodes, int):
        raise TypeError("episodes must be an int")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an int")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int):
        raise TypeError("max_steps must be an int")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise TypeError("alpha must be an int or float")
    if isinstance(gamma, bool) or not isinstance(gamma, (int, float)):
        raise TypeError("gamma must be an int or float")
    if isinstance(lambda_, bool) or not isinstance(lambda_, (int, float)):
        raise TypeError("lambda_ must be an int or float")
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if not _is_finite_number(alpha) or alpha <= 0 or alpha > 1:
        raise ValueError("alpha must be finite and in (0, 1]")
    if not _is_finite_number(gamma) or gamma < 0 or gamma > 1:
        raise ValueError("gamma must be finite and in [0, 1]")
    if not _is_finite_number(lambda_) or lambda_ < 0 or lambda_ > 1:
        raise ValueError("lambda_ must be finite and in [0, 1]")
    if not isinstance(c, float):
        raise TypeError("c must be a float")
    if not math.isfinite(c):
        raise ValueError("c must be finite")
    if c < 0.0 or c >= 1.0:
        raise ValueError("c must be in [0, 1)")
    if isinstance(epochs, bool) or not isinstance(epochs, int):
        raise TypeError("epochs must be a non-bool int")
    if epochs <= 0:
        raise ValueError("epochs must be positive")

    actions = tuple(_ACTIONS)  # U, R, D, L
    states = sorted(
        state
        for state in _reachable_cells(env)
        if env._cell(state) != "G"
    )
    state_index = {state: i for i, state in enumerate(states)}
    h = [[0.0 for _ in actions] for _ in states]
    values = [0.0 for _ in states]
    rng = random.Random(seed)
    all_episodes = []
    all_objectives = []

    for _ in range(episodes):
        state = env.reset()
        trajectory = []
        step_records = []
        for _ in range(max_steps):
            s = state_index[state]
            row = h[s]
            m = max(row)
            weights = [math.exp(logit - m) for logit in row]
            total = sum(weights)
            if not math.isfinite(total) or total <= 0.0:
                raise ValueError(
                    "softmax normalizer must be finite and positive"
                )
            probs = [weight / total for weight in weights]
            log_total = math.log(total)
            u = rng.random()
            cumulative = 0.0
            a = 3
            for k, p_k in enumerate(probs):
                cumulative += p_k
                if cumulative > u:
                    a = k
                    break
            action = actions[a]
            old_logp = row[a] - m - log_total
            if not math.isfinite(old_logp):
                raise ValueError("log-probabilities must be finite")
            value = values[s]
            next_state, reward, done = env.step(action)
            value_next = 0.0 if done else values[state_index[next_state]]
            trajectory.append(
                (s, a, reward, old_logp, value, value_next, done)
            )
            step_records.append(
                [state[0], state[1], action, reward, done]
            )
            if done:
                break
            state = next_state

        advantages = [0.0] * len(trajectory)
        a_t = 0.0
        for t in range(len(trajectory) - 1, -1, -1):
            (
                _s,
                _a,
                reward,
                _old_logp,
                value,
                value_next,
                _done,
            ) = trajectory[t]
            delta = reward + gamma * value_next - value
            if not math.isfinite(delta):
                raise ValueError("GAE delta must be finite")
            a_t = delta + gamma * lambda_ * a_t
            if not math.isfinite(a_t):
                raise ValueError("GAE advantage must be finite")
            advantages[t] = a_t

        batch = [
            [s, a, old_logp, adv]
            for (s, a, _r, old_logp, _v, _v2, _done), adv in zip(
                trajectory, advantages
            )
        ]
        result = ppo_update(h, batch, float(alpha), c, epochs)
        h = result["logits"]
        all_objectives.append(result["objectives"])

        for (s, _a, _r, _old_logp, value, _v2, _done), adv in zip(
            trajectory, advantages
        ):
            new_value = values[s] + alpha * (
                adv + value - values[s]
            )
            if not math.isfinite(new_value):
                raise ValueError("updated value must be finite")
            values[s] = new_value
        all_episodes.append(step_records)

    h_table = [
        [
            float(r),
            float(c),
            h[i][0],
            h[i][1],
            h[i][2],
            h[i][3],
        ]
        for i, (r, c) in enumerate(states)
    ]
    return {
        "h": h_table,
        "episodes": all_episodes,
        "objectives": all_objectives,
    }


def evaluate(env, h, episodes, max_steps, seed, window, threshold) -> dict:
    """按 softmax 策略评估偏好 H，返回回合统计与收敛判定。

    h 须为 {((row, col), action): float}，恰覆盖从 S 可达的非 G 格
    与 U/R/D/L 的全部组合，值为有限 float；evaluate 不修改 h。
    每回合 reset，单回合最多 max_steps 步；每步按 softmax(H[s,·])
    采样动作，仅调用一次 random()。全部随机性来自一个
    random.Random(seed)。

    返回 {"episodes": ..., "success_rate": ..., "converged": ...}；
    episodes 元素为 [steps, total_reward, success]，分别是实际步数、
    未折扣 reward 和、是否到达 G。converged 仅在 episodes >= window
    且末 window 回合成功率 >= threshold 时为 True。
    """
    if not isinstance(env, GridWorld):
        raise TypeError("env must be a GridWorld")
    if not isinstance(h, dict):
        raise TypeError("h must be a dict")
    if isinstance(episodes, bool) or not isinstance(episodes, int):
        raise TypeError("episodes must be an int")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int):
        raise TypeError("max_steps must be an int")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an int")
    if isinstance(window, bool) or not isinstance(window, int):
        raise TypeError("window must be an int")
    if isinstance(threshold, bool) or not isinstance(
        threshold, (int, float)
    ):
        raise TypeError("threshold must be an int or float")
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if window <= 0:
        raise ValueError("window must be positive")
    if not _is_finite_number(threshold) or threshold < 0 or threshold > 1:
        raise ValueError("threshold must be finite and in [0, 1]")

    actions = tuple(_ACTIONS)  # U, R, D, L
    states = sorted(
        state
        for state in _reachable_cells(env)
        if env._cell(state) != "G"
    )
    expected = {
        (state, action) for state in states for action in actions
    }
    if set(h) != expected:
        raise ValueError(
            "h must cover exactly the reachable non-G cells x U/R/D/L"
        )
    for value in h.values():
        if not isinstance(value, float) or not math.isfinite(value):
            raise ValueError("h values must be finite floats")

    rng = random.Random(seed)
    results = []
    for _ in range(episodes):
        state = env.reset()
        steps = 0
        total_reward = 0
        success = False
        for _ in range(max_steps):
            m = max(h[(state, a)] for a in actions)
            weights = [math.exp(h[(state, a)] - m) for a in actions]
            total = sum(weights)
            probs = [weight / total for weight in weights]
            u = rng.random()
            cumulative = 0.0
            action = "L"
            for a, p_a in zip(actions, probs):
                cumulative += p_a
                if cumulative > u:
                    action = a
                    break
            state, reward, done = env.step(action)
            steps += 1
            total_reward += reward
            if done:
                success = True
                break
        results.append([steps, total_reward, success])
    successes = sum(1 for result in results if result[2])
    converged = episodes >= window and (
        sum(1 for result in results[-window:] if result[2]) / window
        >= threshold
    )
    return {
        "episodes": results,
        "success_rate": successes / episodes,
        "converged": converged,
    }


def ppo_evaluate(
    env, h, episodes=100, max_steps=1000, seed=0, window=20, threshold=0.9
) -> dict:
    """按 softmax 策略评估 PPO 风格 logits 表，返回回合统计与收敛判定。

    h 须为 list，按从 S 可达的非 G 格坐标升序排列，每行恰为六项
    list [r, c, lU, lR, lD, lL]，均为有限 float，且 r、c 分别等于
    对应坐标的 float；h 或行非 list、元素非 float 抛 TypeError，
    空表、行长、坐标域/顺序或有限性错误抛 ValueError。评估前将
    r、c 转为 int，按 U/R/D/L 把四个 logit 映射为 evaluate 的
    {((row, col), action): float} 偏好；其余参数校验与异常、回合
    边界、稳定 softmax 采样及 random.Random(seed) 消费顺序均沿用
    evaluate，且不修改 h。

    返回键依次为 episodes、success_rate、windows、converged；
    episodes 元素为 [steps, total_reward, success]，success_rate
    为成功回合比例，均与 evaluate 一致；windows 按起点升序，含每个
    完整长度为 window 的滑窗内 success 的算术均值（float）；
    converged 仅在 windows 非空且末项 >= threshold 时为 True。
    相同输入与 seed 逐值一致。
    """
    if not isinstance(env, GridWorld):
        raise TypeError("env must be a GridWorld")
    if not isinstance(h, list):
        raise TypeError("h must be a list")
    if isinstance(episodes, bool) or not isinstance(episodes, int):
        raise TypeError("episodes must be an int")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int):
        raise TypeError("max_steps must be an int")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an int")
    if isinstance(window, bool) or not isinstance(window, int):
        raise TypeError("window must be an int")
    if isinstance(threshold, bool) or not isinstance(
        threshold, (int, float)
    ):
        raise TypeError("threshold must be an int or float")
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if window <= 0:
        raise ValueError("window must be positive")
    if not _is_finite_number(threshold) or threshold < 0 or threshold > 1:
        raise ValueError("threshold must be finite and in [0, 1]")

    actions = tuple(_ACTIONS)  # U, R, D, L
    states = sorted(
        state
        for state in _reachable_cells(env)
        if env._cell(state) != "G"
    )
    if not h:
        raise ValueError("h must be non-empty")
    if len(h) != len(states):
        raise ValueError(
            "h must have exactly one row per reachable non-G cell"
        )
    preferences = {}
    for row, (er, ec) in zip(h, states):
        if not isinstance(row, list):
            raise TypeError("every row of h must be a list")
        if len(row) != 6:
            raise ValueError(
                "every row of h must have exactly six elements"
            )
        for item in row:
            if not isinstance(item, float):
                raise TypeError("h rows must contain only float")
            if not math.isfinite(item):
                raise ValueError("h rows must contain only finite float")
        if row[0] != float(er) or row[1] != float(ec):
            raise ValueError(
                "h rows must match the reachable non-G coordinates"
                " in ascending order"
            )
        state = (int(row[0]), int(row[1]))
        for action, logit in zip(actions, row[2:]):
            preferences[(state, action)] = logit

    result = evaluate(
        env, preferences, episodes, max_steps, seed, window, threshold
    )
    results = result["episodes"]
    windows = [
        sum(1 for item in results[start:start + window] if item[2])
        / window
        for start in range(len(results) - window + 1)
    ]
    converged = bool(windows) and windows[-1] >= threshold
    return {
        "episodes": results,
        "success_rate": result["success_rate"],
        "windows": windows,
        "converged": converged,
    }


def ppo_evaluate_trace(env, h, episodes=100, max_steps=1000, seed=0) -> dict:
    """按 softmax 策略评估 PPO 风格 logits 表，返回可逐步复现的轨迹。

    参数合法域及 TypeError/ValueError 分类与 ppo_evaluate 同名参数
    一致（无 window、threshold）；评估前同样将每行 r、c 转为 int，
    按 U/R/D/L 映射为偏好表，且不修改 h。每回合 reset，单回合最多
    max_steps 步；URDL 稳定 softmax 及单个 random.Random(seed)
    每步恰取一次 random 的消费顺序均沿用 ppo_evaluate。

    每步追加 [r, c, action, next_r, next_c, reward, done]：r、c 为
    调用 step 前状态，next_r、next_c 为返回状态，坐标与 reward 为
    int，action 为 str，done 为 bool。到达 G 立即结束；到达步限
    截断时保留末步且末步 done=False。

    返回键依次为 episodes、success_rate；episodes 为回合 dict 列表，
    各项键序 trace、total_reward、success，trace 为步记录列表，
    total_reward 为未折扣 int 和，success 仅表示是否到达 G；
    success_rate 为成功数除以回合数的 float。相同输入与 seed
    逐值一致。
    """
    if not isinstance(env, GridWorld):
        raise TypeError("env must be a GridWorld")
    if not isinstance(h, list):
        raise TypeError("h must be a list")
    if isinstance(episodes, bool) or not isinstance(episodes, int):
        raise TypeError("episodes must be an int")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int):
        raise TypeError("max_steps must be an int")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an int")
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")

    actions = tuple(_ACTIONS)  # U, R, D, L
    states = sorted(
        state
        for state in _reachable_cells(env)
        if env._cell(state) != "G"
    )
    if not h:
        raise ValueError("h must be non-empty")
    if len(h) != len(states):
        raise ValueError(
            "h must have exactly one row per reachable non-G cell"
        )
    preferences = {}
    for row, (er, ec) in zip(h, states):
        if not isinstance(row, list):
            raise TypeError("every row of h must be a list")
        if len(row) != 6:
            raise ValueError(
                "every row of h must have exactly six elements"
            )
        for item in row:
            if not isinstance(item, float):
                raise TypeError("h rows must contain only float")
            if not math.isfinite(item):
                raise ValueError("h rows must contain only finite float")
        if row[0] != float(er) or row[1] != float(ec):
            raise ValueError(
                "h rows must match the reachable non-G coordinates"
                " in ascending order"
            )
        state = (int(row[0]), int(row[1]))
        for action, logit in zip(actions, row[2:]):
            preferences[(state, action)] = logit

    rng = random.Random(seed)
    episode_records = []
    successes = 0
    for _ in range(episodes):
        state = env.reset()
        trace = []
        total_reward = 0
        success = False
        for _ in range(max_steps):
            r, c = state
            m = max(preferences[(state, a)] for a in actions)
            weights = [
                math.exp(preferences[(state, a)] - m) for a in actions
            ]
            total = sum(weights)
            probs = [weight / total for weight in weights]
            u = rng.random()
            cumulative = 0.0
            action = "L"
            for a, p_a in zip(actions, probs):
                cumulative += p_a
                if cumulative > u:
                    action = a
                    break
            next_state, reward, done = env.step(action)
            next_r, next_c = next_state
            trace.append(
                [r, c, action, next_r, next_c, reward, done]
            )
            total_reward += reward
            state = next_state
            if done:
                success = True
                break
        if success:
            successes += 1
        episode_records.append(
            {
                "trace": trace,
                "total_reward": total_reward,
                "success": success,
            }
        )
    return {
        "episodes": episode_records,
        "success_rate": successes / episodes,
    }


def ppo_evaluate_many(
    env, h, seeds, episodes=100, max_steps=1000, window=20, threshold=0.9
) -> dict:
    """按多种子评估 PPO 风格 logits 表，返回汇总统计。

    seeds 须为非空 list/tuple，元素为非 bool 的 int，允许重复且不修改
    seeds；容器或元素类型不符抛 TypeError，空序列抛 ValueError。其余
    参数校验与异常沿用 ppo_evaluate。按 seeds 顺序逐项复用
    ppo_evaluate，每项仅替换 seed（各自创建独立随机流），h 不变。

    返回键依次为 results、means、passed、converged；results 与 seeds
    同序，每项为键序 seed、result 的 dict，result 即对应 ppo_evaluate
    的完整返回值。means 键序 success、last_window：success 为各
    result.success_rate 按 results 序从 0.0 累加后除以项数所得
    float；last_window 对 windows 非空的 result 的末项同算，无项则
    为 None。passed 为 result.converged 为 True 的数量；converged
    等价于 passed == len(seeds)。相同输入逐值一致。
    """
    if not isinstance(seeds, (list, tuple)):
        raise TypeError("seeds must be a list or tuple")
    if not seeds:
        raise ValueError("seeds must be non-empty")
    for seed in seeds:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("every seed must be a non-bool int")

    results = []
    for seed in seeds:
        result = ppo_evaluate(
            env,
            h,
            episodes=episodes,
            max_steps=max_steps,
            seed=seed,
            window=window,
            threshold=threshold,
        )
        results.append({"seed": seed, "result": result})

    success_total = 0.0
    for item in results:
        success_total += item["result"]["success_rate"]
    success_mean = success_total / len(results)

    last_window_total = 0.0
    last_window_count = 0
    for item in results:
        windows = item["result"]["windows"]
        if windows:
            last_window_total += windows[-1]
            last_window_count += 1
    last_window_mean = (
        last_window_total / last_window_count if last_window_count else None
    )

    passed = sum(1 for item in results if item["result"]["converged"])
    return {
        "results": results,
        "means": {"success": success_mean, "last_window": last_window_mean},
        "passed": passed,
        "converged": passed == len(seeds),
    }


def ppo_stability_report(data, threshold=0.9, stable_ratio=1.0) -> dict:
    """汇总 ppo_evaluate_many 的跨种子滑窗稳定性，返回报告字典。

    data 须为 ppo_evaluate_many 的完整返回值：顶层键序依次为
    results、means、passed、converged；results 为非空 list，每项为
    键序 seed、result 的 dict，seed 为非 bool 的 int（允许重复），
    result 为 ppo_evaluate 的完整返回（键序 episodes、success_rate、
    windows、converged）；episodes 为非空 list，每行恰为三项
    list：steps 为非 bool 正 int，reward 为有限的非 bool int/float，
    done 为 bool；windows 为 list，每项为 [0, 1] 内的有限 float。
    data 非 dict 抛 TypeError；顶层键序、results 非空、seed/result
    层级、episodes 行结构或 windows 取值违约均抛 ValueError，且不
    修改 data。threshold、stable_ratio 须为非 bool 的 int/float，
    类型错抛 TypeError，非有限或越出 [0, 1] 抛 ValueError。

    按 i 升序取各 result.windows[i]（缺失该下标的 result 略过），
    依 results 序从 0.0 累加，得行
    [i, count, mean, min, max, pass_rate]：i、count 为 int，其余为
    float；count 为实际取到的项数，mean 为其算术均值，min、max 为
    其最小值、最大值，pass_rate 为其中值 >= threshold 的项所占
    比例。stable_window 为首个覆盖全部 results（count 等于 results
    项数）且 pass_rate >= stable_ratio 的 i，无则 None。返回键序
    依次为 seeds、windows、stable_window、stable：seeds 按 results
    序保留（含重复 seed）；windows 为上述行表，所有 result 均无窗口
    时为 []；stable 等价于 stable_window 非 None。相同输入逐值一致。
    """
    if not isinstance(data, dict):
        raise TypeError("data must be a dict")
    if isinstance(threshold, bool) or not isinstance(
        threshold, (int, float)
    ):
        raise TypeError("threshold must be an int or float")
    if isinstance(stable_ratio, bool) or not isinstance(
        stable_ratio, (int, float)
    ):
        raise TypeError("stable_ratio must be an int or float")
    if not _is_finite_number(threshold) or threshold < 0 or threshold > 1:
        raise ValueError("threshold must be finite and in [0, 1]")
    if (
        not _is_finite_number(stable_ratio)
        or stable_ratio < 0
        or stable_ratio > 1
    ):
        raise ValueError("stable_ratio must be finite and in [0, 1]")

    if list(data) != ["results", "means", "passed", "converged"]:
        raise ValueError(
            "data must have exactly the keys results, means, passed,"
            " converged"
        )
    entries = data["results"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("data['results'] must be a non-empty list")

    seeds = []
    windows_list = []
    for entry in entries:
        if not isinstance(entry, dict) or list(entry) != ["seed", "result"]:
            raise ValueError(
                "every result entry must be a dict with keys seed, result"
            )
        seed = entry["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("every seed must be a non-bool int")
        result = entry["result"]
        if not isinstance(result, dict) or list(result) != [
            "episodes",
            "success_rate",
            "windows",
            "converged",
        ]:
            raise ValueError(
                "every result must have exactly the keys episodes,"
                " success_rate, windows, converged"
            )
        episodes = result["episodes"]
        if not isinstance(episodes, list) or not episodes:
            raise ValueError("episodes must be a non-empty list")
        for row in episodes:
            if not isinstance(row, list) or len(row) != 3:
                raise ValueError(
                    "every episode row must be a list of three fields"
                )
            steps, reward, done = row
            if isinstance(steps, bool) or not isinstance(steps, int):
                raise ValueError("steps must be an int")
            if steps <= 0:
                raise ValueError("steps must be positive")
            if isinstance(reward, bool) or not isinstance(
                reward, (int, float)
            ):
                raise ValueError("reward must be an int or float")
            if not _is_finite_number(reward):
                raise ValueError("reward must be finite")
            if not isinstance(done, bool):
                raise ValueError("done must be a bool")
        windows = result["windows"]
        if not isinstance(windows, list):
            raise ValueError("windows must be a list")
        for value in windows:
            if not isinstance(value, float) or not math.isfinite(value):
                raise ValueError("windows values must be finite floats")
            if value < 0 or value > 1:
                raise ValueError("windows values must be in [0, 1]")
        seeds.append(seed)
        windows_list.append(windows)

    total_count = len(windows_list)
    max_len = max((len(windows) for windows in windows_list), default=0)
    report_windows = []
    stable_window = None
    for i in range(max_len):
        values = [
            windows[i] for windows in windows_list if i < len(windows)
        ]
        count = len(values)
        total = 0.0
        for value in values:
            total += value
        mean = total / count
        pass_rate = sum(1 for value in values if value >= threshold) / count
        report_windows.append(
            [i, count, mean, min(values), max(values), pass_rate]
        )
        if (
            stable_window is None
            and count == total_count
            and pass_rate >= stable_ratio
        ):
            stable_window = i

    return {
        "seeds": seeds,
        "windows": report_windows,
        "stable_window": stable_window,
        "stable": stable_window is not None,
    }


def _validate_evaluate_many_payload(data, name):
    """校验一份 ppo_evaluate_many 完整返回值，返回 results 项列表。"""
    if list(data) != ["results", "means", "passed", "converged"]:
        raise ValueError(
            f"{name} must have exactly the keys results, means,"
            " passed, converged"
        )
    entries = data["results"]
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{name}['results'] must be a non-empty list")

    for entry in entries:
        if not isinstance(entry, dict) or list(entry) != ["seed", "result"]:
            raise ValueError(
                f"every {name} result entry must be a dict with keys"
                " seed, result"
            )
        seed = entry["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"every {name} seed must be a non-bool int")
        result = entry["result"]
        if not isinstance(result, dict) or list(result) != [
            "episodes",
            "success_rate",
            "windows",
            "converged",
        ]:
            raise ValueError(
                f"every {name} result must have exactly the keys"
                " episodes, success_rate, windows, converged"
            )
        episodes = result["episodes"]
        if not isinstance(episodes, list) or not episodes:
            raise ValueError(f"{name} episodes must be a non-empty list")
        for row in episodes:
            if not isinstance(row, list) or len(row) != 3:
                raise ValueError(
                    f"every {name} episode row must be a list of three"
                    " fields"
                )
            steps, reward, success = row
            if isinstance(steps, bool) or not isinstance(steps, int):
                raise ValueError(f"{name} steps must be a non-bool int")
            if steps <= 0:
                raise ValueError(f"{name} steps must be positive")
            if isinstance(reward, bool) or not isinstance(
                reward, (int, float)
            ):
                raise ValueError(f"{name} reward must be an int or float")
            if not _is_finite_number(reward):
                raise ValueError(f"{name} reward must be finite")
            if not isinstance(success, bool):
                raise ValueError(f"{name} success flag must be a bool")
        rate = result["success_rate"]
        if not isinstance(rate, float):
            raise ValueError(f"{name} success_rate must be a float")
        if not math.isfinite(rate):
            raise ValueError(f"{name} success_rate must be finite")
        if rate < 0.0 or rate > 1.0:
            raise ValueError(f"{name} success_rate must be in [0, 1]")
        windows = result["windows"]
        if not isinstance(windows, list):
            raise ValueError(f"{name} windows must be a list")
        for value in windows:
            if not isinstance(value, float):
                raise ValueError(
                    f"{name} windows values must be floats"
                )
            if not math.isfinite(value):
                raise ValueError(
                    f"{name} windows values must be finite floats"
                )
            if value < 0.0 or value > 1.0:
                raise ValueError(
                    f"{name} windows values must be in [0, 1]"
                )
        if not isinstance(result["converged"], bool):
            raise ValueError(f"{name} result converged flag must be a bool")

    means = data["means"]
    if not isinstance(means, dict) or list(means) != [
        "success",
        "last_window",
    ]:
        raise ValueError(
            f"{name} means must be a dict with keys success, last_window"
        )
    success_mean = means["success"]
    if not isinstance(success_mean, float):
        raise ValueError(f"{name} means.success must be a float")
    if not math.isfinite(success_mean):
        raise ValueError(f"{name} means.success must be finite")
    if success_mean < 0.0 or success_mean > 1.0:
        raise ValueError(f"{name} means.success must be in [0, 1]")
    last_window_mean = means["last_window"]
    if last_window_mean is not None:
        if not isinstance(last_window_mean, float):
            raise ValueError(
                f"{name} means.last_window must be None or a float"
            )
        if not math.isfinite(last_window_mean):
            raise ValueError(
                f"{name} means.last_window must be finite when not None"
            )
        if last_window_mean < 0.0 or last_window_mean > 1.0:
            raise ValueError(
                f"{name} means.last_window must be in [0, 1]"
            )
    passed = data["passed"]
    if isinstance(passed, bool) or not isinstance(passed, int):
        raise ValueError(f"{name} passed must be a non-bool int")
    if not isinstance(data["converged"], bool):
        raise ValueError(f"{name} converged must be a bool")

    success_total = 0.0
    for entry in entries:
        success_total += entry["result"]["success_rate"]
    expected_success = success_total / len(entries)
    if float.hex(expected_success) != float.hex(success_mean):
        raise ValueError(
            f"{name} means.success must equal the mean of success_rate"
        )

    last_total = 0.0
    last_count = 0
    for entry in entries:
        windows = entry["result"]["windows"]
        if windows:
            last_total += windows[-1]
            last_count += 1
    if last_count == 0:
        if last_window_mean is not None:
            raise ValueError(
                f"{name} means.last_window must be None when no result"
                " has windows"
            )
    else:
        if last_window_mean is None:
            raise ValueError(
                f"{name} means.last_window must not be None when a result"
                " has windows"
            )
        expected_last = last_total / last_count
        if float.hex(expected_last) != float.hex(last_window_mean):
            raise ValueError(
                f"{name} means.last_window must equal the mean of the"
                " last window values"
            )

    expected_passed = sum(
        1 for entry in entries if entry["result"]["converged"]
    )
    if passed != expected_passed:
        raise ValueError(
            f"{name} passed must equal the number of converged results"
        )
    if data["converged"] != (expected_passed == len(entries)):
        raise ValueError(
            f"{name} converged must equal (passed == number of results)"
        )
    return entries


def ppo_reproducibility_report(left, right) -> dict:
    """逐值比较两份 ppo_evaluate_many 返回值，输出可复现性报告。

    left、right 须为 dict，否则抛 TypeError；每份都须严格符合
    ppo_evaluate_many 的完整返回契约：顶层键序为 results、means、
    passed、converged；results 为非空 list，每项键序为 seed、result，
    seed 为非 bool int（允许重复），result 键序为 episodes、
    success_rate、windows、converged；episodes 为非空 list，行恰为
    [steps, reward, success]，steps 为非 bool 正 int，reward 为有限的
    非 bool int/float，success 为 bool；success_rate 为 [0, 1] 内有限
    float，windows 为 [0, 1] 内有限 float 的 list（可空），converged
    为 bool；means 键序为 success、last_window，前者为 [0, 1] 内有限
    float，后者为同类 float 或 None；passed 为非 bool int，converged
    为 bool。means 两字段、passed、converged 还须分别等于公开汇总等式
    （success_rate 自 0.0 累加均值、对 windows 非空 result 的末窗同算
    均值或无项时 None、converged 计数、passed==results 项数）。任一
    层结构、键序、类型、非 bool、有限性或范围违约均抛 ValueError；
    不修改输入。

    两侧 results 长度不同或同位置 seed 不同抛 ValueError；重复 seed
    按位置区分。校验通过后按公开键序深度优先比较全部字段，results 按
    位置比较：序列先比长度（路径追加 .length），再按索引（追加 [i]）；
    标量要求类型与值均相同，float 以 float.hex() 比较（故 0.0 与
    -0.0 视为不同），int 与 float 即使数值相同也视为不同。差异路径以
    $ 开头，字典字段追加 .name，无差异路径为 None。

    返回键序为 identical、results、first_difference：results 按种子
    顺序保留（含重复），每项键序为 seed、identical、first_difference，
    后两项为该位置整项（seed、result）的比较结论与首差绝对路径；
    顶层 first_difference 覆盖全部字段（results 先于 means、passed、
    converged）；identical 仅在相应范围逐值相同时为 True。相同输入
    逐值一致，仅用标准库，不引入命令行入口。
    """
    if not isinstance(left, dict):
        raise TypeError("left must be a dict")
    if not isinstance(right, dict):
        raise TypeError("right must be a dict")
    left_entries = _validate_evaluate_many_payload(left, "left")
    right_entries = _validate_evaluate_many_payload(right, "right")

    if len(left_entries) != len(right_entries):
        raise ValueError(
            "left and right results lists must have equal length"
        )
    for index, (left_entry, right_entry) in enumerate(
        zip(left_entries, right_entries)
    ):
        if left_entry["seed"] != right_entry["seed"]:
            raise ValueError(
                f"seeds at position {index} must match"
            )

    def _scalar_equal(a, b):
        if type(a) is not type(b):
            return False
        if isinstance(a, float):
            return float.hex(a) == float.hex(b)
        return a == b

    def _first_difference(a, b, path):
        """深度优先按公开键序返回首个差异路径，无差异返回 None。"""
        if isinstance(a, dict):
            for key in a:
                difference = _first_difference(
                    a[key], b[key], f"{path}.{key}"
                )
                if difference is not None:
                    return difference
            return None
        if isinstance(a, list):
            if len(a) != len(b):
                return f"{path}.length"
            for index, (item_a, item_b) in enumerate(zip(a, b)):
                difference = _first_difference(
                    item_a, item_b, f"{path}[{index}]"
                )
                if difference is not None:
                    return difference
            return None
        return None if _scalar_equal(a, b) else path

    items = []
    for index, (left_entry, right_entry) in enumerate(
        zip(left_entries, right_entries)
    ):
        entry_path = f"$.results[{index}]"
        difference = _first_difference(
            left_entry, right_entry, entry_path
        )
        items.append(
            {
                "seed": left_entry["seed"],
                "identical": difference is None,
                "first_difference": difference,
            }
        )
    overall = _first_difference(left, right, "$")
    return {
        "identical": overall is None,
        "results": items,
        "first_difference": overall,
    }


def _fingerprint_encode(value):
    """将 None/bool/int/float/str/list/dict 递归编码为确定性文本。"""
    if value is None:
        return "N;"
    if isinstance(value, bool):
        return "B1;" if value else "B0;"
    if isinstance(value, int):
        return "I" + str(value) + ";"
    if isinstance(value, float):
        return "F" + float.hex(value) + ";"
    if isinstance(value, str):
        return "S" + str(len(value.encode("utf-8"))) + ":" + value
    if isinstance(value, list):
        parts = [
            "L" + str(len(value)) + ":"
        ]
        for item in value:
            parts.append(_fingerprint_encode(item))
        return "".join(parts)
    if isinstance(value, dict):
        parts = [
            "D" + str(len(value)) + ":"
        ]
        for key, item_value in value.items():
            parts.append(_fingerprint_encode(key))
            parts.append(_fingerprint_encode(item_value))
        return "".join(parts)
    raise TypeError("fingerprint values must be JSON-like data")


def _fingerprint_hex(value):
    return hashlib.sha256(
        _fingerprint_encode(value).encode("utf-8")
    ).hexdigest()


def ppo_reproducibility_fingerprint(data) -> dict:
    """为一份 ppo_evaluate_many 返回值计算结构指纹。

    data 须为 dict，否则抛 TypeError；且须严格符合
    ppo_reproducibility_report 单侧输入的完整契约（结构、键序、
    类型、非 bool、有限性、范围及汇总等式），任一违约抛 ValueError；
    不修改输入。

    依次判型并递归编码：None=`N;`；bool=`B0;`/`B1;`；
    int=`I`+十进制+`;`；float=`F`+float.hex()+`;`；
    str=`S`+UTF-8 字节数+`:`+原文；list=`L`+元素数+`:`+各元素；
    dict=`D`+键数+`:`+按插入（公开）键序拼接键、值编码。编码文本取
    UTF-8 字节后做 sha256。对完整 data 及 results 各完整项（含
    seed、result）分别摘要。返回键序 algorithm、fingerprint、
    results，依次为 "sha256"、总体摘要、与 results 同序的逐项摘要
    list；摘要均为小写 64 位十六进制 str。深拷贝同摘要；0.0 与
    -0.0、int 与 float 摘要不同。仅用标准库，不引入命令行入口。
    """
    if not isinstance(data, dict):
        raise TypeError("data must be a dict")
    entries = _validate_evaluate_many_payload(data, "data")

    result_fingerprints = [
        _fingerprint_hex(entry) for entry in entries
    ]
    return {
        "algorithm": "sha256",
        "fingerprint": _fingerprint_hex(data),
        "results": result_fingerprints,
    }


_HEX_DIGITS = frozenset("0123456789abcdef")


def _validate_fingerprint_payload(data, name):
    """校验一份指纹结构，返回 (总体摘要, 逐项摘要 list)，不修改输入。"""
    if list(data) != ["algorithm", "fingerprint", "results"]:
        raise ValueError(
            f"{name} must have exactly the keys algorithm, fingerprint,"
            " results"
        )
    algorithm = data["algorithm"]
    if not isinstance(algorithm, str):
        raise TypeError(f"{name} algorithm must be a str")
    if algorithm != "sha256":
        raise ValueError(f"{name} algorithm must be 'sha256'")

    fingerprint = data["fingerprint"]
    if not isinstance(fingerprint, str):
        raise TypeError(f"{name} fingerprint must be a str")
    if len(fingerprint) != 64 or any(
        char not in _HEX_DIGITS for char in fingerprint
    ):
        raise ValueError(
            f"{name} fingerprint must be a 64-character lowercase hex"
            " string"
        )

    results = data["results"]
    if not isinstance(results, list):
        raise TypeError(f"{name} results must be a list")
    if not results:
        raise ValueError(f"{name} results must be non-empty")
    for index, digest in enumerate(results):
        if not isinstance(digest, str):
            raise TypeError(f"{name} results[{index}] must be a str")
        if len(digest) != 64 or any(
            char not in _HEX_DIGITS for char in digest
        ):
            raise ValueError(
                f"{name} results[{index}] must be a 64-character"
                " lowercase hex string"
            )
    return fingerprint, results


def ppo_reproducibility_fingerprint_compare(left, right) -> dict:
    """逐位置比较两份 ppo_reproducibility_fingerprint 返回值。

    left、right 须为 dict，否则抛 TypeError；各自键序须恰为
    algorithm、fingerprint、results：algorithm 须为 "sha256"，
    fingerprint 须为 64 位小写十六进制 str，results 须为非空 list，
    成员为同格式 str（重复摘要允许，按位置处理）。字段类型错抛
    TypeError；键序、空列表、算法值或摘要格式错抛 ValueError。两侧
    均完整校验通过后才比较，不修改输入，也不返回部分结果。

    两侧 results 长度不同抛 ValueError。返回键序为 identical、results、
    mismatches：results 为与输入摘要列表等长的 bool 列表，依次表示同
    位置摘要是否相等；mismatches 为所有不等位置的升序零基 int 列表；
    identical 仅当总体 fingerprint 相等且 results 全为 True。相同输入
    及深拷贝逐值一致，仅用标准库，不引入命令行入口。
    """
    if not isinstance(left, dict):
        raise TypeError("left must be a dict")
    if not isinstance(right, dict):
        raise TypeError("right must be a dict")
    left_fingerprint, left_results = _validate_fingerprint_payload(
        left, "left"
    )
    right_fingerprint, right_results = _validate_fingerprint_payload(
        right, "right"
    )
    if len(left_results) != len(right_results):
        raise ValueError(
            "left and right results lists must have equal length"
        )

    equal_flags = [
        left_digest == right_digest
        for left_digest, right_digest in zip(left_results, right_results)
    ]
    mismatches = [
        index for index, equal in enumerate(equal_flags) if not equal
    ]
    return {
        "identical": left_fingerprint == right_fingerprint
        and all(equal_flags),
        "results": equal_flags,
        "mismatches": mismatches,
    }


def ppo_reproducibility_fingerprint_consensus(items) -> dict:
    """以第 0 项为基准汇总多份 ppo_reproducibility_fingerprint 返回值。

    items 须为至少两项的 list：非 list 抛 TypeError，少于两项抛
    ValueError。各项须为 dict 且完整符合
    ppo_reproducibility_fingerprint_compare 的单侧载荷契约：键序须恰为
    algorithm、fingerprint、results，algorithm 须为 "sha256"，
    fingerprint 须为 64 位小写十六进制 str，results 须为非空 list，
    成员为同格式 str（重复摘要允许，按位置处理）。项本身或字段类型错
    抛 TypeError；键序、算法值、空列表或摘要格式错抛 ValueError。先按
    索引完整校验全部项，再以第 0 项为基准；任一校验失败均不返回部分
    结果，不修改输入。

    任一对 results 长度不同抛 ValueError。返回键序为 identical、
    comparisons、divergent_runs：comparisons 与 items 等长，其 [0] 为
    ppo_reproducibility_fingerprint_compare 对第 0 项自身的完整返回，
    第 i 项为其对第 0 项与 items[i] 的完整返回；divergent_runs 为其中
    identical 为 False 的非零索引按升序组成的 list；顶层 identical 当
    且仅当该列表为空。重复调用及深拷贝逐值一致，仅用标准库，不引入
    命令行入口。
    """
    if not isinstance(items, list):
        raise TypeError("items must be a list")
    if len(items) < 2:
        raise ValueError("items must contain at least two entries")

    result_lists = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise TypeError(f"items[{index}] must be a dict")
        _, results = _validate_fingerprint_payload(
            item, f"items[{index}]"
        )
        result_lists.append(results)

    baseline_length = len(result_lists[0])
    for index in range(1, len(result_lists)):
        if len(result_lists[index]) != baseline_length:
            raise ValueError(
                f"items[{index}] results list must have the same length"
                " as items[0]"
            )

    comparisons = [
        ppo_reproducibility_fingerprint_compare(items[0], item)
        for item in items
    ]
    divergent_runs = [
        index
        for index, comparison in enumerate(comparisons)
        if index != 0 and not comparison["identical"]
    ]
    return {
        "identical": not divergent_runs,
        "comparisons": comparisons,
        "divergent_runs": divergent_runs,
    }


def ppo_seed_reproducibility_report(runs) -> dict:
    """按 seed 分组汇总多种子指纹一致性，返回可复现性报告。

    runs 须为至少两项的 list：非 list 抛 TypeError，少于两项抛
    ValueError。每项须为键序恰为 seed、payload 的 dict：seed 为非
    bool 的 int；payload 须完整符合
    ppo_reproducibility_fingerprint_compare 的单侧载荷契约（键序
    algorithm、fingerprint、results，algorithm 为 "sha256"，
    fingerprint 为 64 位小写十六进制 str，results 为非空 list，成员为
    同格式 str）。项本身或字段类型错抛 TypeError；键序、算法值、摘要
    格式或空列表错抛 ValueError。先按索引完整校验全部项，再按 seed
    首次出现顺序分组、组内保持原序；任一 seed 少于两项或同 seed 的
    results 长度不一均抛 ValueError，任一失败都不返回部分结果，且不
    修改输入。

    对各组按组内原序的 payload 调用
    ppo_reproducibility_fingerprint_consensus。返回键序为
    identical、groups、divergent_seeds：groups 按 seed 首次出现序排列，
    每项键序为 seed、indices、consensus，indices 为该组各项在原 runs
    中的零基索引升序 list，consensus 为该组共识的完整返回值；
    divergent_seeds 按组序收集 consensus.identical 为 False 的 seed；
    identical 当且仅当 divergent_seeds 为空。重复调用及深拷贝逐值
    一致，仅用标准库，不引入命令行入口。
    """
    if not isinstance(runs, list):
        raise TypeError("runs must be a list")
    if len(runs) < 2:
        raise ValueError("runs must contain at least two entries")

    validated = []
    for index, item in enumerate(runs):
        if not isinstance(item, dict):
            raise TypeError(f"runs[{index}] must be a dict")
        if list(item) != ["seed", "payload"]:
            raise ValueError(
                f"runs[{index}] must have exactly the keys seed, payload"
            )
        seed = item["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError(f"runs[{index}] seed must be a non-bool int")
        payload = item["payload"]
        if not isinstance(payload, dict):
            raise TypeError(f"runs[{index}] payload must be a dict")
        _validate_fingerprint_payload(
            payload, f"runs[{index}] payload"
        )
        validated.append((seed, payload))

    seed_order = []
    indices_by_seed = {}
    payloads_by_seed = {}
    for index, (seed, payload) in enumerate(validated):
        if seed not in indices_by_seed:
            seed_order.append(seed)
            indices_by_seed[seed] = []
            payloads_by_seed[seed] = []
        indices_by_seed[seed].append(index)
        payloads_by_seed[seed].append(payload)

    for seed in seed_order:
        group_payloads = payloads_by_seed[seed]
        if len(indices_by_seed[seed]) < 2:
            raise ValueError(f"seed {seed} must have at least two runs")
        baseline_length = len(group_payloads[0]["results"])
        for payload in group_payloads[1:]:
            if len(payload["results"]) != baseline_length:
                raise ValueError(
                    f"all runs with seed {seed} must have results lists of"
                    " equal length"
                )

    groups = []
    divergent_seeds = []
    for seed in seed_order:
        consensus = ppo_reproducibility_fingerprint_consensus(
            payloads_by_seed[seed]
        )
        groups.append(
            {
                "seed": seed,
                "indices": indices_by_seed[seed],
                "consensus": consensus,
            }
        )
        if not consensus["identical"]:
            divergent_seeds.append(seed)

    return {
        "identical": not divergent_seeds,
        "groups": groups,
        "divergent_seeds": divergent_seeds,
    }


def ppo_seed_trajectory_report(runs) -> dict:
    """按 seed 分组逐值比较多份 ppo_evaluate_many 返回值，输出轨迹报告。

    runs 须为至少两项的 list：非 list 抛 TypeError，少于两项抛
    ValueError。每项须为键序恰为 seed、payload 的 dict：seed 为非
    bool 的 int；payload 须完整符合 ppo_evaluate_many 的完整返回契约
    （即 ppo_reproducibility_report 的单侧输入契约：顶层键序为
    results、means、passed、converged；results 为非空 list，逐项
    seed、result 层级、类型、范围及 means、passed、converged 汇总
    等式均成立）。项本身或 payload 非 dict、seed 为 bool 或非 int
    抛 TypeError；键序或 payload 契约违约抛 ValueError。先按索引
    完整校验全部项，再按 seed 首次出现顺序分组、组内保持原序；任一
    seed 少于两项、同 seed 各 payload 的 results 长度不一或同位置
    results 项的 seed 不同均抛 ValueError，任一失败都不返回部分结果，
    且不修改输入。

    对各组以组内首个 payload 为基准，按组内原序（含基准自身）逐一调用
    ppo_reproducibility_report：逐值比较、首差路径、完整报告及异常均
    沿用该接口。返回键序为 identical、groups、unreproducible_seeds：
    groups 按 seed 首次出现序排列，每项键序为 seed、indices、reports、
    identical，indices 为该组各项在原 runs 中的零基索引升序 list，
    reports 为与组内各项同序的完整报告 list（其 [0] 为基准对自身的
    报告），identical 当且仅当各报告的 identical 全为 True；
    unreproducible_seeds 按组序收集组 identical 为 False 的 seed；
    顶层 identical 当且仅当该列表为空。重复调用及深拷贝逐值一致，仅
    用标准库，不引入命令行入口。
    """
    if not isinstance(runs, list):
        raise TypeError("runs must be a list")
    if len(runs) < 2:
        raise ValueError("runs must contain at least two entries")

    validated = []
    for index, item in enumerate(runs):
        if not isinstance(item, dict):
            raise TypeError(f"runs[{index}] must be a dict")
        if list(item) != ["seed", "payload"]:
            raise ValueError(
                f"runs[{index}] must have exactly the keys seed, payload"
            )
        seed = item["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError(f"runs[{index}] seed must be a non-bool int")
        payload = item["payload"]
        if not isinstance(payload, dict):
            raise TypeError(f"runs[{index}] payload must be a dict")
        _validate_evaluate_many_payload(
            payload, f"runs[{index}] payload"
        )
        validated.append((seed, payload))

    seed_order = []
    indices_by_seed = {}
    payloads_by_seed = {}
    for index, (seed, payload) in enumerate(validated):
        if seed not in indices_by_seed:
            seed_order.append(seed)
            indices_by_seed[seed] = []
            payloads_by_seed[seed] = []
        indices_by_seed[seed].append(index)
        payloads_by_seed[seed].append(payload)

    for seed in seed_order:
        group_payloads = payloads_by_seed[seed]
        if len(indices_by_seed[seed]) < 2:
            raise ValueError(f"seed {seed} must have at least two runs")
        baseline_entries = group_payloads[0]["results"]
        for payload in group_payloads[1:]:
            entries = payload["results"]
            if len(entries) != len(baseline_entries):
                raise ValueError(
                    f"all runs with seed {seed} must have results lists of"
                    " equal length"
                )
            for position, (baseline_entry, entry) in enumerate(
                zip(baseline_entries, entries)
            ):
                if baseline_entry["seed"] != entry["seed"]:
                    raise ValueError(
                        f"all runs with seed {seed} must have the same seed"
                        f" at results position {position}"
                    )

    groups = []
    unreproducible_seeds = []
    for seed in seed_order:
        group_payloads = payloads_by_seed[seed]
        baseline_payload = group_payloads[0]
        reports = [
            ppo_reproducibility_report(baseline_payload, payload)
            for payload in group_payloads
        ]
        group_identical = all(
            report["identical"] for report in reports
        )
        groups.append(
            {
                "seed": seed,
                "indices": indices_by_seed[seed],
                "reports": reports,
                "identical": group_identical,
            }
        )
        if not group_identical:
            unreproducible_seeds.append(seed)

    return {
        "identical": not unreproducible_seeds,
        "groups": groups,
        "unreproducible_seeds": unreproducible_seeds,
    }


def ppo_seed_stability_report(runs) -> dict:
    """按 seed 分组从轨迹、收敛、汇总三维度判定多种子稳定性，返回报告。

    runs 的全量校验、异常、按 seed 首次出现顺序分组及组约束完全沿用
    ppo_seed_trajectory_report：runs 须为至少两项的 list，非 list 抛
    TypeError，少于两项抛 ValueError；每项须为键序恰为 seed、payload
    的 dict，seed 为非 bool 的 int，payload 须完整符合
    ppo_evaluate_many 的完整返回契约（即 ppo_reproducibility_report 的
    单侧输入契约）；项本身或 payload 非 dict、seed 为 bool 或非 int
    抛 TypeError；键序或 payload 契约违约抛 ValueError；任一 seed 少于
    两项、同 seed 各 payload 的 results 长度不一或同位置 results 项的
    seed 不同抛 ValueError。先按索引完整校验全部项再分组，任一失败都
    不返回部分结果，且不修改输入。

    各组以组内首个 payload 为基准，对组内各 payload（含基准自身）按
    组内原序调用 ppo_reproducibility_report，保留同序完整 reports，其
    [0] 为基准对自身的报告；逐值口径沿用该报告：标量要求类型与值均
    相同，float 以 float.hex() 比较（0.0 与 -0.0 不同），int 与 float
    即使数值相同也不同，序列先比长度再按位置比较。令 R 为各
    payload.results 中同位置 result 跨组内各 payload 构成的同序列表，
    在三个维度上以基准为参照逐位置逐值判定全组是否一致：T 比较 R 各项
    的 episodes；C 比较 R 各项 result 的 converged 以及各 payload 顶层
    的 converged；M 比较 R 各项 result 的 success_rate、windows 以及各
    payload 的 means、passed。

    返回键序为 stable、groups、unstable_seeds：groups 按 seed 首次出现
    序排列，每项为 [seed, indices, reports, [T, C, M]]，indices 为该组
    各项在原 runs 中的零基索引升序 list，reports 为与组内各项同序的
    完整报告 list，T、C、M 为该维全组一致的 bool；unstable_seeds 按组
    序收集任一维为 False 的 seed；stable 当且仅当 unstable_seeds 为空。
    重复调用及深拷贝逐值一致，仅用标准库，不引入命令行入口。
    """
    if not isinstance(runs, list):
        raise TypeError("runs must be a list")
    if len(runs) < 2:
        raise ValueError("runs must contain at least two entries")

    validated = []
    for index, item in enumerate(runs):
        if not isinstance(item, dict):
            raise TypeError(f"runs[{index}] must be a dict")
        if list(item) != ["seed", "payload"]:
            raise ValueError(
                f"runs[{index}] must have exactly the keys seed, payload"
            )
        seed = item["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError(f"runs[{index}] seed must be a non-bool int")
        payload = item["payload"]
        if not isinstance(payload, dict):
            raise TypeError(f"runs[{index}] payload must be a dict")
        _validate_evaluate_many_payload(
            payload, f"runs[{index}] payload"
        )
        validated.append((seed, payload))

    seed_order = []
    indices_by_seed = {}
    payloads_by_seed = {}
    for index, (seed, payload) in enumerate(validated):
        if seed not in indices_by_seed:
            seed_order.append(seed)
            indices_by_seed[seed] = []
            payloads_by_seed[seed] = []
        indices_by_seed[seed].append(index)
        payloads_by_seed[seed].append(payload)

    for seed in seed_order:
        group_payloads = payloads_by_seed[seed]
        if len(indices_by_seed[seed]) < 2:
            raise ValueError(f"seed {seed} must have at least two runs")
        baseline_entries = group_payloads[0]["results"]
        for payload in group_payloads[1:]:
            entries = payload["results"]
            if len(entries) != len(baseline_entries):
                raise ValueError(
                    f"all runs with seed {seed} must have results lists of"
                    " equal length"
                )
            for position, (baseline_entry, entry) in enumerate(
                zip(baseline_entries, entries)
            ):
                if baseline_entry["seed"] != entry["seed"]:
                    raise ValueError(
                        f"all runs with seed {seed} must have the same seed"
                        f" at results position {position}"
                    )

    def _values_equal(left, right):
        """沿用 ppo_reproducibility_report 的逐值口径递归比较。"""
        if type(left) is not type(right):
            return False
        if isinstance(left, float):
            return float.hex(left) == float.hex(right)
        if isinstance(left, list):
            if len(left) != len(right):
                return False
            return all(
                _values_equal(item_a, item_b)
                for item_a, item_b in zip(left, right)
            )
        if isinstance(left, dict):
            if list(left) != list(right):
                return False
            return all(
                _values_equal(left[key], right[key]) for key in left
            )
        return left == right

    groups = []
    unstable_seeds = []
    for seed in seed_order:
        group_payloads = payloads_by_seed[seed]
        baseline_payload = group_payloads[0]
        reports = [
            ppo_reproducibility_report(baseline_payload, payload)
            for payload in group_payloads
        ]

        trajectory_stable = True
        convergence_stable = True
        metrics_stable = True
        baseline_results = baseline_payload["results"]
        for payload in group_payloads[1:]:
            for baseline_entry, entry in zip(
                baseline_results, payload["results"]
            ):
                baseline_result = baseline_entry["result"]
                result = entry["result"]
                if not _values_equal(
                    baseline_result["episodes"], result["episodes"]
                ):
                    trajectory_stable = False
                if baseline_result["converged"] != result["converged"]:
                    convergence_stable = False
                if not _values_equal(
                    baseline_result["success_rate"],
                    result["success_rate"],
                ):
                    metrics_stable = False
                if not _values_equal(
                    baseline_result["windows"], result["windows"]
                ):
                    metrics_stable = False
            if baseline_payload["converged"] != payload["converged"]:
                convergence_stable = False
            if not _values_equal(
                baseline_payload["means"], payload["means"]
            ):
                metrics_stable = False
            if baseline_payload["passed"] != payload["passed"]:
                metrics_stable = False

        groups.append(
            [
                seed,
                indices_by_seed[seed],
                reports,
                [trajectory_stable, convergence_stable, metrics_stable],
            ]
        )
        if not (
            trajectory_stable
            and convergence_stable
            and metrics_stable
        ):
            unstable_seeds.append(seed)

    return {
        "stable": not unstable_seeds,
        "groups": groups,
        "unstable_seeds": unstable_seeds,
    }


def ppo_seed_robustness(runs, minimum=0.8, gap=0.1) -> dict:
    """汇总多种子 ppo_evaluate_many 结果的鲁棒性，返回判定报告。

    runs 须为至少两项的 list：非 list 抛 TypeError，少于两项抛
    ValueError。每项须为恰含两元素的 list [seed, payload]：seed 为
    互异的非 bool int；payload 须完整符合 ppo_evaluate_many 的完整
    返回契约（即 ppo_reproducibility_report 的单侧输入契约：顶层键序
    为 results、means、passed、converged，各层结构、类型、范围及
    means、passed、converged 汇总等式均成立）。runs、项或 payload
    容器类型错、seed 为 bool 或非 int 抛 TypeError；项长度不为二、
    seed 重复或 payload 契约违约抛 ValueError。minimum、gap 须为非
    bool 的 int/float，类型错抛 TypeError，非有限或越出 [0, 1] 抛
    ValueError。先全量校验全部参数再计算，任一失败都不返回部分结果，
    且不修改输入。

    设 n 为项数。S 为各 payload.means.success 按输入序从 0.0 累加后
    除以 n 所得 float；L 仅对非 None 的 means.last_window 同算，无项
    时 L 为 None，否则为 float；R 为 payload.converged 为 True 的项数
    除以 n 所得 float。若某项 abs(means.success - S) > gap，其 seed
    为离群。返回键序为 robust、rate、means、failed、outliers：rate
    为 R；means 为 [S, L]；failed 按输入序收集 converged 为 False 的
    seed；outliers 按输入序收集离群 seed；robust 当且仅当
    R >= minimum 且 outliers 为空。相同输入逐值一致，仅用标准库，不
    引入命令行入口。
    """
    if not isinstance(runs, list):
        raise TypeError("runs must be a list")
    if len(runs) < 2:
        raise ValueError("runs must contain at least two entries")
    if isinstance(minimum, bool) or not isinstance(minimum, (int, float)):
        raise TypeError("minimum must be an int or float")
    if isinstance(gap, bool) or not isinstance(gap, (int, float)):
        raise TypeError("gap must be an int or float")
    if not _is_finite_number(minimum) or minimum < 0 or minimum > 1:
        raise ValueError("minimum must be finite and in [0, 1]")
    if not _is_finite_number(gap) or gap < 0 or gap > 1:
        raise ValueError("gap must be finite and in [0, 1]")

    validated = []
    seen_seeds = set()
    for index, item in enumerate(runs):
        if not isinstance(item, list):
            raise TypeError(f"runs[{index}] must be a list")
        if len(item) != 2:
            raise ValueError(
                f"runs[{index}] must have exactly two elements"
            )
        seed, payload = item
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError(f"runs[{index}] seed must be a non-bool int")
        if seed in seen_seeds:
            raise ValueError(f"runs[{index}] seed must be distinct")
        seen_seeds.add(seed)
        if not isinstance(payload, dict):
            raise TypeError(f"runs[{index}] payload must be a dict")
        _validate_evaluate_many_payload(
            payload, f"runs[{index}] payload"
        )
        validated.append((seed, payload))

    n = len(validated)
    success_total = 0.0
    for _, payload in validated:
        success_total += payload["means"]["success"]
    success_mean = success_total / n

    last_window_total = 0.0
    last_window_count = 0
    for _, payload in validated:
        last_window = payload["means"]["last_window"]
        if last_window is not None:
            last_window_total += last_window
            last_window_count += 1
    last_window_mean = (
        last_window_total / last_window_count if last_window_count else None
    )

    converged_count = sum(
        1 for _, payload in validated if payload["converged"]
    )
    rate = converged_count / n

    failed = [
        seed for seed, payload in validated if not payload["converged"]
    ]
    outliers = [
        seed
        for seed, payload in validated
        if abs(payload["means"]["success"] - success_mean) > gap
    ]
    return {
        "robust": rate >= minimum and not outliers,
        "rate": rate,
        "means": [success_mean, last_window_mean],
        "failed": failed,
        "outliers": outliers,
    }


def _validate_seed_runs(runs, name):
    """按 ppo_seed_robustness 的 runs 契约校验并返回 [(seed, payload)]。"""
    if not isinstance(runs, list):
        raise TypeError(f"{name} must be a list")
    if len(runs) < 2:
        raise ValueError(f"{name} must contain at least two entries")
    validated = []
    seen_seeds = set()
    for index, item in enumerate(runs):
        if not isinstance(item, list):
            raise TypeError(f"{name}[{index}] must be a list")
        if len(item) != 2:
            raise ValueError(
                f"{name}[{index}] must have exactly two elements"
            )
        seed, payload = item
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError(f"{name}[{index}] seed must be a non-bool int")
        if seed in seen_seeds:
            raise ValueError(f"{name}[{index}] seed must be distinct")
        seen_seeds.add(seed)
        if not isinstance(payload, dict):
            raise TypeError(f"{name}[{index}] payload must be a dict")
        _validate_evaluate_many_payload(
            payload, f"{name}[{index}] payload"
        )
        validated.append((seed, payload))
    return validated


def ppo_seed_regression(base, new, gap=0.0) -> dict:
    """逐种子对比两组 ppo_evaluate_many 结果，返回回归判定报告。

    base、new 均须满足 ppo_seed_robustness 的 runs 输入契约与异常：
    至少两项的 list，每项为恰含两元素的 list [seed, payload]，seed
    为互异的非 bool int，payload 完整符合 ppo_evaluate_many 的返回
    契约；容器类型错、seed 为 bool 或非 int 抛 TypeError，项长度不
    为二、seed 重复或 payload 契约违约抛 ValueError。base 与 new 的
    seed 集合必须相同（顺序可不同），否则抛 ValueError。gap 须为非
    bool 的 int/float，类型错抛 TypeError，非有限或越出 [0, 1] 抛
    ValueError。先全量校验全部参数再计算，任一失败都不返回部分结
    果，且不修改输入。

    校验通过后按 base 顺序逐种子配对。每对取
    ds = new.means.success - base.means.success；当两侧
    means.last_window 均非 None 时同法取 dl，否则 dl 为 None。返回
    键序为 passed、rows、means、improved、degraded：rows 各项为
    [seed, ds, dl]；means 为 [S, L]，S 为全部 ds 按 rows 序从 0.0
    累加后除以项数所得 float 均值，L 为非 None 的 dl 同法所得均
    值、无项时为 None；improved、degraded 按 rows 序分别收集
    ds > gap 与 ds < -gap 的 seed；passed 当且仅当 S >= gap 且
    degraded 为空。ds、S 及非 None 的 dl、L 均为 float。相同输入逐
    值一致，仅用标准库，不引入命令行入口。
    """
    base_validated = _validate_seed_runs(base, "base")
    new_validated = _validate_seed_runs(new, "new")
    if isinstance(gap, bool) or not isinstance(gap, (int, float)):
        raise TypeError("gap must be an int or float")
    if not _is_finite_number(gap) or gap < 0 or gap > 1:
        raise ValueError("gap must be finite and in [0, 1]")

    new_by_seed = dict(new_validated)
    if {seed for seed, _ in base_validated} != set(new_by_seed):
        raise ValueError("base and new must have the same seed set")

    rows = []
    improved = []
    degraded = []
    success_total = 0.0
    last_window_total = 0.0
    last_window_count = 0
    for seed, base_payload in base_validated:
        new_payload = new_by_seed[seed]
        ds = float(
            new_payload["means"]["success"]
            - base_payload["means"]["success"]
        )
        base_last_window = base_payload["means"]["last_window"]
        new_last_window = new_payload["means"]["last_window"]
        if base_last_window is not None and new_last_window is not None:
            dl = float(new_last_window - base_last_window)
            last_window_total += dl
            last_window_count += 1
        else:
            dl = None
        rows.append([seed, ds, dl])
        success_total += ds
        if ds > gap:
            improved.append(seed)
        if ds < -gap:
            degraded.append(seed)

    n = len(rows)
    success_mean = success_total / n
    last_window_mean = (
        last_window_total / last_window_count if last_window_count else None
    )
    return {
        "passed": success_mean >= gap and not degraded,
        "rows": rows,
        "means": [success_mean, last_window_mean],
        "improved": improved,
        "degraded": degraded,
    }


def ppo_seed_select(base, candidates, gap=0.0) -> dict:
    """在多个候选 runs 中按回归报告选出最优者，返回选择报告。

    base 及各候选 runs 均沿用 ppo_seed_robustness 的 runs 输入契约与
    异常：至少两项的 list，每项为恰含两元素的 list [seed, payload]，
    seed 为互异的非 bool int，payload 完整符合 ppo_evaluate_many 的
    返回契约；runs、项或 payload 容器类型错、seed 为 bool 或非 int
    抛 TypeError，项长度不为二、seed 重复或 payload 契约违约抛
    ValueError。candidates 须为非空 list，否则非 list 抛 TypeError、
    空表抛 ValueError；每项须为恰含两元素的 list [name, runs]：name
    须为非空 str 且各项互异，项本身非 list 或 name 非 str 抛
    TypeError，项长度不为二、name 为空或重名抛 ValueError。各候选
    runs 的 seed 集合须与 base 完全相同（顺序可不同），否则抛
    ValueError。gap 的类型、有限性与范围校验沿用 ppo_seed_regression：
    非 bool int/float，类型错抛 TypeError，非有限或越出 [0, 1] 抛
    ValueError。先全量校验全部参数再计算，任一失败都不返回部分结
    果，且不修改输入。

    校验通过后按 candidates 顺序对每个候选调用
    ppo_seed_regression(base, runs, gap)。仅报告 passed 为 True 的
    候选参与选择，以报告 means[0]（成功均值差 S）最大者胜，同值取
    candidates 中较前者；无合格候选时 selected 为 None。返回键序为
    selected、reports、rejected：selected 为胜者 name 或 None；
    reports 与 candidates 同序，每项为 [name, 完整回归报告]；
    rejected 为 passed 为 False 的候选 name 按 candidates 序组成的
    list。相同输入逐值一致，仅用标准库，不引入命令行入口。
    """
    base_validated = _validate_seed_runs(base, "base")
    if not isinstance(candidates, list):
        raise TypeError("candidates must be a list")
    if not candidates:
        raise ValueError("candidates must be non-empty")
    if isinstance(gap, bool) or not isinstance(gap, (int, float)):
        raise TypeError("gap must be an int or float")
    if not _is_finite_number(gap) or gap < 0 or gap > 1:
        raise ValueError("gap must be finite and in [0, 1]")

    base_seeds = {seed for seed, _ in base_validated}
    validated_candidates = []
    seen_names = set()
    for index, item in enumerate(candidates):
        if not isinstance(item, list):
            raise TypeError(f"candidates[{index}] must be a list")
        if len(item) != 2:
            raise ValueError(
                f"candidates[{index}] must have exactly two elements"
            )
        name, runs = item
        if not isinstance(name, str):
            raise TypeError(f"candidates[{index}] name must be a str")
        if not name:
            raise ValueError(f"candidates[{index}] name must be non-empty")
        if name in seen_names:
            raise ValueError(f"candidates[{index}] name must be distinct")
        seen_names.add(name)
        runs_validated = _validate_seed_runs(
            runs, f"candidates[{index}] runs"
        )
        if {seed for seed, _ in runs_validated} != base_seeds:
            raise ValueError(
                f"candidates[{index}] runs must have the same seed set as"
                " base"
            )
        validated_candidates.append((name, runs))

    reports = []
    rejected = []
    selected = None
    best_mean = None
    for name, runs in validated_candidates:
        report = ppo_seed_regression(base, runs, gap)
        reports.append([name, report])
        if report["passed"]:
            mean = report["means"][0]
            if selected is None or mean > best_mean:
                selected = name
                best_mean = mean
        else:
            rejected.append(name)
    return {
        "selected": selected,
        "reports": reports,
        "rejected": rejected,
    }


def convergence_report(
    episode_results, window=20, tolerance=0.01, patience=3
) -> dict:
    """根据逐回合 reward 的滑窗均值判定收敛，返回报告字典。

    episode_results 每项须为 [steps, reward, done] 三项 list：steps 为
    非 bool 正 int，reward 为可转为有限 float 的非 bool int/float，
    done 为 bool；空列表合法且视为未收敛。window、patience 为非 bool
    正 int，tolerance 为非 bool 有限 int/float 且 >= 0。

    对每个长度为 window 的完整连续滑窗按
    sum(map(float, r), 0.0) / window 求 reward 均值，依序得 float
    列表 M。找最小 j >= patience，使 abs(M[k]-M[k-1]) <= tolerance
    对 k=j-patience+1..j 均成立。命中时返回
    True、window+j（从 1 起的回合号）、M[j]、M；否则返回
    False、None、None、M。M 不截断，输入不被修改。
    """
    if not isinstance(episode_results, list):
        raise TypeError("episode_results must be a list")
    if isinstance(window, bool) or not isinstance(window, int):
        raise TypeError("window must be an int")
    if isinstance(patience, bool) or not isinstance(patience, int):
        raise TypeError("patience must be an int")
    if isinstance(tolerance, bool) or not isinstance(
        tolerance, (int, float)
    ):
        raise TypeError("tolerance must be an int or float")
    if window <= 0:
        raise ValueError("window must be positive")
    if patience <= 0:
        raise ValueError("patience must be positive")
    if not _is_finite_number(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and >= 0")

    for row in episode_results:
        if not isinstance(row, list):
            raise TypeError("every episode result must be a list")
        if len(row) != 3:
            raise ValueError("every episode result must have 3 fields")
        steps, reward, done = row
        if isinstance(steps, bool) or not isinstance(steps, int):
            raise TypeError("steps must be an int")
        if steps <= 0:
            raise ValueError("steps must be positive")
        if isinstance(reward, bool) or not isinstance(
            reward, (int, float)
        ):
            raise TypeError("reward must be an int or float")
        try:
            reward_float = float(reward)
        except OverflowError:
            raise ValueError(
                "reward must be convertible to a finite float"
            )
        if not math.isfinite(reward_float):
            raise ValueError("reward must be a finite number")
        if not isinstance(done, bool):
            raise TypeError("done must be a bool")

    rewards = [row[1] for row in episode_results]
    means = []
    for start in range(len(rewards) - window + 1):
        r = rewards[start:start + window]
        means.append(sum(map(float, r), 0.0) / window)

    converged = False
    episode = None
    reward_mean = None
    for j in range(patience, len(means)):
        if all(
            abs(means[k] - means[k - 1]) <= tolerance
            for k in range(j - patience + 1, j + 1)
        ):
            converged = True
            episode = window + j
            reward_mean = means[j]
            break
    return {
        "converged": converged,
        "episode": episode,
        "reward_mean": reward_mean,
        "windows": means,
    }


def ppo_seed_convergence_report(
    runs, window=20, tolerance=0.01, patience=3
) -> dict:
    """按 seed 分组汇总多种子逐回合 reward 的收敛判定，返回汇总报告。

    runs 须为至少两项的 list：非 list 抛 TypeError，少于两项抛
    ValueError。每项须为键序恰为 seed、episode_results 的 dict：
    seed 为非 bool 的 int，episode_results 须为 list；项本身或字段
    类型错抛 TypeError，键序错抛 ValueError。各 episode_results 与
    window、tolerance、patience 的校验、异常与算法完全沿用
    convergence_report。先依输入序完整校验全部成员（含三个参数），
    再按 seed 首次出现顺序分组、组内保持原序；任一 seed 少于两项
    抛 ValueError，任一失败都不返回部分结果，且不修改输入。

    对各组按组内原序的每个 episode_results 调用 convergence_report
    （window、tolerance、patience 相同）。返回键序为 converged、
    groups、inconsistent_seeds、unconverged_seeds：groups 按 seed
    首次出现序排列，每项键序固定为 seed、indices、reports、
    consistent、converged，indices 为该组各项在原 runs 中的零基
    索引升序 list，reports 为按组内原序排列的完整 convergence_report
    返回值 list，consistent 表示各报告 converged 是否全同，converged
    表示各报告 converged 是否全真；inconsistent_seeds、
    unconverged_seeds 分别按组序收集 consistent、converged 为 False
    的 seed，顶层 converged 当且仅当 unconverged_seeds 为空。重复
    调用及深拷贝逐值一致，仅用标准库，不引入命令行入口。
    """
    if not isinstance(runs, list):
        raise TypeError("runs must be a list")
    if len(runs) < 2:
        raise ValueError("runs must contain at least two entries")

    validated = []
    for index, item in enumerate(runs):
        if not isinstance(item, dict):
            raise TypeError(f"runs[{index}] must be a dict")
        if list(item) != ["seed", "episode_results"]:
            raise ValueError(
                f"runs[{index}] must have exactly the keys"
                " seed, episode_results"
            )
        seed = item["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError(f"runs[{index}] seed must be a non-bool int")
        episode_results = item["episode_results"]
        if not isinstance(episode_results, list):
            raise TypeError(
                f"runs[{index}] episode_results must be a list"
            )
        # 完整沿用 convergence_report 的入参与成员契约，先全量校验。
        convergence_report(
            episode_results,
            window=window,
            tolerance=tolerance,
            patience=patience,
        )
        validated.append((seed, episode_results))

    seed_order = []
    indices_by_seed = {}
    results_by_seed = {}
    for index, (seed, episode_results) in enumerate(validated):
        if seed not in indices_by_seed:
            seed_order.append(seed)
            indices_by_seed[seed] = []
            results_by_seed[seed] = []
        indices_by_seed[seed].append(index)
        results_by_seed[seed].append(episode_results)

    for seed in seed_order:
        if len(indices_by_seed[seed]) < 2:
            raise ValueError(f"seed {seed} must have at least two runs")

    groups = []
    inconsistent_seeds = []
    unconverged_seeds = []
    for seed in seed_order:
        reports = [
            convergence_report(
                episode_results,
                window=window,
                tolerance=tolerance,
                patience=patience,
            )
            for episode_results in results_by_seed[seed]
        ]
        converged_flags = [report["converged"] for report in reports]
        consistent = all(
            flag == converged_flags[0] for flag in converged_flags
        )
        all_converged = all(converged_flags)
        groups.append(
            {
                "seed": seed,
                "indices": indices_by_seed[seed],
                "reports": reports,
                "consistent": consistent,
                "converged": all_converged,
            }
        )
        if not consistent:
            inconsistent_seeds.append(seed)
        if not all_converged:
            unconverged_seeds.append(seed)

    return {
        "converged": not unconverged_seeds,
        "groups": groups,
        "inconsistent_seeds": inconsistent_seeds,
        "unconverged_seeds": unconverged_seeds,
    }


def ppo_success_streak(successes, window=20, threshold=0.9, patience=3) -> dict:
    """根据逐回合成功标记的滑窗成功率判定连续收敛，返回报告字典。

    successes 须为非空 list/tuple 且仅含 bool；容器或元素类型错抛
    TypeError，空序列抛 ValueError。window、patience 须为非 bool 正
    int；threshold 须为非 bool 的 int/float，类型错抛 TypeError，转
    float 溢出、非有限或越出 [0, 1] 抛 ValueError。阈值先转为 float。

    对每个长度为 window 的完整连续窗，按零基起点 i 升序令 x 为窗内
    True 数、r=x/window，生成 [i+1, i+window, x, r]，前三项为 int、
    r 为 float。找最小 j >= patience-1，使第 j-patience+1 至 j 行的
    r 均不小于阈值；命中时 episode 为第 j 行第二项（i+window），否则
    为 None。无完整窗或窗数少于 patience 均不收敛。

    返回键序 converged、episode、windows：converged 为 episode 是否
    非 None，episode 为 int 或 None，windows 为全部窗行列表。不修改
    输入，相同输入逐值一致，仅用标准库。
    """
    if not isinstance(successes, (list, tuple)):
        raise TypeError("successes must be a list or tuple")
    if len(successes) == 0:
        raise ValueError("successes must be non-empty")
    for item in successes:
        if not isinstance(item, bool):
            raise TypeError("successes must contain only bool")
    if isinstance(window, bool) or not isinstance(window, int):
        raise TypeError("window must be a non-bool int")
    if isinstance(patience, bool) or not isinstance(patience, int):
        raise TypeError("patience must be a non-bool int")
    if isinstance(threshold, bool) or not isinstance(
        threshold, (int, float)
    ):
        raise TypeError("threshold must be an int or float")
    if window <= 0:
        raise ValueError("window must be positive")
    if patience <= 0:
        raise ValueError("patience must be positive")
    try:
        t = float(threshold)
    except OverflowError:
        raise ValueError("threshold must convert to a finite float")
    if not math.isfinite(t):
        raise ValueError("threshold must be finite")
    if t < 0.0 or t > 1.0:
        raise ValueError("threshold must be in [0, 1]")

    n = len(successes)
    windows = []
    for i in range(n - window + 1):
        x = 0
        for k in range(i, i + window):
            if successes[k]:
                x += 1
        windows.append([i + 1, i + window, x, x / window])

    episode = None
    for j in range(patience - 1, len(windows)):
        if all(
            windows[k][3] >= t
            for k in range(j - patience + 1, j + 1)
        ):
            episode = windows[j][1]
            break
    return {
        "converged": episode is not None,
        "episode": episode,
        "windows": windows,
    }


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
