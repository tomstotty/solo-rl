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
