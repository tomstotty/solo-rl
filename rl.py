"""GridWorld 环境与命令行入口，仅用标准库。"""

import hashlib
import json
import math
import os
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


def policy_iteration(env, gamma=0.9, tolerance=1e-9, max_iterations=1000):
    """同步策略迭代，返回 (policy, values, iterations)。

    policy 为 {(row, col): "U|R|D|L"}，values 为 {(row, col): float}，
    键为从 S 四向避墙可达的非 G 格子（按坐标升序）；
    iterations 为实际迭代轮数（每轮含一次策略评估与一次策略改进）。
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

    reachable = sorted(_reachable_cells(env))
    states = [state for state in reachable if env._cell(state) != "G"]
    policy = {state: "U" for state in states}
    values = {state: 0.0 for state in reachable}

    iterations = 0
    while iterations < max_iterations:
        # 策略评估：以上轮 V 为初值，固定策略同步迭代至最大变化不超过 tolerance。
        while True:
            new_values = {state: 0.0 for state in reachable}
            delta = 0.0
            for state in states:
                nxt, reward, _done = env.transition(state, policy[state])
                updated = reward + gamma * values[nxt]
                if not math.isfinite(updated):
                    raise ValueError("non-finite value during policy evaluation")
                new_values[state] = updated
                change = abs(updated - values[state])
                if change > delta:
                    delta = change
            values = new_values
            if delta <= tolerance:
                break

        # 策略改进：按 URDL 计算 reward + gamma*V[next]，取首个严格最大者。
        new_policy = {}
        stable = True
        for state in states:
            best_action = None
            best_value = None
            for action in _ACTIONS:
                nxt, reward, _done = env.transition(state, action)
                candidate = reward + gamma * values[nxt]
                if not math.isfinite(candidate):
                    raise ValueError("non-finite value during policy improvement")
                if best_value is None or candidate > best_value:
                    best_value = candidate
                    best_action = action
            new_policy[state] = best_action
            if best_action != policy[state]:
                stable = False

        iterations += 1
        policy = new_policy
        if stable:
            ordered_policy = {state: policy[state] for state in states}
            ordered_values = {}
            for state in states:
                value = float(values[state])
                if not math.isfinite(value):
                    raise ValueError("non-finite value in policy iteration output")
                ordered_values[state] = value
            return ordered_policy, ordered_values, iterations

    raise RuntimeError("policy iteration did not converge")


def q_learning(
    env,
    episodes=500,
    alpha=0.5,
    gamma=0.9,
    epsilon=0.1,
    seed=0,
    epsilon_end=None,
    epsilon_mode="linear",
) -> dict:
    """Q-learning，返回 {((row, col), action): float}。

    Q 覆盖从 S 可达的非 G 格与 U/R/D/L 的全部组合，初始为 0.0。
    每回合 reset，单回合最多 1000 步；全部随机性来自一个
    random.Random(seed)。

    epsilon_end 为 None 时各回合探索率恒为 epsilon；否则 episodes 为 1
    时恒为 epsilon，第 e 回合（0 起）令 t=e/(episodes-1)（截断到
    [0, 1]），按 epsilon_mode 在 epsilon 与 epsilon_end 间变化：
    linear 为 epsilon+(epsilon_end-epsilon)*t；cosine 为
    epsilon_end+(epsilon-epsilon_end)*(1+cos(pi*t))/2；
    exponential 为 epsilon*(epsilon_end/epsilon)**t。
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
    if not isinstance(epsilon_mode, str):
        raise TypeError("epsilon_mode must be a str")
    if epsilon_mode not in ("linear", "cosine", "exponential"):
        raise ValueError(
            "epsilon_mode must be one of linear, cosine, exponential"
        )
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
    if epsilon_mode == "exponential" and epsilon_end is not None:
        if epsilon <= 0 or epsilon_end <= 0:
            raise ValueError(
                "exponential epsilon schedule endpoints must be in (0, 1]"
            )

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
            t = episode / (episodes - 1)
            if t < 0.0:
                t = 0.0
            elif t > 1.0:
                t = 1.0
            if epsilon_mode == "linear":
                rate = epsilon + (epsilon_end - epsilon) * t
            elif epsilon_mode == "cosine":
                rate = epsilon_end + (
                    epsilon - epsilon_end
                ) * (1.0 + math.cos(math.pi * t)) / 2.0
            else:
                rate = epsilon * (epsilon_end / epsilon) ** t
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
            new_value = q[key] + alpha * (target - q[key])
            if not math.isfinite(new_value):
                raise ValueError("Q value must remain finite")
            q[key] = new_value
            if done:
                break
            state = next_state
    return q


def ucb_q_learning(
    env,
    episodes=500,
    alpha=0.5,
    gamma=0.9,
    c=1.0,
    max_steps=1000,
    c_end=None,
    c_mode="linear",
) -> dict:
    """UCB1 探索的确定性 Q-learning，返回键序 q、counts 的 dict。

    Q、N 均覆盖从 S 可达的非 G 格与 U/R/D/L 的全部组合，键序为坐标
    升序 × URDL，初始分别为 0.0、0，两表互相独立。每回合 reset，单
    回合最多 max_steps 步；全程不消费任何随机性。

    c_end 为 None 时各回合探索系数恒为 c；否则 episodes 为 1 时恒为
    c，第 e 回合（0 起）令 t=e/(episodes-1)，按 c_mode 在 c 与 c_end
    间变化：linear 为 c+(c_end-c)*t；cosine 为
    c_end+(c-c_end)*(1+cos(pi*t))/2；exponential 为
    c*(c_end/c)**t，指数模式两端点必须为正。

    状态 s 若存在 N[s,a]==0 的动作，取 URDL 中首个未访问动作；否则
    取 Q[s,a]+c_e*sqrt(log(Σb N[s,b])/N[s,a]) 最大且 URDL 中首个的
    动作。任一回合系数 c_e 或各动作分数非有限时，在 step 前抛
    ValueError。step 后先令 N[s,a]+=1；done 时目标为 reward，否则
    目标为 reward+gamma*max_a Q[next,a]，随后
    Q[s,a]+=alpha*(目标-Q[s,a])。到达 G 立即结束；到达步限截断时
    末步 done=False，仍照常自举更新。reward 或新 Q 非有限时抛
    ValueError。
    """
    if not isinstance(env, GridWorld):
        raise TypeError("env must be a GridWorld")
    if isinstance(episodes, bool) or not isinstance(episodes, int):
        raise TypeError("episodes must be an int")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int):
        raise TypeError("max_steps must be an int")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise TypeError("alpha must be an int or float")
    if isinstance(gamma, bool) or not isinstance(gamma, (int, float)):
        raise TypeError("gamma must be an int or float")
    if isinstance(c, bool) or not isinstance(c, (int, float)):
        raise TypeError("c must be an int or float")
    if c_end is not None and (
        isinstance(c_end, bool) or not isinstance(c_end, (int, float))
    ):
        raise TypeError("c_end must be None, an int or float")
    if not isinstance(c_mode, str):
        raise TypeError("c_mode must be a str")
    if c_mode not in ("linear", "cosine", "exponential"):
        raise ValueError(
            "c_mode must be one of linear, cosine, exponential"
        )
    try:
        alpha = float(alpha)
        gamma = float(gamma)
        c = float(c)
        if c_end is not None:
            c_end = float(c_end)
    except OverflowError:
        raise ValueError(
            "parameter is too large to convert to float"
        )
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if not math.isfinite(alpha) or alpha <= 0 or alpha > 1:
        raise ValueError("alpha must be finite and in (0, 1]")
    if not math.isfinite(gamma) or gamma < 0 or gamma >= 1:
        raise ValueError("gamma must be finite and in [0, 1)")
    if not math.isfinite(c) or c < 0:
        raise ValueError("c must be finite and in [0, +inf)")
    if c_end is not None and (not math.isfinite(c_end) or c_end < 0):
        raise ValueError("c_end must be finite and in [0, +inf)")
    if c_mode == "exponential" and c_end is not None:
        if c <= 0 or c_end <= 0:
            raise ValueError(
                "exponential c schedule endpoints must be positive"
            )

    actions = tuple(_ACTIONS)  # U, R, D, L
    states = sorted(
        state
        for state in _reachable_cells(env)
        if env._cell(state) != "G"
    )
    q = {(state, action): 0.0 for state in states for action in actions}
    counts = {(state, action): 0 for state in states for action in actions}

    for episode in range(episodes):
        if c_end is None or episodes == 1:
            c_e = c
        else:
            t = episode / (episodes - 1)
            if c_mode == "linear":
                c_e = c + (c_end - c) * t
            elif c_mode == "cosine":
                c_e = c_end + (
                    c - c_end
                ) * (1.0 + math.cos(math.pi * t)) / 2.0
            else:
                c_e = c * (c_end / c) ** t
        if not math.isfinite(c_e):
            raise ValueError("exploration coefficient must be finite")
        state = env.reset()
        for _ in range(max_steps):
            action = None
            for candidate in actions:
                if counts[(state, candidate)] == 0:
                    action = candidate
                    break
            if action is None:
                total = sum(counts[(state, a)] for a in actions)
                log_total = math.log(total)
                best_action = None
                best_score = None
                for candidate in actions:
                    score = q[(state, candidate)] + c_e * math.sqrt(
                        log_total / counts[(state, candidate)]
                    )
                    if not math.isfinite(score):
                        raise ValueError("UCB score must be finite")
                    if best_action is None or score > best_score:
                        best_action = candidate
                        best_score = score
                action = best_action
            next_state, reward, done = env.step(action)
            if not math.isfinite(reward):
                raise ValueError("reward must be finite")
            key = (state, action)
            counts[key] += 1
            if done:
                target = reward
            else:
                target = reward + gamma * max(
                    q[(next_state, a)] for a in actions
                )
            new_value = q[key] + alpha * (target - q[key])
            if not math.isfinite(new_value):
                raise ValueError("Q value must remain finite")
            q[key] = new_value
            if done:
                break
            state = next_state

    return {"q": q, "counts": counts}


def q_learning_trace(
    env,
    episodes=500,
    alpha=0.5,
    gamma=0.9,
    epsilon=0.1,
    seed=0,
    max_steps=1000,
) -> dict:
    """Q-learning，返回可逐步复现的轨迹与最终 Q 表。

    参数合法域及 TypeError/ValueError 分类与 q_learning 同名参数
    一致；max_steps 须为正 int（非 bool）。Q 覆盖从 S 可达的非 G
    格与 U/R/D/L 的全部组合，键序为坐标升序 × URDL，初始 0.0。
    每回合 reset，单回合最多 max_steps 步；全部随机性来自一个
    random.Random(seed)，每步先取一次 random()，探索时再取一次
    randrange(4) 按 URDL 选动作，否则取 URDL 序首个最大 Q 动作；
    更新规则与 q_learning 相同，Q 非有限时抛 ValueError。

    每步追加 [r, c, action, next_r, next_c, reward, done]：r、c 为
    调用 step 前状态，next_r、next_c 为返回状态，坐标与 reward 为
    int，action 为 str，done 为 bool。到达 G 立即结束；到达步限
    截断时保留末步且末步 done=False。

    返回键依次为 q、episodes：q 为 {((row, col), action): float}，
    episodes 为回合列表，每回合为步记录列表。相同参数与 seed
    逐值一致。
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
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)):
        raise TypeError("epsilon must be an int or float")
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if not _is_finite_number(alpha) or alpha <= 0 or alpha > 1:
        raise ValueError("alpha must be finite and in (0, 1]")
    if not _is_finite_number(gamma) or gamma < 0 or gamma >= 1:
        raise ValueError("gamma must be finite and in [0, 1)")
    if not _is_finite_number(epsilon) or epsilon < 0 or epsilon > 1:
        raise ValueError("epsilon must be finite and in [0, 1]")

    actions = tuple(_ACTIONS)  # U, R, D, L
    q = {
        (state, action): 0.0
        for state in sorted(_reachable_cells(env))
        if env._cell(state) != "G"
        for action in actions
    }
    rng = random.Random(seed)

    episode_records = []
    for _ in range(episodes):
        state = env.reset()
        trace = []
        for _ in range(max_steps):
            r, c = state
            if rng.random() < epsilon:
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
            new_value = q[key] + alpha * (target - q[key])
            if not math.isfinite(new_value):
                raise ValueError("Q value must remain finite")
            q[key] = new_value
            next_r, next_c = next_state
            trace.append(
                [r, c, action, next_r, next_c, reward, done]
            )
            if done:
                break
            state = next_state
        episode_records.append(trace)
    return {"q": q, "episodes": episode_records}


def q_learning_trace_bytes(env, data) -> bytes:
    """将 q_learning_trace 的结果严格校验并规范序列化为单行 JSON 字节。

    env 须为 GridWorld、data 须为 dict，否则抛 TypeError；其余任何
    违约均抛 ValueError。data 键序须恰为 q、episodes。q 须为恰覆盖
    从 S 可达的非 G 格与 U/R/D/L 全部组合的 dict，键为
    ((row, col), action)，值须为有限 float；输入键序不限。episodes
    须为非空 list，每回合为非空步 list；每步恰为
    [r, c, action, next_r, next_c, reward, done]，五个数值字段为非
    bool 的 int，action 为 U、R、D、L 之一，done 为 bool。逐回合
    首步 (r, c) 须为 S，其余各步 (r, c) 须等于上一步的
    (next_r, next_c)；每步须严格等于
    env.transition((r, c), action) 的结果；done 为 True 仅允许出现
    在末步。不修改输入。

    输出 JSON 键序为 q、episodes：q 转为按坐标升序 × URDL 的
    [r, c, action, value] 列表，episodes 保序原样嵌入。返回
    (json.dumps(out, ensure_ascii=True, allow_nan=False,
    separators=(",", ":")) + "\\n").encode("utf-8")：无额外空白，
    末尾恰一个 LF，保留 -0.0；相同输入逐字节一致。仅用标准库，
    不引入命令行入口。
    """
    if not isinstance(env, GridWorld):
        raise TypeError("env must be a GridWorld")
    if not isinstance(data, dict):
        raise TypeError("data must be a dict")
    if list(data) != ["q", "episodes"]:
        raise ValueError("data must have exactly the keys q, episodes")

    actions = tuple(_ACTIONS)  # U, R, D, L
    states = [
        state
        for state in sorted(_reachable_cells(env))
        if env._cell(state) != "G"
    ]
    expected_keys = {
        (state, action) for state in states for action in actions
    }

    q = data["q"]
    if not isinstance(q, dict):
        raise ValueError("q must be a dict")
    for key, value in q.items():
        if (
            not isinstance(key, tuple)
            or len(key) != 2
            or not isinstance(key[0], tuple)
            or len(key[0]) != 2
            or any(
                isinstance(v, bool) or not isinstance(v, int)
                for v in key[0]
            )
            or not isinstance(key[1], str)
            or key[1] not in _ACTIONS
        ):
            raise ValueError(
                "q keys must be ((row, col), action) tuples with action"
                " one of U, R, D, L"
            )
        if not isinstance(value, float) or not math.isfinite(value):
            raise ValueError("q values must be finite floats")
    if set(q) != expected_keys:
        raise ValueError(
            "q must cover exactly the reachable non-G cells times"
            " U, R, D, L"
        )

    episodes = data["episodes"]
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("episodes must be a non-empty list")
    start = env._start
    for ep_index, episode in enumerate(episodes):
        if not isinstance(episode, list) or not episode:
            raise ValueError(
                f"episodes[{ep_index}] must be a non-empty list"
            )
        for step_index, step in enumerate(episode):
            where = f"episodes[{ep_index}][{step_index}]"
            if not isinstance(step, list) or len(step) != 7:
                raise ValueError(
                    f"{where} must be a list of exactly 7 items"
                )
            r, c, action, next_r, next_c, reward, done = step
            for field in (r, c, next_r, next_c, reward):
                if isinstance(field, bool) or not isinstance(field, int):
                    raise ValueError(
                        f"{where} numeric fields must be non-bool ints"
                    )
            if not isinstance(action, str) or action not in _ACTIONS:
                raise ValueError(
                    f"{where} action must be one of U, R, D, L"
                )
            if not isinstance(done, bool):
                raise ValueError(f"{where} done must be a bool")
            expected_state = start if step_index == 0 else prev_next
            if (r, c) != expected_state:
                raise ValueError(
                    f"{where} (r, c) must be {expected_state},"
                    f" got {(r, c)}"
                )
            outcome = env.transition((r, c), action)
            if outcome != ((next_r, next_c), reward, done):
                raise ValueError(
                    f"{where} does not match env.transition: expected"
                    f" {outcome}, got {((next_r, next_c), reward, done)}"
                )
            if done and step_index != len(episode) - 1:
                raise ValueError(
                    f"{where} done may be True only at the last step"
                )
            prev_next = (next_r, next_c)

    out = {
        "q": [
            [state[0], state[1], action, q[(state, action)]]
            for state in states
            for action in actions
        ],
        "episodes": episodes,
    }
    return (
        json.dumps(
            out,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _load_q_learning_trace_payload(env, payload):
    """严格解析 q_learning_trace_bytes 产物。

    返回 (states, actions, episodes, recorded_q)：states 为坐标升序的
    可达非 G 格列表，actions 为 (U, R, D, L)，episodes 为原轨迹，
    recorded_q 为 {((row, col), action): float}。

    payload 须恰为 bytes（bytearray、memoryview 等均拒绝），且逐字节
    符合 q_learning_trace_bytes 的规范产物；空字节、非 UTF-8、BOM、
    JSON 语法错误或尾随内容、重复对象键、NaN/Infinity/-Infinity、根值
    非 dict、键序非 q、episodes、结构或取值违约、重编码后字节不一致
    均抛 ValueError。
    """
    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")

    if not payload:
        raise ValueError("payload must not be empty")
    if payload.startswith(_UTF8_BOM):
        raise ValueError("payload must not start with a BOM")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("payload must be valid UTF-8") from exc
    if not text.endswith("\n"):
        raise ValueError("payload must end with exactly one LF")
    body = text[:-1]

    def reject_constant(value):
        raise ValueError(f"invalid JSON constant: {value}")

    def reject_duplicate_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate object key: {key!r}")
            result[key] = value
        return result

    decoder = json.JSONDecoder(
        object_pairs_hook=reject_duplicate_keys,
        parse_constant=reject_constant,
    )
    try:
        data, end = decoder.raw_decode(body)
    except RecursionError as exc:
        raise ValueError("payload nesting too deep") from exc
    except json.JSONDecodeError as exc:
        raise ValueError("payload must be valid JSON") from exc
    if end != len(body):
        raise ValueError("payload must not contain trailing content")
    if not isinstance(data, dict) or list(data) != ["q", "episodes"]:
        raise ValueError("payload root must be a dict with keys q, episodes")

    actions = tuple(_ACTIONS)  # U, R, D, L
    states = [
        state
        for state in sorted(_reachable_cells(env))
        if env._cell(state) != "G"
    ]
    expected_keys = {
        (state, action) for state in states for action in actions
    }

    q_rows = data["q"]
    if not isinstance(q_rows, list) or not q_rows:
        raise ValueError("q must be a non-empty list")
    recorded_q = {}
    for row_index, row in enumerate(q_rows):
        where = f"q[{row_index}]"
        if not isinstance(row, list) or len(row) != 4:
            raise ValueError(
                f"{where} must be a list of exactly 4 items"
            )
        r, c, action, value = row
        if isinstance(r, bool) or not isinstance(r, int):
            raise ValueError(f"{where} row must be a non-bool int")
        if isinstance(c, bool) or not isinstance(c, int):
            raise ValueError(f"{where} col must be a non-bool int")
        if not isinstance(action, str) or action not in _ACTIONS:
            raise ValueError(
                f"{where} action must be one of U, R, D, L"
            )
        if isinstance(value, bool) or not isinstance(value, float):
            raise ValueError(f"{where} value must be a float")
        if not math.isfinite(value):
            raise ValueError(f"{where} value must be finite")
        recorded_q[((r, c), action)] = value

    episodes = data["episodes"]

    # 将 q 行还原为元组键字典后，交由 q_learning_trace_bytes 执行完整
    # data 校验（覆盖域、episodes 结构、状态链、env.transition 一致性、
    # done 仅可末步等）并生成规范字节；与输入逐字节不同即违约。
    canonical = q_learning_trace_bytes(
        env, {"q": recorded_q, "episodes": episodes}
    )
    if canonical != payload:
        raise ValueError(
            "payload must be canonical q_learning_trace_bytes output"
        )
    return states, actions, episodes, recorded_q


def q_learning_trace_replay(env, payload, alpha=0.5, gamma=0.9) -> dict:
    """重放 q_learning_trace_bytes 产物中的轨迹并核对最终 Q 表。

    env 须为 GridWorld、payload 须恰为 bytes（bytearray、memoryview 等
    均拒绝）、alpha/gamma 须为非 bool 的 int/float，否则抛 TypeError。
    alpha 须有限且属 (0, 1]、gamma 须有限且属 [0, 1)，否则抛
    ValueError。

    payload 须逐字节符合 q_learning_trace_bytes 的产物：空字节、严格
    UTF-8 解码失败、BOM、JSON 语法错误或尾随内容、重复对象键、
    NaN/Infinity/-Infinity、根值非 dict、键序非 q、episodes、结构或
    取值违约、重编码后字节不一致均抛 ValueError。将 q 行还原为以
    ((row, col), action) 为键的 dict 后，q 与 episodes 须通过
    q_learning_trace_bytes 的完整 data 校验（覆盖域、步字段、状态链、
    env.transition 一致性、done 仅可末步等）。

    Q 同覆盖域置 0.0，严格按记录逐步重放，全程只用 env.transition
    查询而不修改 env；done 时目标为 reward，否则为
    reward + gamma * max_a Q[next, a]，再作
    Q += alpha * (目标 - Q)，新值非有限抛 ValueError。以 float.hex()
    逐项比较记录 Q 与重放 Q。

    返回键序恰为 matched、differences：matched 仅在无任何差异时为
    True；differences 按 Q 序（坐标升序 × URDL）列出每项差异
    [r, c, action, recorded, replayed]，末两项为 float。仅用标准库，
    不引入命令行入口。
    """
    if not isinstance(env, GridWorld):
        raise TypeError("env must be a GridWorld")
    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise TypeError("alpha must be an int or float")
    if isinstance(gamma, bool) or not isinstance(gamma, (int, float)):
        raise TypeError("gamma must be an int or float")
    if not _is_finite_number(alpha) or alpha <= 0 or alpha > 1:
        raise ValueError("alpha must be finite and in (0, 1]")
    if not _is_finite_number(gamma) or gamma < 0 or gamma >= 1:
        raise ValueError("gamma must be finite and in [0, 1)")

    states, actions, episodes, recorded_q = (
        _load_q_learning_trace_payload(env, payload)
    )
    expected_keys = {
        (state, action) for state in states for action in actions
    }

    q = {key: 0.0 for key in expected_keys}
    for episode in episodes:
        for r, c, action, next_r, next_c, reward, done in episode:
            key = ((r, c), action)
            if done:
                target = reward
            else:
                target = reward + gamma * max(
                    q[((next_r, next_c), a)] for a in actions
                )
            new_value = q[key] + alpha * (target - q[key])
            if not math.isfinite(new_value):
                raise ValueError("Q value must remain finite")
            q[key] = new_value

    differences = []
    for state in states:
        for action in actions:
            key = (state, action)
            recorded = recorded_q[key]
            replayed = q[key]
            if recorded.hex() != replayed.hex():
                differences.append(
                    [state[0], state[1], action, recorded, replayed]
                )
    return {"matched": not differences, "differences": differences}


def q_learning_trace_convergence(
    env,
    payload,
    alpha=0.5,
    gamma=0.9,
    window=20,
    tolerance=0.01,
    patience=3,
) -> dict:
    """按回合重放 q_learning_trace_bytes 轨迹并判定 Q 表是否收敛。

    env、payload、alpha、gamma 的合法域及 TypeError/ValueError 分类
    与 q_learning_trace_replay 一致；payload 须为规范产物且 q 覆盖域
    与 env 匹配，否则抛 ValueError。window、patience 须为非 bool 的
    正 int，否则类型违约抛 TypeError、非正抛 ValueError。tolerance
    须为非 bool 且可转为有限 float 的非负 int/float，否则类型违约抛
    TypeError，其余（NaN/Infinity、负数、超出 float 范围的大整数等）
    抛 ValueError。

    Q 同覆盖域置 0.0，严格按记录逐回合重放，更新式与
    q_learning_trace_replay 相同，全程只用 env.transition 查询而不
    修改 env；新 Q 非有限抛 ValueError。每回合记录
    delta = max(各次更新 |new-old|)，保回合序。

    以每个可能起点生成完整滑窗 [start, end, max_delta, passed]：
    start、end 为 1 基 int，end=start+window-1，max_delta 为窗内
    delta 的最大值，passed 等价于 max_delta <= float(tolerance)；
    window 大于回合数时合法，此时无任何窗。episode 取首个连续
    patience 个 passed 窗末窗的 end，不存在则为 None。

    返回键序恰为 converged、episode、deltas、windows：
    converged = (episode is not None)，deltas 保回合序。仅用标准库，
    不引入命令行入口。
    """
    if not isinstance(env, GridWorld):
        raise TypeError("env must be a GridWorld")
    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise TypeError("alpha must be an int or float")
    if isinstance(gamma, bool) or not isinstance(gamma, (int, float)):
        raise TypeError("gamma must be an int or float")
    if isinstance(window, bool) or not isinstance(window, int):
        raise TypeError("window must be an int")
    if isinstance(patience, bool) or not isinstance(patience, int):
        raise TypeError("patience must be an int")
    if isinstance(tolerance, bool) or not isinstance(
        tolerance, (int, float)
    ):
        raise TypeError("tolerance must be an int or float")
    if not _is_finite_number(alpha) or alpha <= 0 or alpha > 1:
        raise ValueError("alpha must be finite and in (0, 1]")
    if not _is_finite_number(gamma) or gamma < 0 or gamma >= 1:
        raise ValueError("gamma must be finite and in [0, 1)")
    if window <= 0:
        raise ValueError("window must be positive")
    if patience <= 0:
        raise ValueError("patience must be positive")
    try:
        tol = float(tolerance)
    except OverflowError as exc:
        raise ValueError(
            "tolerance must be convertible to a finite float"
        ) from exc
    if not math.isfinite(tol) or tol < 0:
        raise ValueError(
            "tolerance must be a finite non-negative number"
        )

    states, actions, episodes, _recorded_q = (
        _load_q_learning_trace_payload(env, payload)
    )

    q = {
        (state, action): 0.0
        for state in states
        for action in actions
    }
    deltas = []
    for episode in episodes:
        episode_delta = 0.0
        for r, c, action, next_r, next_c, reward, done in episode:
            key = ((r, c), action)
            if done:
                target = reward
            else:
                target = reward + gamma * max(
                    q[((next_r, next_c), a)] for a in actions
                )
            old_value = q[key]
            new_value = old_value + alpha * (target - old_value)
            if not math.isfinite(new_value):
                raise ValueError("Q value must remain finite")
            q[key] = new_value
            change = abs(new_value - old_value)
            if change > episode_delta:
                episode_delta = change
        deltas.append(episode_delta)

    windows = []
    count = len(deltas)
    for start in range(1, count - window + 2):
        end = start + window - 1
        max_delta = max(deltas[start - 1:end])
        windows.append(
            [start, end, max_delta, max_delta <= tol]
        )

    episode = None
    streak = 0
    for entry in windows:
        if entry[3]:
            streak += 1
            if streak >= patience:
                episode = entry[1]
                break
        else:
            streak = 0

    return {
        "converged": episode is not None,
        "episode": episode,
        "deltas": deltas,
        "windows": windows,
    }


def q_trace_stability(env, runs, tol=0.01, ratio=1.0, patience=2) -> dict:
    """跨 seed 诊断 q_learning_trace_bytes 轨迹的稳定性与收敛。

    env 须为 GridWorld，否则抛 TypeError。runs 须为至少 2 项的 list，
    每项恰为 [seed, payloads]：seed 为互异的非 bool int，payloads 为
    至少 2 项的 bytes list，各 run 的 payloads 等长；runs 非 list、
    项非 list、seed/payload/patience 等类型违约抛 TypeError，空、项
    数不足、seed 重复、payloads 不等长、patience 非正抛 ValueError。
    tol、ratio 须为非 bool 的有限 int/float（tol≥0、ratio∈[0, 1]），
    类型违约抛 TypeError，NaN/Infinity、越界或 float 转换溢出抛
    ValueError。

    每个 payload 须经 q_learning_trace_replay 重放且 matched 为真，
    否则抛 ValueError（payload 非规范产物同样抛 ValueError）。全程
    不修改 env 与任何入参。

    自检查点 2 起逐项比较相邻检查点（index 为 1 基）：D 为全部 seed、
    全部 Q 键在相邻检查点间差的绝对值之最大；A 为状态跨 seed 贪心
    动作一致的比例——每状态以 URDL 序取首个最大 Q 动作为贪心动作，
    所有 seed 一致才计入，A=一致状态数/状态数。每 checkpoint 为
    [index, D, A, passed]，passed 等价于 D≤tol 且 A≥ratio。checkpoint
    取首次连续 patience 个 passed 项的 index，否则为 None。

    返回键序恰为 converged、checkpoint、checkpoints；converged 等价
    于 checkpoint 非 None。仅用标准库，不引入命令行入口；其余 API
    与 run/value 行为不变。
    """
    if not isinstance(env, GridWorld):
        raise TypeError("env must be a GridWorld")
    if not isinstance(runs, list):
        raise TypeError("runs must be a list")
    if isinstance(tol, bool) or not isinstance(tol, (int, float)):
        raise TypeError("tol must be an int or float")
    if isinstance(ratio, bool) or not isinstance(ratio, (int, float)):
        raise TypeError("ratio must be an int or float")
    if isinstance(patience, bool) or not isinstance(patience, int):
        raise TypeError("patience must be an int")
    try:
        tol_value = float(tol)
    except OverflowError as exc:
        raise ValueError(
            "tol must be convertible to a finite float"
        ) from exc
    try:
        ratio_value = float(ratio)
    except OverflowError as exc:
        raise ValueError(
            "ratio must be convertible to a finite float"
        ) from exc
    if not math.isfinite(tol_value) or tol_value < 0:
        raise ValueError("tol must be a finite non-negative number")
    if (
        not math.isfinite(ratio_value)
        or ratio_value < 0
        or ratio_value > 1
    ):
        raise ValueError("ratio must be finite and in [0, 1]")
    if patience <= 0:
        raise ValueError("patience must be positive")
    if len(runs) < 2:
        raise ValueError("runs must contain at least 2 items")

    seeds = []
    run_payloads = []
    for run_index, item in enumerate(runs):
        if not isinstance(item, list):
            raise TypeError(f"runs[{run_index}] must be a list")
        if len(item) != 2:
            raise ValueError(
                f"runs[{run_index}] must be exactly [seed, payloads]"
            )
        seed, payloads = item
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError(
                f"runs[{run_index}] seed must be a non-bool int"
            )
        if seed in seeds:
            raise ValueError(f"duplicate seed: {seed}")
        seeds.append(seed)
        if not isinstance(payloads, list):
            raise TypeError(
                f"runs[{run_index}] payloads must be a list"
            )
        if len(payloads) < 2:
            raise ValueError(
                f"runs[{run_index}] payloads must contain at least 2"
                " items"
            )
        for payload_index, payload in enumerate(payloads):
            if not isinstance(payload, bytes):
                raise TypeError(
                    f"runs[{run_index}] payloads[{payload_index}] must"
                    " be bytes"
                )
        run_payloads.append(payloads)

    checkpoint_count = len(run_payloads[0])
    for run_index, payloads in enumerate(run_payloads):
        if len(payloads) != checkpoint_count:
            raise ValueError(
                f"runs[{run_index}] payloads must have equal length"
            )

    actions = tuple(_ACTIONS)  # U, R, D, L
    states = [
        state
        for state in sorted(_reachable_cells(env))
        if env._cell(state) != "G"
    ]

    # matched 为真时记录 Q 与重放 Q 逐项 float.hex 一致，直接以记录 Q
    # 作为各检查点 Q 表；每次重放不修改 env 与 payload。
    tables = []
    for run_index, payloads in enumerate(run_payloads):
        series = []
        for payload_index, payload in enumerate(payloads):
            result = q_learning_trace_replay(env, payload)
            if not result["matched"]:
                raise ValueError(
                    f"runs[{run_index}] payloads[{payload_index}] replay"
                    " does not match recorded Q"
                )
            _states, _actions, _episodes, recorded_q = (
                _load_q_learning_trace_payload(env, payload)
            )
            series.append(recorded_q)
        tables.append(series)

    checkpoints = []
    for index in range(2, checkpoint_count + 1):
        current = [series[index - 1] for series in tables]
        previous = [series[index - 2] for series in tables]
        d_max = 0.0
        for cur_q, prev_q in zip(current, previous):
            for state in states:
                for action in actions:
                    key = (state, action)
                    difference = abs(cur_q[key] - prev_q[key])
                    if difference > d_max:
                        d_max = difference
        consistent = 0
        for state in states:
            greedy = [
                max(
                    actions,
                    key=lambda action, q_table=cur_q: q_table[
                        (state, action)
                    ],
                )
                for cur_q in current
            ]
            first = greedy[0]
            if all(action == first for action in greedy[1:]):
                consistent += 1
        a_ratio = consistent / len(states)
        passed = d_max <= tol_value and a_ratio >= ratio_value
        checkpoints.append([index, d_max, a_ratio, passed])

    checkpoint = None
    streak = 0
    for entry in checkpoints:
        if entry[3]:
            streak += 1
            if streak >= patience:
                checkpoint = entry[0]
                break
        else:
            streak = 0

    return {
        "converged": checkpoint is not None,
        "checkpoint": checkpoint,
        "checkpoints": checkpoints,
    }


def dueling_q_learning(
    env,
    episodes=500,
    alpha=0.5,
    gamma=0.9,
    epsilon=0.1,
    seed=0,
    max_steps=1000,
) -> dict:
    """Dueling Q-learning，返回 {((row, col), action): float}。

    V 覆盖从 S 可达的非 G 格，A 覆盖其与 U/R/D/L 的全部组合，
    初始均为 0.0；Q(s,a)=V(s)+A(s,a)-ΣA(s,·)/4。每回合 reset，
    单回合最多 max_steps 步；全部随机性来自一个
    random.Random(seed)。

    每步先调一次 random()：其值 < epsilon 时按 URDL 调
    randrange(4) 取探索动作，否则取 Q 最大且 URDL 中首个的
    动作。step 后 done 时目标为 r，否则为
    r+gamma*max_a Q(s2,a)；td=目标-Q(s,a)，随后
    V(s)+=alpha*td 且 A(s,a)+=alpha*td。新值非有限抛
    ValueError。done 即停；到达步限且未终止的末步仍照常更新。
    返回 Q 按状态坐标升序、动作 URDL 键序，值均为有限 float。
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
    if isinstance(max_steps, bool) or not isinstance(max_steps, int):
        raise TypeError("max_steps must be an int")
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if not _is_finite_number(alpha) or alpha <= 0 or alpha > 1:
        raise ValueError("alpha must be finite and in (0, 1]")
    if not _is_finite_number(gamma) or gamma < 0 or gamma >= 1:
        raise ValueError("gamma must be finite and in [0, 1)")
    if not _is_finite_number(epsilon) or epsilon < 0 or epsilon > 1:
        raise ValueError("epsilon must be finite and in [0, 1]")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")

    actions = tuple(_ACTIONS)  # U, R, D, L
    states = sorted(
        state
        for state in _reachable_cells(env)
        if env._cell(state) != "G"
    )
    v = {state: 0.0 for state in states}
    adv = {
        (state, action): 0.0 for state in states for action in actions
    }
    rng = random.Random(seed)

    def q_value(state, action):
        return v[state] + adv[(state, action)] - sum(
            adv[(state, a)] for a in actions
        ) / 4.0

    for _ in range(episodes):
        state = env.reset()
        for _ in range(max_steps):
            if rng.random() < epsilon:
                action = actions[rng.randrange(4)]
            else:
                action = max(actions, key=lambda a: q_value(state, a))
            next_state, reward, done = env.step(action)
            if done:
                target = reward
            else:
                target = reward + gamma * max(
                    q_value(next_state, a) for a in actions
                )
            td = target - q_value(state, action)
            new_v = v[state] + alpha * td
            new_a = adv[(state, action)] + alpha * td
            if not math.isfinite(new_v) or not math.isfinite(new_a):
                raise ValueError("V and A values must remain finite")
            v[state] = new_v
            adv[(state, action)] = new_a
            if done:
                break
            state = next_state

    q = {}
    for state in states:
        for action in actions:
            value = float(q_value(state, action))
            if not math.isfinite(value):
                raise ValueError("Q value must be finite")
            q[(state, action)] = value
    return q


def dyna_q(
    env,
    episodes=500,
    alpha=0.5,
    gamma=0.9,
    epsilon=0.1,
    planning_steps=5,
    seed=0,
    max_steps=1000,
) -> dict:
    """Dyna-Q，返回 {((row, col), action): float}。

    Q 覆盖从 S 可达的非 G 格与 U/R/D/L 的全部组合，初始为 0.0。
    每回合 reset，done 即停；单回合最多 max_steps 步（末步照常执行）。
    每个真实步先且仅调用一次 rng.random() 决定是否探索，探索时再按
    URDL 调用 rng.randrange(4)，否则取 URDL 中首个最大 Q 动作。
    真实转移按首次出现写入模型（重复键覆盖但保持插入序），模型非空时
    每步按插入序用 rng.randrange(len(model)) 采样 planning_steps 次，
    以同式更新 Q；规划过程不调用 rng.random()。
    全部随机性来自一个 random.Random(seed)。
    """
    if not isinstance(env, GridWorld):
        raise TypeError("env must be a GridWorld")
    if isinstance(episodes, bool) or not isinstance(episodes, int):
        raise TypeError("episodes must be an int")
    if isinstance(planning_steps, bool) or not isinstance(planning_steps, int):
        raise TypeError("planning_steps must be an int")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an int")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int):
        raise TypeError("max_steps must be an int")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise TypeError("alpha must be an int or float")
    if isinstance(gamma, bool) or not isinstance(gamma, (int, float)):
        raise TypeError("gamma must be an int or float")
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)):
        raise TypeError("epsilon must be an int or float")
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if planning_steps <= 0:
        raise ValueError("planning_steps must be positive")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if not _is_finite_number(alpha) or alpha <= 0 or alpha > 1:
        raise ValueError("alpha must be finite and in (0, 1]")
    if not _is_finite_number(gamma) or gamma < 0 or gamma >= 1:
        raise ValueError("gamma must be finite and in [0, 1)")
    if not _is_finite_number(epsilon) or epsilon < 0 or epsilon > 1:
        raise ValueError("epsilon must be finite and in [0, 1]")

    actions = tuple(_ACTIONS)  # U, R, D, L
    states = sorted(
        state
        for state in _reachable_cells(env)
        if env._cell(state) != "G"
    )
    q = {(state, action): 0.0 for state in states for action in actions}
    model = {}
    rng = random.Random(seed)

    for _ in range(episodes):
        state = env.reset()
        for _ in range(max_steps):
            if rng.random() < epsilon:
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
            if not math.isfinite(q[key]):
                raise ValueError("Q value must remain finite")
            # 首次出现插入，重复出现覆盖且保持原插入序。
            model[key] = (next_state, reward, done)
            ordered_keys = list(model)
            for _ in range(planning_steps):
                plan_key = ordered_keys[rng.randrange(len(ordered_keys))]
                plan_next, plan_reward, plan_done = model[plan_key]
                if plan_done:
                    plan_target = plan_reward
                else:
                    plan_target = plan_reward + gamma * max(
                        q[(plan_next, a)] for a in actions
                    )
                q[plan_key] += alpha * (plan_target - q[plan_key])
                if not math.isfinite(q[plan_key]):
                    raise ValueError("Q value must remain finite")
            if done:
                break
            state = next_state
    return q


def replay_q_learning(
    env,
    episodes=500,
    alpha=0.5,
    gamma=0.9,
    epsilon=0.1,
    capacity=1000,
    replay_steps=4,
    seed=0,
    max_steps=1000,
) -> dict:
    """经验回放 Q 学习，返回 {"q": Q, "buffer": 缓冲副本, "updates": int}。

    Q 覆盖从 S 可达的非 G 格与 U/R/D/L 的全部组合，初始为 0.0。
    每回合 reset，done 即停；单回合最多 max_steps 步（截断末步照常
    入缓冲并回放）。每个真实步先且仅调用一次 rng.random() 决定是否
    探索，探索时再按 URDL 调用 rng.randrange(4)，否则取 URDL 中首个
    最大 Q 动作。真实步不直接更新 Q：步后若缓冲已满（长度达到
    capacity）先删头，再追加 [r, c, action, nr, nc, reward, done]
    （行型为 [int, int, str, int, int, int, bool]），继而恰回放
    replay_steps 次：每次 rng.randrange(len(buffer)) 抽一行，目标在
    done 时为 reward，否则为 reward + gamma * maxQ(next)，按当前 Q
    作 Q += alpha * (目标 - Q)；目标或新 Q 非有限即抛 ValueError。
    全部随机性来自一个 random.Random(seed)。
    """
    if not isinstance(env, GridWorld):
        raise TypeError("env must be a GridWorld")
    if isinstance(episodes, bool) or not isinstance(episodes, int):
        raise TypeError("episodes must be an int")
    if isinstance(capacity, bool) or not isinstance(capacity, int):
        raise TypeError("capacity must be an int")
    if isinstance(replay_steps, bool) or not isinstance(replay_steps, int):
        raise TypeError("replay_steps must be an int")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an int")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int):
        raise TypeError("max_steps must be an int")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise TypeError("alpha must be an int or float")
    if isinstance(gamma, bool) or not isinstance(gamma, (int, float)):
        raise TypeError("gamma must be an int or float")
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)):
        raise TypeError("epsilon must be an int or float")
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if capacity <= 0:
        raise ValueError("capacity must be positive")
    if replay_steps <= 0:
        raise ValueError("replay_steps must be positive")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if not _is_finite_number(alpha) or alpha <= 0 or alpha > 1:
        raise ValueError("alpha must be finite and in (0, 1]")
    if not _is_finite_number(gamma) or gamma < 0 or gamma >= 1:
        raise ValueError("gamma must be finite and in [0, 1)")
    if not _is_finite_number(epsilon) or epsilon < 0 or epsilon > 1:
        raise ValueError("epsilon must be finite and in [0, 1]")

    actions = tuple(_ACTIONS)  # U, R, D, L
    states = sorted(
        state
        for state in _reachable_cells(env)
        if env._cell(state) != "G"
    )
    q = {(state, action): 0.0 for state in states for action in actions}
    buffer = []
    updates = 0
    rng = random.Random(seed)

    for _ in range(episodes):
        state = env.reset()
        for _ in range(max_steps):
            if rng.random() < epsilon:
                action = actions[rng.randrange(4)]
            else:
                action = max(actions, key=lambda a: q[(state, a)])
            next_state, reward, done = env.step(action)
            if len(buffer) >= capacity:
                del buffer[0]
            buffer.append([
                state[0],
                state[1],
                action,
                next_state[0],
                next_state[1],
                reward,
                done,
            ])
            for _ in range(replay_steps):
                r, c, act, nr, nc, rew, dn = buffer[
                    rng.randrange(len(buffer))
                ]
                key = ((r, c), act)
                if dn:
                    target = rew
                else:
                    target = rew + gamma * max(
                        q[((nr, nc), a)] for a in actions
                    )
                new_q = q[key] + alpha * (target - q[key])
                if not _is_finite_number(target) or not _is_finite_number(
                    new_q
                ):
                    raise ValueError("Q value must remain finite")
                q[key] = new_q
                updates += 1
            if done:
                break
            state = next_state
    return {"q": q, "buffer": [list(row) for row in buffer], "updates": updates}


def prioritized_replay_q(
    env,
    episodes=500,
    alpha=0.5,
    gamma=0.9,
    epsilon=0.1,
    beta=0.4,
    seed=0,
) -> dict:
    """按比例的优先经验回放 Q 学习，返回
    {"q": Q, "buffer": 缓冲副本, "priorities": float 列表, "updates": int}。

    契约同 replay_q_learning（容量固定 1000、每步回放 4 次、单回合至多
    1000 步，末步照常入缓冲并回放），差异如下。缓冲并行维护同序优先级
    列表：追加时若缓冲已满（长度达到 1000），同步删首行与首优先级；
    新项优先级取现存优先级最大值，无旧项取 1.0。每次回放重算
    w_i = priority_i ** 0.6，以一次 rng.random() * sum(w) 取样，累计值
    首个严格大于样本者命中，均不大于则取末项。P_i = w_i / sum(w)，
    重要度 I_i = (N * P_i) ** (-beta) 后除以全部 I 的最大值归一，仅以
    命中项的归一 I 作 Q += alpha * I * delta，其中
    delta = reward + (done 时 0，否则 gamma * maxQ(next)) - Q（按当前
    Q 计算），随后命中项优先级置为 abs(delta) + 1e-6。任何运算溢出或
    非有限结果均抛 ValueError。全部随机性来自一个 random.Random(seed)。
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
    if isinstance(beta, bool) or not isinstance(beta, (int, float)):
        raise TypeError("beta must be an int or float")
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if not _is_finite_number(alpha) or alpha <= 0 or alpha > 1:
        raise ValueError("alpha must be finite and in (0, 1]")
    if not _is_finite_number(gamma) or gamma < 0 or gamma >= 1:
        raise ValueError("gamma must be finite and in [0, 1)")
    if not _is_finite_number(epsilon) or epsilon < 0 or epsilon > 1:
        raise ValueError("epsilon must be finite and in [0, 1]")
    try:
        beta = float(beta)
    except OverflowError:
        raise ValueError("beta must convert to a finite float")
    if not math.isfinite(beta):
        raise ValueError("beta must be finite")
    if beta < 0.0 or beta > 1.0:
        raise ValueError("beta must be in [0, 1]")

    actions = tuple(_ACTIONS)  # U, R, D, L
    states = sorted(
        state
        for state in _reachable_cells(env)
        if env._cell(state) != "G"
    )
    q = {(state, action): 0.0 for state in states for action in actions}
    buffer = []
    priorities = []
    updates = 0
    rng = random.Random(seed)

    capacity = 1000
    replay_steps = 4
    max_steps = 1000

    for _ in range(episodes):
        state = env.reset()
        for _ in range(max_steps):
            if rng.random() < epsilon:
                action = actions[rng.randrange(4)]
            else:
                action = max(actions, key=lambda a: q[(state, a)])
            next_state, reward, done = env.step(action)
            if len(buffer) >= capacity:
                del buffer[0]
                del priorities[0]
            new_priority = max(priorities) if priorities else 1.0
            buffer.append([
                state[0],
                state[1],
                action,
                next_state[0],
                next_state[1],
                reward,
                done,
            ])
            priorities.append(float(new_priority))
            for _ in range(replay_steps):
                n_items = len(buffer)
                try:
                    weights = [p ** 0.6 for p in priorities]
                    total_w = sum(weights)
                    sample = rng.random() * total_w
                    cumulative = 0.0
                    index = n_items - 1
                    for j, w in enumerate(weights):
                        cumulative += w
                        if cumulative > sample:
                            index = j
                            break
                    probs = [w / total_w for w in weights]
                    raw_weights = [
                        (n_items * p) ** (-beta) for p in probs
                    ]
                    max_raw = max(raw_weights)
                    importance = raw_weights[index] / max_raw
                    r, c, act, nr, nc, rew, dn = buffer[index]
                    key = ((r, c), act)
                    if dn:
                        target = rew
                    else:
                        target = rew + gamma * max(
                            q[((nr, nc), a)] for a in actions
                        )
                    delta = target - q[key]
                    new_q = q[key] + alpha * importance * delta
                    new_priority = abs(delta) + 1e-6
                except OverflowError as exc:
                    raise ValueError(
                        "Replay arithmetic must remain finite"
                    ) from exc
                if not all(math.isfinite(w) for w in weights):
                    raise ValueError("Replay arithmetic must remain finite")
                if not all(math.isfinite(p) for p in probs):
                    raise ValueError("Replay arithmetic must remain finite")
                if not all(math.isfinite(iw) for iw in raw_weights):
                    raise ValueError("Replay arithmetic must remain finite")
                for value in (
                    total_w,
                    sample,
                    max_raw,
                    importance,
                    target,
                    delta,
                    new_q,
                    new_priority,
                ):
                    if not math.isfinite(value):
                        raise ValueError(
                            "Replay arithmetic must remain finite"
                        )
                q[key] = new_q
                priorities[index] = float(new_priority)
                updates += 1
            if done:
                break
            state = next_state
    return {
        "q": q,
        "buffer": [list(row) for row in buffer],
        "priorities": [float(p) for p in priorities],
        "updates": updates,
    }


def prioritized_sweeping(
    env,
    episodes=500,
    alpha=0.5,
    gamma=0.9,
    epsilon=0.1,
    planning_steps=5,
    theta=1e-6,
    seed=0,
    max_steps=1000,
) -> dict:
    """优先扫描（Prioritized Sweeping），返回 {((row, col), action): float}。

    Q 覆盖从 S 可达的非 G 格与 U/R/D/L 的全部组合，初始为 0.0。
    每回合 reset，done 即停；单回合最多 max_steps 步（末步照常执行）。
    每个真实步先且仅调用一次 rng.random() 决定是否探索，探索时再按
    URDL 调用 rng.randrange(4)，否则取 URDL 中首个最大 Q 动作。
    真实步不直接更新 Q，而是以 (state, action) 覆盖模型值
    (next_state, reward, done)（重复键覆盖但保持插入序），并计算优先级
    p = abs(target - Q)，p > theta 时把队列该键设为 p 与旧优先级的较大者。
    随后每步至多规划 planning_steps 次：队列空即停，否则弹出优先级最大项
    （平局取 Q 键序首项），按模型与当前 Q 重算目标并更新 Q，再按 Q 键序
    扫描已建模键，凡其 next_state 等于刚更新键的 state 者重算 p 并按同一
    规则入队。模型与队列跨回合保留，规划过程不消耗随机数。
    全部随机性来自一个 random.Random(seed)。
    """
    if not isinstance(env, GridWorld):
        raise TypeError("env must be a GridWorld")
    if isinstance(episodes, bool) or not isinstance(episodes, int):
        raise TypeError("episodes must be an int")
    if isinstance(planning_steps, bool) or not isinstance(planning_steps, int):
        raise TypeError("planning_steps must be an int")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an int")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int):
        raise TypeError("max_steps must be an int")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise TypeError("alpha must be an int or float")
    if isinstance(gamma, bool) or not isinstance(gamma, (int, float)):
        raise TypeError("gamma must be an int or float")
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)):
        raise TypeError("epsilon must be an int or float")
    if isinstance(theta, bool) or not isinstance(theta, (int, float)):
        raise TypeError("theta must be an int or float")
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if planning_steps <= 0:
        raise ValueError("planning_steps must be positive")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if not _is_finite_number(alpha) or alpha <= 0 or alpha > 1:
        raise ValueError("alpha must be finite and in (0, 1]")
    if not _is_finite_number(gamma) or gamma < 0 or gamma >= 1:
        raise ValueError("gamma must be finite and in [0, 1)")
    if not _is_finite_number(epsilon) or epsilon < 0 or epsilon > 1:
        raise ValueError("epsilon must be finite and in [0, 1]")
    if not _is_finite_number(theta) or theta <= 0:
        raise ValueError("theta must be finite and positive")

    actions = tuple(_ACTIONS)  # U, R, D, L
    states = sorted(
        state
        for state in _reachable_cells(env)
        if env._cell(state) != "G"
    )
    q = {(state, action): 0.0 for state in states for action in actions}
    model = {}
    queue = {}
    rng = random.Random(seed)

    def _target(key):
        plan_next, plan_reward, plan_done = model[key]
        if plan_done:
            return plan_reward
        return plan_reward + gamma * max(q[(plan_next, a)] for a in actions)

    def _offer(key):
        target = _target(key)
        if not math.isfinite(target):
            raise ValueError("target must remain finite")
        p = abs(target - q[key])
        if not math.isfinite(p):
            raise ValueError("priority must remain finite")
        if p > theta:
            old = queue.get(key)
            queue[key] = p if old is None else max(p, old)

    for _ in range(episodes):
        state = env.reset()
        for _ in range(max_steps):
            if rng.random() < epsilon:
                action = actions[rng.randrange(4)]
            else:
                action = max(actions, key=lambda a: q[(state, a)])
            next_state, reward, done = env.step(action)
            key = (state, action)
            # 真实步不直接更新 Q，仅覆盖模型值并按优先级入队。
            model[key] = (next_state, reward, done)
            _offer(key)
            for _ in range(planning_steps):
                if not queue:
                    break
                # 弹出优先级最大项，平局取 Q 键序首项。
                best_key = None
                best_p = None
                for q_key in q:
                    if q_key in queue:
                        p = queue[q_key]
                        if best_p is None or p > best_p:
                            best_key = q_key
                            best_p = p
                del queue[best_key]
                target = _target(best_key)
                if not math.isfinite(target):
                    raise ValueError("target must remain finite")
                new_q = q[best_key] + alpha * (target - q[best_key])
                if not math.isfinite(new_q):
                    raise ValueError("Q value must remain finite")
                q[best_key] = new_q
                # 按 Q 键序扫描已建模的前驱键，重算 p 并按同一规则入队。
                for q_key in q:
                    if q_key in model and model[q_key][0] == best_key[0]:
                        _offer(q_key)
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


def watkins_q_lambda(
    env,
    episodes=500,
    alpha=0.5,
    gamma=0.9,
    epsilon=0.1,
    lambda_=0.9,
    seed=0,
    max_steps=1000,
) -> dict:
    """Watkins Q(λ)，返回 {((row, col), action): float}。

    Q 覆盖从 S 可达的非 G 格与 U/R/D/L 的全部组合，初始为 0.0。
    每回合 reset，资格迹 E 清零，先选动作；单回合最多 max_steps 步，
    末步未终止也照常选择下一动作、自举并处理迹后截断。
    以 s2 上 URDL 首个最大 Q 动作为贪心动作自举；仅当实际选出的
    下一动作就是该贪心动作时迹按 gamma*lambda_ 衰减，否则全部清零。
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
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)):
        raise TypeError("epsilon must be an int or float")
    if isinstance(lambda_, bool) or not isinstance(lambda_, (int, float)):
        raise TypeError("lambda_ must be an int or float")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int):
        raise TypeError("max_steps must be an int")
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
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")

    actions = tuple(_ACTIONS)  # U, R, D, L
    states = sorted(
        state
        for state in _reachable_cells(env)
        if env._cell(state) != "G"
    )
    q = {(state, action): 0.0 for state in states for action in actions}
    rng = random.Random(seed)

    def choose_action(state):
        if rng.random() < epsilon:
            return actions[rng.randrange(4)]
        return max(actions, key=lambda a: q[(state, a)])

    for _ in range(episodes):
        state = env.reset()
        eligibility = {key: 0.0 for key in q}
        action = choose_action(state)
        for _ in range(max_steps):
            next_state, reward, done = env.step(action)
            eligibility[(state, action)] += 1
            if done:
                delta = reward - q[(state, action)]
                for key in q:
                    q[key] += alpha * delta * eligibility[key]
                    if not math.isfinite(q[key]):
                        raise ValueError("Q value must remain finite")
                break
            greedy = max(actions, key=lambda a: q[(next_state, a)])
            next_action = choose_action(next_state)
            delta = (
                reward
                + gamma * q[(next_state, greedy)]
                - q[(state, action)]
            )
            for key in q:
                q[key] += alpha * delta * eligibility[key]
                if not math.isfinite(q[key]):
                    raise ValueError("Q value must remain finite")
            if next_action == greedy:
                for key in q:
                    eligibility[key] *= gamma * lambda_
            else:
                for key in q:
                    eligibility[key] = 0.0
            state, action = next_state, next_action
    return q


def replacing_sarsa_lambda(
    env,
    episodes=500,
    alpha=0.5,
    gamma=0.9,
    epsilon=0.1,
    lambda_=0.9,
    seed=0,
    max_steps=1000,
) -> dict:
    """替换迹 SARSA(λ)，返回 {((row, col), action): float}。

    Q 覆盖从 S 可达的非 G 格与 U/R/D/L 的全部组合，初始为 0.0。
    每回合 reset，资格迹 E 按 Q 键序置 0.0，先选动作；单回合最多
    max_steps 步，末步未终止也照常选择下一动作、自举并更新后截断。
    每步 step 得 (s2, r, done)：done 时 delta=r-Q[s,a] 且不再选
    动作，否则立即按 ε 贪心选 a2 并令
    delta=r+gamma*Q[s2,a2]-Q[s,a]。随后采用替换迹：按 URDL 先令
    当前状态其他动作 E[s,b]=0.0，再令 E[s,a]=1.0；再按 Q 键序作
    Q[k]+=alpha*delta*E[k]，任一新值非有限抛 ValueError；最后按
    E 键序作 E[k]*=gamma*lambda_。done 即停，否则令 s,a=s2,a2。
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
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)):
        raise TypeError("epsilon must be an int or float")
    if isinstance(lambda_, bool) or not isinstance(lambda_, (int, float)):
        raise TypeError("lambda_ must be an int or float")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int):
        raise TypeError("max_steps must be an int")
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
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")

    actions = tuple(_ACTIONS)  # U, R, D, L
    states = sorted(
        state
        for state in _reachable_cells(env)
        if env._cell(state) != "G"
    )
    q = {(state, action): 0.0 for state in states for action in actions}
    rng = random.Random(seed)

    def choose_action(state):
        if rng.random() < epsilon:
            return actions[rng.randrange(4)]
        return max(actions, key=lambda a: q[(state, a)])

    for _ in range(episodes):
        state = env.reset()
        eligibility = {key: 0.0 for key in q}
        action = choose_action(state)
        for _ in range(max_steps):
            next_state, reward, done = env.step(action)
            if done:
                delta = reward - q[(state, action)]
            else:
                next_action = choose_action(next_state)
                delta = (
                    reward
                    + gamma * q[(next_state, next_action)]
                    - q[(state, action)]
                )
            for other in actions:
                if other != action:
                    eligibility[(state, other)] = 0.0
            eligibility[(state, action)] = 1.0
            for key in q:
                q[key] += alpha * delta * eligibility[key]
                if not math.isfinite(q[key]):
                    raise ValueError("Q value must remain finite")
            for key in eligibility:
                eligibility[key] *= gamma * lambda_
            if done:
                break
            state, action = next_state, next_action
    return q


def expected_sarsa_lambda(
    env,
    episodes=500,
    alpha=0.5,
    gamma=0.9,
    epsilon=0.1,
    lambda_=0.9,
    seed=0,
    max_steps=1000,
) -> dict:
    """Expected SARSA(λ)，返回 {((row, col), action): float}。

    Q 覆盖从 S 可达的非 G 格与 U/R/D/L 的全部组合，初始为 0.0。
    每回合 reset，资格迹 E 按 Q 键序置 0.0，先选动作；单回合最多
    max_steps 步，末步未终止也照常选择下一动作、自举并更新后截断。
    每步 step 得 (s2, r, done)：done 时 delta=r-Q[s,a] 且不再选
    动作；否则以更新前 Q 取 s2 上 URDL 首个最大动作 g，令
    π(g)=1-epsilon+epsilon/4、其余各 epsilon/4，自 0.0 按 URDL
    累加 v=Σπ(b)*Q[s2,b]，再按同一 ε 贪心选 a2，令
    delta=r+gamma*v-Q[s,a]。随后作累积迹 E[s,a]+=1.0，按 Q 键序作
    Q[k]+=alpha*delta*E[k]，任一新值非有限抛 ValueError；再按
    E 键序作 E[k]*=gamma*lambda_。done 即停，否则令 s,a=s2,a2。
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
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)):
        raise TypeError("epsilon must be an int or float")
    if isinstance(lambda_, bool) or not isinstance(lambda_, (int, float)):
        raise TypeError("lambda_ must be an int or float")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int):
        raise TypeError("max_steps must be an int")
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
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")

    actions = tuple(_ACTIONS)  # U, R, D, L
    states = sorted(
        state
        for state in _reachable_cells(env)
        if env._cell(state) != "G"
    )
    q = {(state, action): 0.0 for state in states for action in actions}
    rng = random.Random(seed)

    def choose_action(state):
        if rng.random() < epsilon:
            return actions[rng.randrange(4)]
        return max(actions, key=lambda a: q[(state, a)])

    for _ in range(episodes):
        state = env.reset()
        eligibility = {key: 0.0 for key in q}
        action = choose_action(state)
        for _ in range(max_steps):
            next_state, reward, done = env.step(action)
            if done:
                delta = reward - q[(state, action)]
            else:
                greedy = max(actions, key=lambda a: q[(next_state, a)])
                greedy_prob = 1 - epsilon + epsilon / 4
                other_prob = epsilon / 4
                v = 0.0
                for a in actions:
                    p = greedy_prob if a == greedy else other_prob
                    v += p * q[(next_state, a)]
                next_action = choose_action(next_state)
                delta = reward + gamma * v - q[(state, action)]
            eligibility[(state, action)] += 1.0
            for key in q:
                q[key] += alpha * delta * eligibility[key]
                if not math.isfinite(q[key]):
                    raise ValueError("Q value must remain finite")
            for key in eligibility:
                eligibility[key] *= gamma * lambda_
            if done:
                break
            state, action = next_state, next_action
    return q


def boltzmann_q(
    env,
    episodes=500,
    alpha=0.5,
    gamma=0.9,
    temp=1.0,
    seed=0,
    max_steps=1000,
) -> dict:
    """Boltzmann(softmax) 探索的 Q-learning，返回 {((row, col), action): float}。

    Q 覆盖从 S 可达的非 G 格与 U/R/D/L 的全部组合，初始为 0.0。
    每回合 reset，单回合最多 max_steps 步，末步未终止仍自举。
    全部随机性来自一个 random.Random(seed)。

    每步按 URDL 令 m=max_a Q[s,a]、w[a]=exp((Q[s,a]-m)/temp)、
    z=sum(w,0.0)，取 u=rng.random()*z，从 0.0 按 URDL 累加 w 并
    选择首个累计值严格大于 u 的动作，未命中取 L。step 后 done 时
    目标为 r，否则为 r+gamma*max_a Q[s2,a]，作
    Q[s,a]+=alpha*(目标-Q[s,a])；新 Q 非有限抛 ValueError。
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
    if isinstance(temp, bool) or not isinstance(temp, (int, float)):
        raise TypeError("temp must be an int or float")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int):
        raise TypeError("max_steps must be an int")
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if not _is_finite_number(alpha) or alpha <= 0 or alpha > 1:
        raise ValueError("alpha must be finite and in (0, 1]")
    if not _is_finite_number(gamma) or gamma < 0 or gamma >= 1:
        raise ValueError("gamma must be finite and in [0, 1)")
    try:
        temp = float(temp)
    except OverflowError:
        raise ValueError("temp must convert to a finite float")
    if not math.isfinite(temp) or temp <= 0:
        raise ValueError("temp must be finite and > 0")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")

    actions = tuple(_ACTIONS)  # U, R, D, L
    q = {
        (state, action): 0.0
        for state in sorted(_reachable_cells(env))
        if env._cell(state) != "G"
        for action in actions
    }
    rng = random.Random(seed)

    for _ in range(episodes):
        state = env.reset()
        for _ in range(max_steps):
            m = max(q[(state, a)] for a in actions)
            w = [math.exp((q[(state, a)] - m) / temp) for a in actions]
            z = sum(w, 0.0)
            u = rng.random() * z
            cumulative = 0.0
            action = "L"
            for a, weight in zip(actions, w):
                cumulative += weight
                if cumulative > u:
                    action = a
                    break
            next_state, reward, done = env.step(action)
            if done:
                target = reward
            else:
                target = reward + gamma * max(
                    q[(next_state, a)] for a in actions
                )
            key = (state, action)
            new_value = q[key] + alpha * (target - q[key])
            if not math.isfinite(new_value):
                raise ValueError("Q value must remain finite")
            q[key] = new_value
            if done:
                break
            state = next_state
    return q


def annealed_boltzmann_q(
    env,
    episodes=500,
    alpha=0.5,
    gamma=0.9,
    temp_start=1.0,
    temp_end=0.1,
    decay_episodes=500,
    seed=0,
    max_steps=1000,
) -> dict:
    """温度线性退火的 Boltzmann(softmax) 探索 Q-learning，返回
    {((row, col), action): float}。

    Q 覆盖从 S 可达的非 G 格与 U/R/D/L 的全部组合，初始为 0.0。
    每回合 reset，单回合最多 max_steps 步，末步未终止仍自举。
    全部随机性来自一个 random.Random(seed)。

    第 e 回合（0 起）的温度 T：e>=decay_episodes 时 T=temp_end，
    否则令 t=e/decay_episodes、T=temp_start+(temp_end-temp_start)*t。
    每步按 URDL 令 m=max_a Q[s,a]、w[a]=exp((Q[s,a]-m)/T)、
    z=sum(w,0.0)，取 u=rng.random()*z，从 0.0 按 URDL 累加 w 并
    选择首个累计值严格大于 u 的动作，未命中取 L。step 后 done 时
    目标为 r，否则为 r+gamma*max_a Q[s2,a]，作
    Q[s,a]+=alpha*(目标-Q[s,a])；新 Q 非有限抛 ValueError。
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
    if isinstance(temp_start, bool) or not isinstance(
        temp_start, (int, float)
    ):
        raise TypeError("temp_start must be an int or float")
    if isinstance(temp_end, bool) or not isinstance(temp_end, (int, float)):
        raise TypeError("temp_end must be an int or float")
    if isinstance(decay_episodes, bool) or not isinstance(
        decay_episodes, int
    ):
        raise TypeError("decay_episodes must be an int")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int):
        raise TypeError("max_steps must be an int")
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if not _is_finite_number(alpha) or alpha <= 0 or alpha > 1:
        raise ValueError("alpha must be finite and in (0, 1]")
    if not _is_finite_number(gamma) or gamma < 0 or gamma >= 1:
        raise ValueError("gamma must be finite and in [0, 1)")
    try:
        temp_start = float(temp_start)
    except OverflowError:
        raise ValueError("temp_start must convert to a finite float")
    if not math.isfinite(temp_start) or temp_start <= 0:
        raise ValueError("temp_start must be finite and > 0")
    try:
        temp_end = float(temp_end)
    except OverflowError:
        raise ValueError("temp_end must convert to a finite float")
    if not math.isfinite(temp_end) or temp_end <= 0:
        raise ValueError("temp_end must be finite and > 0")
    if decay_episodes <= 0:
        raise ValueError("decay_episodes must be positive")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")

    actions = tuple(_ACTIONS)  # U, R, D, L
    q = {
        (state, action): 0.0
        for state in sorted(_reachable_cells(env))
        if env._cell(state) != "G"
        for action in actions
    }
    rng = random.Random(seed)

    for episode in range(episodes):
        if episode >= decay_episodes:
            temp = temp_end
        else:
            t = episode / decay_episodes
            temp = temp_start + (temp_end - temp_start) * t
        state = env.reset()
        for _ in range(max_steps):
            m = max(q[(state, a)] for a in actions)
            w = [math.exp((q[(state, a)] - m) / temp) for a in actions]
            z = sum(w, 0.0)
            u = rng.random() * z
            cumulative = 0.0
            action = "L"
            for a, weight in zip(actions, w):
                cumulative += weight
                if cumulative > u:
                    action = a
                    break
            next_state, reward, done = env.step(action)
            if done:
                target = reward
            else:
                target = reward + gamma * max(
                    q[(next_state, a)] for a in actions
                )
            key = (state, action)
            new_value = q[key] + alpha * (target - q[key])
            if not math.isfinite(new_value):
                raise ValueError("Q value must remain finite")
            q[key] = new_value
            if done:
                break
            state = next_state
    return q


def true_online_sarsa_lambda(
    env,
    episodes=500,
    alpha=0.5,
    gamma=0.9,
    epsilon=0.1,
    lambda_=0.9,
    seed=0,
    max_steps=1000,
) -> dict:
    """True Online SARSA(λ)，返回 {((row, col), action): float}。

    Q 覆盖从 S 可达的非 G 格与 U/R/D/L 的全部组合，初始为 0.0。
    每回合 reset，资格迹 E 按 Q 键序置 0.0，q_old=0.0，先选动作；
    单回合最多 max_steps 步，末步未终止也照常选择下一动作、自举
    并更新后截断。每步先记 q=Q[s,a] 再 step；done 时 q2=0.0 且不
    再选下一动作，否则立即选 a2 并记更新前 q2=Q[s2,a2]。令
    delta=r+gamma*q2-q；按键序先作 E[k]*=gamma*lambda_，再作
    E[s,a]+=1-alpha*E[s,a]；随后按键序同步作
    Q[k]+=alpha*(delta+q-q_old)*E[k]，再作 Q[s,a]-=alpha*(q-q_old)。
    任一次写入所得 Q 值非有限即抛 ValueError。done 即停，否则令
    s,a,q_old=s2,a2,q2。全部随机性来自一个 random.Random(seed)。
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
    if isinstance(max_steps, bool) or not isinstance(max_steps, int):
        raise TypeError("max_steps must be an int")
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
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")

    actions = tuple(_ACTIONS)  # U, R, D, L
    states = sorted(
        state
        for state in _reachable_cells(env)
        if env._cell(state) != "G"
    )
    q = {(state, action): 0.0 for state in states for action in actions}
    rng = random.Random(seed)

    def choose_action(state):
        if rng.random() < epsilon:
            return actions[rng.randrange(4)]
        return max(actions, key=lambda a: q[(state, a)])

    for _ in range(episodes):
        state = env.reset()
        eligibility = {key: 0.0 for key in q}
        q_old = 0.0
        action = choose_action(state)
        for _ in range(max_steps):
            key = (state, action)
            q_sa = q[key]
            next_state, reward, done = env.step(action)
            if done:
                q_next = 0.0
            else:
                next_action = choose_action(next_state)
                q_next = q[(next_state, next_action)]
            delta = reward + gamma * q_next - q_sa
            for k in eligibility:
                eligibility[k] *= gamma * lambda_
            eligibility[key] += 1 - alpha * eligibility[key]
            coeff = alpha * (delta + q_sa - q_old)
            for k in q:
                q[k] += coeff * eligibility[k]
                if not math.isfinite(q[k]):
                    raise ValueError("Q value must remain finite")
            q[key] -= alpha * (q_sa - q_old)
            if not math.isfinite(q[key]):
                raise ValueError("Q value must remain finite")
            if done:
                break
            state, action, q_old = next_state, next_action, q_next
    return q


def nstep_sarsa(
    env,
    episodes=500,
    alpha=0.5,
    gamma=0.9,
    epsilon=0.1,
    n_steps=5,
    seed=0,
    max_steps=1000,
) -> dict:
    """n 步 SARSA，返回 {((row, col), action): float}。

    Q 覆盖从 S 可达的非 G 格与 U/R/D/L 的全部组合，初始为 0.0。
    每回合 reset、清空 FIFO 并先选动作；单回合最多 max_steps 步。
    step 得 (s2, r, done)：done 时 a2=None 且不再采样，否则立即按
    ε 贪心选 a2（步限末步也选）；将 [s, a, r, done, s2, a2] 入队。
    队长达 n_steps 即更新队头并弹头；回合结束后冲刷，m 取当前队长。
    取前 m 项，G=Σ(k=0..m-1) gamma^k*r[k]，若第 m 项未 done 再加
    gamma^m*Q[(s2, a2)]；终止尾不自举，步限尾用已选 a2 自举，Q 均
    取更新当时值。作 Q[(s, a)]+=alpha*(G-Q[(s, a)])，新值非有限抛
    ValueError。全部随机性来自一个 random.Random(seed)。
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
    if isinstance(n_steps, bool) or not isinstance(n_steps, int):
        raise TypeError("n_steps must be an int")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int):
        raise TypeError("max_steps must be an int")
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if not _is_finite_number(alpha) or alpha <= 0 or alpha > 1:
        raise ValueError("alpha must be finite and in (0, 1]")
    if not _is_finite_number(gamma) or gamma < 0 or gamma >= 1:
        raise ValueError("gamma must be finite and in [0, 1)")
    if not _is_finite_number(epsilon) or epsilon < 0 or epsilon > 1:
        raise ValueError("epsilon must be finite and in [0, 1]")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")

    actions = tuple(_ACTIONS)  # U, R, D, L
    states = sorted(
        state
        for state in _reachable_cells(env)
        if env._cell(state) != "G"
    )
    q = {(state, action): 0.0 for state in states for action in actions}
    rng = random.Random(seed)

    def choose_action(state):
        if rng.random() < epsilon:
            return actions[rng.randrange(4)]
        return max(actions, key=lambda a: q[(state, a)])

    def update(queue, m):
        """以队头起 m 项计算 n 步回报，仅更新队头的 (s, a)。"""
        g = 0.0
        discount = 1.0
        for k in range(m):
            g += discount * queue[k][2]
            discount *= gamma
        tail = queue[m - 1]
        if not tail[3]:
            g += discount * q[(tail[4], tail[5])]
        key = (queue[0][0], queue[0][1])
        new_value = q[key] + alpha * (g - q[key])
        if not math.isfinite(new_value):
            raise ValueError("Q value must remain finite")
        q[key] = new_value

    for _ in range(episodes):
        state = env.reset()
        queue = []
        action = choose_action(state)
        for _ in range(max_steps):
            next_state, reward, done = env.step(action)
            if done:
                next_action = None
            else:
                next_action = choose_action(next_state)
            queue.append(
                [state, action, reward, done, next_state, next_action]
            )
            if len(queue) == n_steps:
                update(queue, n_steps)
                queue.pop(0)
            if done:
                break
            state, action = next_state, next_action
        while queue:
            update(queue, len(queue))
            queue.pop(0)
    return q


def tree_backup(
    env,
    episodes=500,
    alpha=0.5,
    gamma=0.9,
    epsilon=0.1,
    n_steps=5,
    seed=0,
    max_steps=1000,
) -> dict:
    """n 步 Tree Backup，返回 {((row, col), action): float}。

    Q 覆盖从 S 可达的非 G 格与 U/R/D/L 的全部组合，初始为 0.0。
    每回合 reset、清空 FIFO 并先选动作；单回合最多 max_steps 步。
    每次选动作时按选前 Q 保存 ε 贪心策略 π：URDL 中首个最大 Q 动作
    概率为 1-epsilon+epsilon/4，其余各 epsilon/4。step 得
    (s2, r, done)：done 时 a2、π2 均为 None，否则立即选 a2（步限
    末步也选）并记其策略 π2；将 [s, a, r, done, s2, a2, π2] 入队。
    队长达 n_steps 即更新队头并弹头；回合结束后冲刷，m 取 n_steps
    或当前队长。取前 m 项，末项 done 时令 G=0.0，否则令
    G=Q[s2,a2]，再从末项逆推：done 项令 G=r，否则令
    G=r+gamma*(Σ(b!=a2) π2[b]*Q[s2,b]+π2[a2]*G)，求和自 0.0 按
    URDL 累加，Q 取本次更新时值。作
    Q[s,a]+=alpha*(G-Q[s,a])，G 或新 Q 非有限抛 ValueError。
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
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)):
        raise TypeError("epsilon must be an int or float")
    if isinstance(n_steps, bool) or not isinstance(n_steps, int):
        raise TypeError("n_steps must be an int")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int):
        raise TypeError("max_steps must be an int")
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if not _is_finite_number(alpha) or alpha <= 0 or alpha > 1:
        raise ValueError("alpha must be finite and in (0, 1]")
    if not _is_finite_number(gamma) or gamma < 0 or gamma >= 1:
        raise ValueError("gamma must be finite and in [0, 1)")
    if not _is_finite_number(epsilon) or epsilon < 0 or epsilon > 1:
        raise ValueError("epsilon must be finite and in [0, 1]")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")

    actions = tuple(_ACTIONS)  # U, R, D, L
    states = sorted(
        state
        for state in _reachable_cells(env)
        if env._cell(state) != "G"
    )
    q = {(state, action): 0.0 for state in states for action in actions}
    rng = random.Random(seed)

    def policy_for(state):
        """选前 Q 下的 ε 贪心策略概率，按 URDL 排列。"""
        greedy = max(actions, key=lambda a: q[(state, a)])
        greedy_prob = 1 - epsilon + epsilon / 4
        other_prob = epsilon / 4
        return tuple(
            greedy_prob if a == greedy else other_prob for a in actions
        )

    def choose_action(state):
        if rng.random() < epsilon:
            action = actions[rng.randrange(4)]
        else:
            action = max(actions, key=lambda a: q[(state, a)])
        return action, policy_for(state)

    def update(queue, m):
        """以队头起 m 项逆推 tree-backup 回报，仅更新队头的 (s, a)。"""
        tail = queue[m - 1]
        if tail[3]:
            g = 0.0
        else:
            g = q[(tail[4], tail[5])]
        for k in range(m - 1, -1, -1):
            s, a, reward, done, next_state, next_action, next_policy = (
                queue[k]
            )
            if done:
                g = reward
            else:
                a2_index = actions.index(next_action)
                expected = 0.0
                for j, b in enumerate(actions):
                    if b != next_action:
                        expected += next_policy[j] * q[(next_state, b)]
                g = reward + gamma * (
                    expected + next_policy[a2_index] * g
                )
        if not math.isfinite(g):
            raise ValueError("return must remain finite")
        key = (queue[0][0], queue[0][1])
        new_value = q[key] + alpha * (g - q[key])
        if not math.isfinite(new_value):
            raise ValueError("Q value must remain finite")
        q[key] = new_value

    for _ in range(episodes):
        state = env.reset()
        queue = []
        action, _policy = choose_action(state)
        for _ in range(max_steps):
            next_state, reward, done = env.step(action)
            if done:
                next_action = None
                next_policy = None
            else:
                next_action, next_policy = choose_action(next_state)
            queue.append(
                [
                    state,
                    action,
                    reward,
                    done,
                    next_state,
                    next_action,
                    next_policy,
                ]
            )
            if len(queue) == n_steps:
                update(queue, n_steps)
                queue.pop(0)
            if done:
                break
            state, action = next_state, next_action
        while queue:
            update(queue, len(queue))
            queue.pop(0)
    return q


def expected_sarsa(
    env,
    episodes=500,
    alpha=0.5,
    gamma=0.9,
    epsilon=0.1,
    seed=0,
    max_steps=1000,
) -> dict:
    """Expected SARSA，返回 {((row, col), action): float}。

    Q 覆盖从 S 可达的非 G 格与 U/R/D/L 的全部组合，初始为 0.0。
    每回合 reset，先选动作；单回合最多 max_steps 步，末步未终止
    也照常选择下一动作并自举。非终止步以 s2 上 URDL 首个最大 Q
    动作 g 的概率为 1-epsilon+epsilon/4、其余各 epsilon/4，自
    0.0 按 URDL 累加 v=Σp(a)*Q[s2,a]，目标为 r+gamma*v；done 时
    目标为 r 且不再选下一动作。全部随机性来自一个
    random.Random(seed)。
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
    if isinstance(max_steps, bool) or not isinstance(max_steps, int):
        raise TypeError("max_steps must be an int")
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if not _is_finite_number(alpha) or alpha <= 0 or alpha > 1:
        raise ValueError("alpha must be finite and in (0, 1]")
    if not _is_finite_number(gamma) or gamma < 0 or gamma >= 1:
        raise ValueError("gamma must be finite and in [0, 1)")
    if not _is_finite_number(epsilon) or epsilon < 0 or epsilon > 1:
        raise ValueError("epsilon must be finite and in [0, 1]")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")

    actions = tuple(_ACTIONS)  # U, R, D, L
    states = sorted(
        state
        for state in _reachable_cells(env)
        if env._cell(state) != "G"
    )
    q = {(state, action): 0.0 for state in states for action in actions}
    rng = random.Random(seed)

    def choose_action(state):
        if rng.random() < epsilon:
            return actions[rng.randrange(4)]
        return max(actions, key=lambda a: q[(state, a)])

    for _ in range(episodes):
        state = env.reset()
        action = choose_action(state)
        for _ in range(max_steps):
            next_state, reward, done = env.step(action)
            if done:
                target = reward
            else:
                next_action = choose_action(next_state)
                greedy = max(actions, key=lambda a: q[(next_state, a)])
                greedy_prob = 1 - epsilon + epsilon / 4
                other_prob = epsilon / 4
                v = 0.0
                for a in actions:
                    p = greedy_prob if a == greedy else other_prob
                    v += p * q[(next_state, a)]
                target = reward + gamma * v
            key = (state, action)
            new_value = q[key] + alpha * (target - q[key])
            if not math.isfinite(new_value):
                raise ValueError("Q value must remain finite")
            q[key] = new_value
            if done:
                break
            state, action = next_state, next_action
    return q


def q_sigma(
    env,
    episodes=500,
    alpha=0.5,
    gamma=0.9,
    epsilon=0.1,
    sigma=0.5,
    seed=0,
    max_steps=1000,
) -> dict:
    """Q(σ)，返回 {((row, col), action): float}。

    Q 覆盖从 S 可达的非 G 格与 U/R/D/L 的全部组合，初始为 0.0。
    每回合 reset，先选动作；单回合最多 max_steps 步，末步未终止
    也照常选择下一动作并自举。非终止步先按 ε 贪心选 a2，再以更新前
    Q 取 s2 上 URDL 首个最大 Q 动作 g：g 的概率为
    1-epsilon+epsilon/4、其余各 epsilon/4，自 0.0 按 URDL 累加
    v=Σp(b)*Q[s2,b]，目标为
    r+gamma*(sigma*Q[s2,a2]+(1-sigma)*v)；done 时目标为 r 且不再
    选下一动作。sigma=1 时退化为 SARSA、sigma=0 时为 Expected
    SARSA。全部随机性来自一个 random.Random(seed)。
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
    if isinstance(sigma, bool) or not isinstance(sigma, (int, float)):
        raise TypeError("sigma must be an int or float")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int):
        raise TypeError("max_steps must be an int")
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if not _is_finite_number(alpha) or alpha <= 0 or alpha > 1:
        raise ValueError("alpha must be finite and in (0, 1]")
    if not _is_finite_number(gamma) or gamma < 0 or gamma >= 1:
        raise ValueError("gamma must be finite and in [0, 1)")
    if not _is_finite_number(epsilon) or epsilon < 0 or epsilon > 1:
        raise ValueError("epsilon must be finite and in [0, 1]")
    if not _is_finite_number(sigma) or sigma < 0 or sigma > 1:
        raise ValueError("sigma must be finite and in [0, 1]")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")

    actions = tuple(_ACTIONS)  # U, R, D, L
    states = sorted(
        state
        for state in _reachable_cells(env)
        if env._cell(state) != "G"
    )
    q = {(state, action): 0.0 for state in states for action in actions}
    rng = random.Random(seed)

    def choose_action(state):
        if rng.random() < epsilon:
            return actions[rng.randrange(4)]
        return max(actions, key=lambda a: q[(state, a)])

    for _ in range(episodes):
        state = env.reset()
        action = choose_action(state)
        for _ in range(max_steps):
            next_state, reward, done = env.step(action)
            if done:
                target = reward
            else:
                next_action = choose_action(next_state)
                greedy = max(actions, key=lambda a: q[(next_state, a)])
                greedy_prob = 1 - epsilon + epsilon / 4
                other_prob = epsilon / 4
                v = 0.0
                for a in actions:
                    p = greedy_prob if a == greedy else other_prob
                    v += p * q[(next_state, a)]
                target = reward + gamma * (
                    sigma * q[(next_state, next_action)] + (1 - sigma) * v
                )
            key = (state, action)
            new_value = q[key] + alpha * (target - q[key])
            if not math.isfinite(new_value):
                raise ValueError("Q value must remain finite")
            q[key] = new_value
            if done:
                break
            state, action = next_state, next_action
    return q


def nstep_q_sigma(
    env,
    episodes=500,
    alpha=0.5,
    gamma=0.9,
    epsilon=0.1,
    sigma=0.5,
    n_steps=5,
    seed=0,
    max_steps=1000,
) -> dict:
    """n 步 Q(σ)，返回 {((row, col), action): float}。

    Q 覆盖从 S 可达的非 G 格与 U/R/D/L 的全部组合，初始为 0.0。
    每回合 reset、清空 FIFO 并先选动作；单回合最多 max_steps 步。
    每次选动作时按选前 Q 保存 ε 贪心策略 π：URDL 中首个最大 Q 动作
    概率为 1-epsilon+epsilon/4，其余各 epsilon/4。step 得
    (s2, r, done)：done 时 a2、π2 均为 None，否则立即选 a2（步限
    末步也选）并记其策略 π2；将 [s, a, r, done, s2, a2, π2] 入队。
    队长达 n_steps 即更新队头并弹头；回合结束后冲刷，m 取当前队长。
    取前 m 项，末项 done 时令 G=0.0，否则令 G=Q[s2,a2]，再从末项
    逆推：done 项令 G=r，否则按 URDL 自 0.0 累加
    S=Σ(b!=a2) π2[b]*Q[s2,b]，令
    G=r+gamma*((1-sigma)*S+(sigma+(1-sigma)*π2[a2])*G)，Q 取本次
    更新时值；sigma=1 时退化为 n 步 SARSA、sigma=0 时为 n 步
    Tree Backup。仅作 Q[s,a]+=alpha*(G-Q[s,a])，G 或新 Q 非有限
    抛 ValueError。全部随机性来自一个 random.Random(seed)。
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
    if isinstance(sigma, bool) or not isinstance(sigma, (int, float)):
        raise TypeError("sigma must be an int or float")
    if isinstance(n_steps, bool) or not isinstance(n_steps, int):
        raise TypeError("n_steps must be an int")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int):
        raise TypeError("max_steps must be an int")
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if not _is_finite_number(alpha) or alpha <= 0 or alpha > 1:
        raise ValueError("alpha must be finite and in (0, 1]")
    if not _is_finite_number(gamma) or gamma < 0 or gamma >= 1:
        raise ValueError("gamma must be finite and in [0, 1)")
    if not _is_finite_number(epsilon) or epsilon < 0 or epsilon > 1:
        raise ValueError("epsilon must be finite and in [0, 1]")
    if not _is_finite_number(sigma) or sigma < 0 or sigma > 1:
        raise ValueError("sigma must be finite and in [0, 1]")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")

    actions = tuple(_ACTIONS)  # U, R, D, L
    states = sorted(
        state
        for state in _reachable_cells(env)
        if env._cell(state) != "G"
    )
    q = {(state, action): 0.0 for state in states for action in actions}
    rng = random.Random(seed)

    def policy_for(state):
        """选前 Q 下的 ε 贪心策略概率，按 URDL 排列。"""
        greedy = max(actions, key=lambda a: q[(state, a)])
        greedy_prob = 1 - epsilon + epsilon / 4
        other_prob = epsilon / 4
        return tuple(
            greedy_prob if a == greedy else other_prob for a in actions
        )

    def choose_action(state):
        if rng.random() < epsilon:
            action = actions[rng.randrange(4)]
        else:
            action = max(actions, key=lambda a: q[(state, a)])
        return action, policy_for(state)

    def update(queue, m):
        """以队头起 m 项逆推 Q(σ) 回报，仅更新队头的 (s, a)。"""
        tail = queue[m - 1]
        if tail[3]:
            g = 0.0
        else:
            g = q[(tail[4], tail[5])]
        for k in range(m - 1, -1, -1):
            s, a, reward, done, next_state, next_action, next_policy = (
                queue[k]
            )
            if done:
                g = reward
            else:
                a2_index = actions.index(next_action)
                expected = 0.0
                for j, b in enumerate(actions):
                    if b != next_action:
                        expected += next_policy[j] * q[(next_state, b)]
                g = reward + gamma * (
                    (1 - sigma) * expected
                    + (sigma + (1 - sigma) * next_policy[a2_index]) * g
                )
        if not math.isfinite(g):
            raise ValueError("return must remain finite")
        key = (queue[0][0], queue[0][1])
        new_value = q[key] + alpha * (g - q[key])
        if not math.isfinite(new_value):
            raise ValueError("Q value must remain finite")
        q[key] = new_value

    for _ in range(episodes):
        state = env.reset()
        queue = []
        action, _policy = choose_action(state)
        for _ in range(max_steps):
            next_state, reward, done = env.step(action)
            if done:
                next_action = None
                next_policy = None
            else:
                next_action, next_policy = choose_action(next_state)
            queue.append(
                [
                    state,
                    action,
                    reward,
                    done,
                    next_state,
                    next_action,
                    next_policy,
                ]
            )
            if len(queue) == n_steps:
                update(queue, n_steps)
                queue.pop(0)
            if done:
                break
            state, action = next_state, next_action
        while queue:
            update(queue, len(queue))
            queue.pop(0)
    return q


def double_q_learning(
    env,
    episodes=500,
    alpha=0.5,
    gamma=0.9,
    epsilon=0.1,
    seed=0,
    max_steps=1000,
) -> dict:
    """Double Q-learning，返回键序 q1、q2、q 的 dict。

    Q1、Q2 均覆盖从 S 可达的非 G 格与 U/R/D/L 的全部组合，初始为
    0.0，二者同键序、同键域且互相独立。每回合 reset，单回合最多
    max_steps 步；全部随机性来自一个 random.Random(seed)。

    选动作先调一次 random()：其值 < epsilon 时按 URDL 调
    randrange(4) 取探索动作，否则取 Q1+Q2 最大且 URDL 中首个的
    动作。step 后必调一次 randrange(2)：0 时更新 Q1、1 时更新 Q2。
    待更新表为 Qi、另一表为 Qj；done 时目标为 r，否则 g 为 Qi 在
    s2 上 URDL 首个最大动作，目标为 r+gamma*Qj[s2,g]，随后
    Qi[s,a]+=alpha*(目标-Qi[s,a])。新值非有限抛 ValueError。done
    即停；到达步限且未终止的末步仍照常自举更新。
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
    if isinstance(max_steps, bool) or not isinstance(max_steps, int):
        raise TypeError("max_steps must be an int")
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if not _is_finite_number(alpha) or alpha <= 0 or alpha > 1:
        raise ValueError("alpha must be finite and in (0, 1]")
    if not _is_finite_number(gamma) or gamma < 0 or gamma >= 1:
        raise ValueError("gamma must be finite and in [0, 1)")
    if not _is_finite_number(epsilon) or epsilon < 0 or epsilon > 1:
        raise ValueError("epsilon must be finite and in [0, 1]")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")

    actions = tuple(_ACTIONS)  # U, R, D, L
    states = sorted(
        state
        for state in _reachable_cells(env)
        if env._cell(state) != "G"
    )
    q1 = {(state, action): 0.0 for state in states for action in actions}
    q2 = {(state, action): 0.0 for state in states for action in actions}
    rng = random.Random(seed)

    for _ in range(episodes):
        state = env.reset()
        for _ in range(max_steps):
            if rng.random() < epsilon:
                action = actions[rng.randrange(4)]
            else:
                action = max(
                    actions,
                    key=lambda a: q1[(state, a)] + q2[(state, a)],
                )
            next_state, reward, done = env.step(action)
            if rng.randrange(2) == 0:
                qi, qj = q1, q2
            else:
                qi, qj = q2, q1
            key = (state, action)
            if done:
                target = reward
            else:
                greedy = max(
                    actions, key=lambda a: qi[(next_state, a)]
                )
                target = reward + gamma * qj[(next_state, greedy)]
            new_value = qi[key] + alpha * (target - qi[key])
            if not math.isfinite(new_value):
                raise ValueError("Q value must remain finite")
            qi[key] = new_value
            if done:
                break
            state = next_state

    q = {
        key: (q1[key] + q2[key]) / 2.0 for key in q1
    }
    return {"q1": q1, "q2": q2, "q": q}


def bootstrapped_q_learning(
    env,
    episodes=500,
    heads=5,
    alpha=0.5,
    gamma=0.9,
    epsilon=0.1,
    mask_prob=0.8,
    seed=0,
    max_steps=1000,
    epsilon_end=None,
    decay_episodes=None,
) -> dict:
    """Bootstrapped Q-learning（多头），返回键序 q、heads 的 dict。

    各头 Q 表均覆盖从 S 可达的非 G 格与 U/R/D/L 的全部组合，初始
    为 0.0，键域、键序同 double_q_learning 的 q1，各表互相独立。
    每回合先调 randrange(heads) 选定本回合用头再 reset，单回合最多
    max_steps 步；全部随机性来自一个 random.Random(seed)。

    每步先调一次 random()：其值 < epsilon 时按 URDL 调
    randrange(4) 取探索动作，否则按本回合用头取 URDL 首个最大 Q
    动作。step 后各头按序各调一次 random()：其值 < mask_prob 者用
    自身 Q 表更新，done 时目标为 r，否则 g 为该头在 s2 上 URDL
    首个最大动作，目标为 r+gamma*Q[s2,g]，随后
    Q[s,a]+=alpha*(目标-Q[s,a])。done 即停；到达步限且未终止的
    末步仍照常自举更新，且各头掩码照抽完。新值非有限抛 ValueError。
    q 为各头 Q 按键序从 0.0 顺序累加后的 float 均值。

    epsilon_end 与 decay_episodes 须同为 None 或同启用：同为 None
    时各回合探索率恒为 epsilon；同启用时第 e 回合（0 起）令
    t=min(e/decay_episodes, 1)，探索率为
    epsilon+(epsilon_end-epsilon)*t，非有限抛 ValueError。
    """
    if not isinstance(env, GridWorld):
        raise TypeError("env must be a GridWorld")
    if isinstance(episodes, bool) or not isinstance(episodes, int):
        raise TypeError("episodes must be an int")
    if isinstance(heads, bool) or not isinstance(heads, int):
        raise TypeError("heads must be an int")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an int")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise TypeError("alpha must be an int or float")
    if isinstance(gamma, bool) or not isinstance(gamma, (int, float)):
        raise TypeError("gamma must be an int or float")
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)):
        raise TypeError("epsilon must be an int or float")
    if isinstance(mask_prob, bool) or not isinstance(
        mask_prob, (int, float)
    ):
        raise TypeError("mask_prob must be an int or float")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int):
        raise TypeError("max_steps must be an int")
    if epsilon_end is not None and (
        isinstance(epsilon_end, bool)
        or not isinstance(epsilon_end, (int, float))
    ):
        raise TypeError("epsilon_end must be None, an int or float")
    if decay_episodes is not None and (
        isinstance(decay_episodes, bool)
        or not isinstance(decay_episodes, int)
    ):
        raise TypeError("decay_episodes must be None or an int")
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if heads < 2:
        raise ValueError("heads must be >= 2")
    if not _is_finite_number(alpha) or alpha <= 0 or alpha > 1:
        raise ValueError("alpha must be finite and in (0, 1]")
    if not _is_finite_number(gamma) or gamma < 0 or gamma >= 1:
        raise ValueError("gamma must be finite and in [0, 1)")
    if not _is_finite_number(epsilon) or epsilon < 0 or epsilon > 1:
        raise ValueError("epsilon must be finite and in [0, 1]")
    if (
        not _is_finite_number(mask_prob)
        or mask_prob <= 0
        or mask_prob > 1
    ):
        raise ValueError("mask_prob must be finite and in (0, 1]")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if epsilon_end is None:
        if decay_episodes is not None:
            raise ValueError(
                "decay_episodes must be None when epsilon_end is None"
            )
    else:
        try:
            epsilon_end = float(epsilon_end)
        except OverflowError:
            raise ValueError("epsilon_end must be finite and in [0, 1]")
        if (
            not math.isfinite(epsilon_end)
            or epsilon_end < 0
            or epsilon_end > 1
        ):
            raise ValueError("epsilon_end must be finite and in [0, 1]")
        if decay_episodes is None:
            raise ValueError(
                "decay_episodes must be set when epsilon_end is set"
            )
        if decay_episodes <= 0:
            raise ValueError("decay_episodes must be positive")

    actions = tuple(_ACTIONS)  # U, R, D, L
    states = sorted(
        state
        for state in _reachable_cells(env)
        if env._cell(state) != "G"
    )
    base = {
        (state, action): 0.0 for state in states for action in actions
    }
    q_tables = [dict(base) for _ in range(heads)]
    rng = random.Random(seed)

    for episode in range(episodes):
        active = rng.randrange(heads)
        if epsilon_end is None:
            rate = epsilon
        else:
            t = min(episode / decay_episodes, 1.0)
            rate = epsilon + (epsilon_end - epsilon) * t
            if not math.isfinite(rate):
                raise ValueError("epsilon must remain finite")
        state = env.reset()
        for _ in range(max_steps):
            if rng.random() < rate:
                action = actions[rng.randrange(4)]
            else:
                action = max(
                    actions,
                    key=lambda a: q_tables[active][(state, a)],
                )
            next_state, reward, done = env.step(action)
            key = (state, action)
            for h in range(heads):
                if rng.random() < mask_prob:
                    q_table = q_tables[h]
                    if done:
                        target = reward
                    else:
                        greedy = max(
                            actions,
                            key=lambda a: q_table[(next_state, a)],
                        )
                        target = (
                            reward
                            + gamma * q_table[(next_state, greedy)]
                        )
                    new_value = (
                        q_table[key] + alpha * (target - q_table[key])
                    )
                    if not math.isfinite(new_value):
                        raise ValueError("Q value must remain finite")
                    q_table[key] = new_value
            if done:
                break
            state = next_state

    q = {}
    for key in q_tables[0]:
        total = 0.0
        for q_table in q_tables:
            total += q_table[key]
        q[key] = total / heads
    return {"q": q, "heads": q_tables}


def bootstrapped_q_converge(
    env,
    episodes=500,
    seed=0,
    check_every=10,
    tolerance=0.01,
    patience=3,
) -> dict:
    """多头 bootstrapped Q-learning 训练并报告收敛早停，返回键序
    q、heads、episodes、checks、converged 的 dict。

    训练完全沿用 bootstrapped_q_learning 的默认项（heads=5、
    alpha=0.5、gamma=0.9、epsilon=0.1、mask_prob=0.8、
    max_steps=1000，不启用 epsilon 衰减）；env、episodes、seed 的
    校验与异常分类沿用原函数。check_every、patience 须为非 bool 的
    正 int，tolerance 须为非 bool 的有限 int/float 且 >= 0；类型
    违约抛 TypeError，其余抛 ValueError，全部在首次 reset 前校验。

    第 e 回合（0 起，即 e+1 回合完成后）逢 check_every 的倍数或末
    回合（e==episodes-1）作一次检查，检查只读取 Q 表、不消费任何
    随机数：M 为按 q 的键序从 0.0 累加五头均值所得 dict（即原函数
    q 的同序构造），old 首次检查时为全 0 的同键 dict，
    D=max|M[key]-old[key]|；A 为各状态上五头分别按 URDL 取首个
    最大动作后五头动作全同的状态比例（一致状态数 / 状态数）。
    checks 追加 [e, D, A, D<=tolerance and A==1.0]，四项依次为
    int、float、float、bool，随后以新 M 更新 old。D、A 非有限抛
    ValueError。连续 patience 行末项为真即在该回合后停止训练，
    否则跑满 episodes 回合。

    episodes 为实际运行的回合数；converged 表示是否命中早停；
    q、heads 的结构与值等同 bootstrapped_q_learning 在相同实跑回
    合数与 seed 下的结果。仅用标准库，不引入命令行入口。
    """
    if not isinstance(env, GridWorld):
        raise TypeError("env must be a GridWorld")
    if isinstance(episodes, bool) or not isinstance(episodes, int):
        raise TypeError("episodes must be an int")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an int")
    if isinstance(check_every, bool) or not isinstance(
        check_every, int
    ):
        raise TypeError("check_every must be an int")
    if isinstance(patience, bool) or not isinstance(patience, int):
        raise TypeError("patience must be an int")
    if isinstance(tolerance, bool) or not isinstance(
        tolerance, (int, float)
    ):
        raise TypeError("tolerance must be an int or float")
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if check_every <= 0:
        raise ValueError("check_every must be positive")
    if patience <= 0:
        raise ValueError("patience must be positive")
    if not _is_finite_number(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and >= 0")

    heads = 5
    alpha = 0.5
    gamma = 0.9
    epsilon = 0.1
    mask_prob = 0.8
    max_steps = 1000

    actions = tuple(_ACTIONS)  # U, R, D, L
    states = sorted(
        state
        for state in _reachable_cells(env)
        if env._cell(state) != "G"
    )
    base = {
        (state, action): 0.0 for state in states for action in actions
    }
    q_tables = [dict(base) for _ in range(heads)]
    rng = random.Random(seed)

    checks = []
    old = dict(base)
    streak = 0
    ran = 0
    converged = False

    for episode in range(episodes):
        active = rng.randrange(heads)
        state = env.reset()
        for _ in range(max_steps):
            if rng.random() < epsilon:
                action = actions[rng.randrange(4)]
            else:
                action = max(
                    actions,
                    key=lambda a: q_tables[active][(state, a)],
                )
            next_state, reward, done = env.step(action)
            key = (state, action)
            for h in range(heads):
                if rng.random() < mask_prob:
                    q_table = q_tables[h]
                    if done:
                        target = reward
                    else:
                        greedy = max(
                            actions,
                            key=lambda a: q_table[(next_state, a)],
                        )
                        target = (
                            reward
                            + gamma * q_table[(next_state, greedy)]
                        )
                    new_value = (
                        q_table[key] + alpha * (target - q_table[key])
                    )
                    if not math.isfinite(new_value):
                        raise ValueError("Q value must remain finite")
                    q_table[key] = new_value
            if done:
                break
            state = next_state

        ran = episode + 1
        is_last = episode == episodes - 1
        if (episode + 1) % check_every == 0 or is_last:
            mean_q = {}
            for key in q_tables[0]:
                total = 0.0
                for q_table in q_tables:
                    total += q_table[key]
                value = total / heads
                if not math.isfinite(value):
                    raise ValueError("Q mean must remain finite")
                mean_q[key] = value
            delta = 0.0
            for key in mean_q:
                change = abs(mean_q[key] - old[key])
                if change > delta:
                    delta = change
            if not math.isfinite(delta):
                raise ValueError("convergence delta must be finite")
            agree = 0
            for cell in states:
                first = max(
                    actions,
                    key=lambda a: q_tables[0][(cell, a)],
                )
                if all(
                    max(
                        actions,
                        key=lambda a: q_table[(cell, a)],
                    )
                    == first
                    for q_table in q_tables
                ):
                    agree += 1
            agreement = agree / len(states)
            if not math.isfinite(agreement):
                raise ValueError("agreement ratio must be finite")
            passed = delta <= tolerance and agreement == 1.0
            checks.append([episode, delta, agreement, passed])
            old = mean_q
            if passed:
                streak += 1
                if streak >= patience:
                    converged = True
                    break
            else:
                streak = 0

    q = {}
    for key in q_tables[0]:
        total = 0.0
        for q_table in q_tables:
            total += q_table[key]
        q[key] = total / heads
    return {
        "q": q,
        "heads": q_tables,
        "episodes": ran,
        "checks": checks,
        "converged": converged,
    }


def off_policy_mc_control(
    env,
    episodes=500,
    gamma=0.9,
    epsilon=0.1,
    seed=0,
    max_steps=1000,
) -> dict:
    """离轨策略 MC 控制，返回键序 q、c 的 dict。

    Q、C 均覆盖从 S 可达的非 G 格与 U/R/D/L 的全部组合，初始为
    0.0，二者同键序、同键域且互相独立。每回合 reset，至 done 或
    max_steps 步；全部随机性来自一个 random.Random(seed)。

    每步先以当前 Q 取状态 s 上 URDL 首个最大动作 g；选 a 时先调
    一次 random()，其值 < epsilon 才调 randrange(4) 取探索动作，
    否则取 g。记录 [s, a, r, b]，b 为行为策略选中 a 的概率：
    a==g 时为 1-epsilon+epsilon/4，否则为 epsilon/4。

    回合结束（done 或步限截断均不自举）令 G=0.0、W=1.0，逆序对
    每项先作 G=r+gamma*G、C[s,a]+=W、
    Q[s,a]+=W/C[s,a]*(G-Q[s,a])；任一新值非有限即抛 ValueError。
    再按更新后 Q 取该 s 的 g；若 a!=g 则停止本回合回溯，否则作
    W/=b 后继续；epsilon=0 时延续项必有 a==g、b=1，不会除零。
    """
    if not isinstance(env, GridWorld):
        raise TypeError("env must be a GridWorld")
    if isinstance(episodes, bool) or not isinstance(episodes, int):
        raise TypeError("episodes must be an int")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an int")
    if isinstance(gamma, bool) or not isinstance(gamma, (int, float)):
        raise TypeError("gamma must be an int or float")
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)):
        raise TypeError("epsilon must be an int or float")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int):
        raise TypeError("max_steps must be an int")
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if not _is_finite_number(gamma) or gamma < 0 or gamma >= 1:
        raise ValueError("gamma must be finite and in [0, 1)")
    if not _is_finite_number(epsilon) or epsilon < 0 or epsilon > 1:
        raise ValueError("epsilon must be finite and in [0, 1]")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")

    actions = tuple(_ACTIONS)  # U, R, D, L
    states = sorted(
        state
        for state in _reachable_cells(env)
        if env._cell(state) != "G"
    )
    q = {(state, action): 0.0 for state in states for action in actions}
    c = {(state, action): 0.0 for state in states for action in actions}
    rng = random.Random(seed)

    for _ in range(episodes):
        state = env.reset()
        trajectory = []
        for _ in range(max_steps):
            greedy = max(actions, key=lambda a: q[(state, a)])
            if rng.random() < epsilon:
                action = actions[rng.randrange(4)]
            else:
                action = greedy
            if action == greedy:
                behavior_prob = 1 - epsilon + epsilon / 4
            else:
                behavior_prob = epsilon / 4
            next_state, reward, done = env.step(action)
            trajectory.append([state, action, reward, behavior_prob])
            if done:
                break
            state = next_state

        g_return = 0.0
        weight = 1.0
        for s, a, r, b in reversed(trajectory):
            g_return = r + gamma * g_return
            if not math.isfinite(g_return):
                raise ValueError("return must remain finite")
            key = (s, a)
            c[key] += weight
            if not math.isfinite(c[key]):
                raise ValueError("C value must remain finite")
            q[key] += weight / c[key] * (g_return - q[key])
            if not math.isfinite(q[key]):
                raise ValueError("Q value must remain finite")
            greedy = max(actions, key=lambda action: q[(s, action)])
            if a != greedy:
                break
            weight /= b
    return {"q": q, "c": c}


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


def entropy_actor_critic(
    env,
    episodes=500,
    alpha=0.05,
    beta=0.1,
    gamma=0.9,
    entropy_coef=0.01,
    seed=0,
    max_steps=1000,
) -> dict:
    """在线一步带熵正则的 actor-critic，返回 h/v 表与逐回合统计。

    H 覆盖从 S 可达的非 G 格与 U/R/D/L 的全部组合，V 覆盖同样的格子，
    初始均为 0.0。每回合 reset，至 done 或 max_steps 步；每步先以旧 H
    按 softmax 得 URDL 概率 p，并令 E=-Σp[b]log(p[b])（p[b]=0 项计
    0），再仅调用一次 random() 采样动作，保存旧 V[s] 后 step，得
    (s2, r, done)：done 时 δ=r-V[s]，否则 δ=r+gamma*V[s2]-V[s]；
    达到步限且未 done 的末步同样按后式自举。以旧 p、E 按 URDL 对各 b
    同步更新
    H[s,b]+=alpha*(δ*(I[b=a]-p[b])+entropy_coef*p[b]*(-log(p[b])-E))，
    p[b]=0 时熵梯度项为 0；再作 V[s]+=beta*δ。δ、E、梯度或任一更新
    值非有限即抛 ValueError。全部随机性来自一个 random.Random(seed)。
    entropy_coef=0 时逐值与 actor_critic 等同。

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
    if isinstance(entropy_coef, bool) or not isinstance(
        entropy_coef, (int, float)
    ):
        raise TypeError("entropy_coef must be an int or float")
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
    if (
        not _is_finite_number(entropy_coef)
        or entropy_coef < 0
        or entropy_coef > 1
    ):
        raise ValueError("entropy_coef must be finite and in [0, 1]")

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
            entropy_sum = 0.0
            for p_a in probs:
                if p_a > 0.0:
                    entropy_sum += p_a * math.log(p_a)
            entropy = -entropy_sum
            if not math.isfinite(entropy):
                raise ValueError("entropy must remain finite")
            u = rng.random()
            cumulative = 0.0
            action = "L"
            for a, p_a in zip(actions, probs):
                cumulative += p_a
                if cumulative > u:
                    action = a
                    break
            value = v[state]
            next_state, reward, done = env.step(action)
            steps += 1
            total_reward += reward
            if done:
                delta = reward - value
            else:
                delta = reward + gamma * v[next_state] - value
            if not math.isfinite(delta):
                raise ValueError("delta must remain finite")
            for b, p_b in zip(actions, probs):
                indicator = 1.0 if b == action else 0.0
                if entropy_coef == 0.0 or p_b == 0.0:
                    update = alpha * delta * (indicator - p_b)
                else:
                    update = alpha * (
                        delta * (indicator - p_b)
                        + entropy_coef
                        * p_b
                        * (-math.log(p_b) - entropy)
                    )
                if not math.isfinite(update):
                    raise ValueError("policy gradient must remain finite")
                new_h = h[(state, b)] + update
                if not math.isfinite(new_h):
                    raise ValueError("H value must remain finite")
                h[(state, b)] = new_h
            new_v = v[state] + beta * delta
            if not math.isfinite(new_v):
                raise ValueError("V value must remain finite")
            v[state] = new_v
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


def adaptive_entropy_ac(
    env,
    episodes=500,
    alpha=0.05,
    beta=0.1,
    gamma=0.9,
    eta=0.01,
    target=1.0,
    coef=0.01,
    seed=0,
    max_steps=1000,
) -> dict:
    """在线一步自适应熵系数的 actor-critic，返回 h/v 表、逐回合统计与系数。

    与 entropy_actor_critic 一致，但熵正则系数 c=exp(z) 逐步自适应：
    初始 z=log(coef)，每步以旧 H 按 URDL softmax 得概率 p、熵 E，按原
    接口计算 δ，并以旧 c 同步更新
    H[s,b]+=alpha*(δ*(I[b=a]-p[b])+c*p[b]*(-log(p[b])-E))（p[b]=0
    时熵项为 0），再作 V[s]+=beta*δ，最后 z+=eta*(target-E)。
    eta 须属 (0,1]，target 须属 [0,log(4)]，coef 须为正数。中间量或
    新值非有限即抛 ValueError。返回键依次为 h、v、episodes、
    coefficients，末项为各回合末 exp(z) 的 float 列表。
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
    if isinstance(eta, bool) or not isinstance(eta, (int, float)):
        raise TypeError("eta must be an int or float")
    if isinstance(target, bool) or not isinstance(target, (int, float)):
        raise TypeError("target must be an int or float")
    if isinstance(coef, bool) or not isinstance(coef, (int, float)):
        raise TypeError("coef must be an int or float")
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
    try:
        eta_value = float(eta)
    except OverflowError as exc:
        raise ValueError(
            "eta must be convertible to a finite float"
        ) from exc
    try:
        target_value = float(target)
    except OverflowError as exc:
        raise ValueError(
            "target must be convertible to a finite float"
        ) from exc
    try:
        coef_value = float(coef)
    except OverflowError as exc:
        raise ValueError(
            "coef must be convertible to a finite float"
        ) from exc
    if not math.isfinite(eta_value) or eta_value <= 0 or eta_value > 1:
        raise ValueError("eta must be finite and in (0, 1]")
    if (
        not math.isfinite(target_value)
        or target_value < 0
        or target_value > math.log(4.0)
    ):
        raise ValueError("target must be finite and in [0, log(4)]")
    if not math.isfinite(coef_value) or coef_value <= 0:
        raise ValueError("coef must be finite and positive")
    z = math.log(coef_value)
    if not math.isfinite(z):
        raise ValueError("log(coef) must remain finite")

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
    coefficients = []

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
            entropy_sum = 0.0
            for p_a in probs:
                if p_a > 0.0:
                    entropy_sum += p_a * math.log(p_a)
            entropy = -entropy_sum
            if not math.isfinite(entropy):
                raise ValueError("entropy must remain finite")
            try:
                entropy_coef = math.exp(z)
            except OverflowError as exc:
                raise ValueError(
                    "entropy coefficient must remain finite"
                ) from exc
            if not math.isfinite(entropy_coef):
                raise ValueError("entropy coefficient must remain finite")
            u = rng.random()
            cumulative = 0.0
            action = "L"
            for a, p_a in zip(actions, probs):
                cumulative += p_a
                if cumulative > u:
                    action = a
                    break
            value = v[state]
            next_state, reward, done = env.step(action)
            steps += 1
            total_reward += reward
            if done:
                delta = reward - value
            else:
                delta = reward + gamma * v[next_state] - value
            if not math.isfinite(delta):
                raise ValueError("delta must remain finite")
            for b, p_b in zip(actions, probs):
                indicator = 1.0 if b == action else 0.0
                if p_b == 0.0:
                    update = alpha * delta * (indicator - p_b)
                else:
                    update = alpha * (
                        delta * (indicator - p_b)
                        + entropy_coef
                        * p_b
                        * (-math.log(p_b) - entropy)
                    )
                if not math.isfinite(update):
                    raise ValueError("policy gradient must remain finite")
                new_h = h[(state, b)] + update
                if not math.isfinite(new_h):
                    raise ValueError("H value must remain finite")
                h[(state, b)] = new_h
            new_v = v[state] + beta * delta
            if not math.isfinite(new_v):
                raise ValueError("V value must remain finite")
            v[state] = new_v
            new_z = z + eta_value * (target_value - entropy)
            if not math.isfinite(new_z):
                raise ValueError("z value must remain finite")
            z = new_z
            if done:
                break
            state = next_state
        episode_results.append([steps, total_reward, done])
        try:
            final_coef = math.exp(z)
        except OverflowError as exc:
            raise ValueError(
                "entropy coefficient must remain finite"
            ) from exc
        if not math.isfinite(final_coef):
            raise ValueError("entropy coefficient must remain finite")
        coefficients.append(float(final_coef))

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
    return {
        "h": h_table,
        "v": v_table,
        "episodes": episode_results,
        "coefficients": coefficients,
    }


def actor_critic_lambda(
    env,
    episodes=500,
    alpha=0.05,
    beta=0.1,
    gamma=0.9,
    lambda_=0.9,
    seed=0,
    max_steps=1000,
) -> dict:
    """在线 actor-critic(λ)（累积资格迹），返回 h/v 表与逐回合统计。

    H 覆盖从 S 可达的非 G 格与 U/R/D/L 的全部组合，V 覆盖同样的格子，
    初始均为 0.0。每回合 reset，按 H/V 键序将策略迹 EH 与值迹 EV
    清零，至 done 或 max_steps 步；每步按 softmax(H[s,·]) 采样动作，
    仅调用一次 random()。step 前保留动作概率 p 与旧 V[s]，step 得
    (s2, r, done)：done 时 δ=r-V[s]，否则 δ=r+gamma*V[s2]-V[s]；
    达到步限且未 done 的末步同样按后式自举。先按键序使所有 EH、EV
    乘 gamma*lambda_，再对 URDL 各 b 作 EH[s,b]+=I[b=a]-p[b] 并作
    EV[s]+=1.0；随后按键序对所有项作 H+=alpha*δ*EH、V+=beta*δ*EV。
    δ、迹或任一更新值非有限即抛 ValueError。全部随机性来自一个
    random.Random(seed)。

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
    if not _is_finite_number(gamma) or gamma < 0 or gamma >= 1:
        raise ValueError("gamma must be finite and in [0, 1)")
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
        eh = {key: 0.0 for key in h}
        ev = {key: 0.0 for key in v}
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
            if not math.isfinite(delta):
                raise ValueError("delta must remain finite")
            decay = gamma * lambda_
            for key in eh:
                eh[key] *= decay
                if not math.isfinite(eh[key]):
                    raise ValueError("policy trace must remain finite")
            for key in ev:
                ev[key] *= decay
                if not math.isfinite(ev[key]):
                    raise ValueError("value trace must remain finite")
            for b in actions:
                indicator = 1.0 if b == action else 0.0
                eh[(state, b)] += indicator - prob[b]
                if not math.isfinite(eh[(state, b)]):
                    raise ValueError("policy trace must remain finite")
            ev[state] += 1.0
            if not math.isfinite(ev[state]):
                raise ValueError("value trace must remain finite")
            for key in h:
                h[key] += alpha * delta * eh[key]
                if not math.isfinite(h[key]):
                    raise ValueError("H value must remain finite")
            for key in v:
                v[key] += beta * delta * ev[key]
                if not math.isfinite(v[key]):
                    raise ValueError("V value must remain finite")
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


def td_lambda_prediction(
    env,
    policy,
    episodes=500,
    alpha=0.1,
    gamma=0.9,
    lambda_=0.9,
    seed=0,
    max_steps=1000,
) -> dict:
    """按给定策略做 TD(λ) 策略评估（累积资格迹），返回 V 表。

    policy 为 dict，键恰为从 S 可达的非 G 格（二元非 bool 整数 tuple），
    值为 U/R/D/L 序的四项 list，各项为有限非负 float 且行和为 1.0。
    V 覆盖从 S 可达的全部格（含 G），按坐标升序、初值均为 0.0；非 G
    累积迹 E 仅覆盖非 G 格。每回合 reset，E 清零，至 done 或 max_steps
    步。每步仅调用一次 random()：按 URDL 累计概率取首个严格大于样本的
    动作，未命中取 L。step 得 (s2, r, done)：done 时 δ=r-V[s]，否则
    δ=r+gamma*V[s2]-V[s]；达到步限且未 done 的末步同样按后式自举。
    随后 E[s]+=1，按坐标序对每个非 G 格 x 作 V[x]+=alpha*δ*E[x]、
    E[x]*=gamma*lambda_（G 格 V 值恒为 0.0）。V 或 E 非有限即抛
    ValueError。全部随机性来自一个 random.Random(seed)。
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
    if not _is_finite_number(gamma) or gamma < 0 or gamma >= 1:
        raise ValueError("gamma must be finite and in [0, 1)")
    if not _is_finite_number(lambda_) or lambda_ < 0 or lambda_ > 1:
        raise ValueError("lambda_ must be finite and in [0, 1]")

    actions = tuple(_ACTIONS)  # U, R, D, L
    states = sorted(
        state
        for state in _reachable_cells(env)
        if env._cell(state) != "G"
    )
    all_states = sorted(_reachable_cells(env))
    if not isinstance(policy, dict):
        raise TypeError("policy must be a dict")
    for key in policy:
        if (
            not isinstance(key, tuple)
            or len(key) != 2
            or any(
                isinstance(v, bool) or not isinstance(v, int) for v in key
            )
        ):
            raise TypeError(
                "policy keys must be tuples of two non-bool ints"
            )
    if set(policy) != set(states):
        raise ValueError(
            "policy keys must exactly be the reachable non-goal cells"
        )
    for state in states:
        row = policy[state]
        if not isinstance(row, list):
            raise TypeError("policy rows must be lists")
        if len(row) != 4:
            raise ValueError(
                "policy rows must have four entries in U, R, D, L order"
            )
        for probability in row:
            if isinstance(probability, bool) or not isinstance(
                probability, float
            ):
                raise TypeError("policy probabilities must be floats")
        for probability in row:
            if not math.isfinite(probability):
                raise ValueError("policy probabilities must be finite")
            if probability < 0:
                raise ValueError("policy probabilities must be non-negative")
        if sum(row, 0.0) != 1.0:
            raise ValueError("policy rows must sum to 1.0")

    v = {cell: 0.0 for cell in all_states}
    rng = random.Random(seed)

    for _ in range(episodes):
        state = env.reset()
        trace = {cell: 0.0 for cell in states}
        for _ in range(max_steps):
            sample = rng.random()
            cumulative = 0.0
            action = "L"
            for candidate, probability in zip(actions, policy[state]):
                cumulative += probability
                if cumulative > sample:
                    action = candidate
                    break
            next_state, reward, done = env.step(action)
            if done:
                delta = reward - v[state]
            else:
                delta = reward + gamma * v[next_state] - v[state]
            trace[state] += 1.0
            decay = gamma * lambda_
            for cell in states:
                v[cell] += alpha * delta * trace[cell]
                if not math.isfinite(v[cell]):
                    raise ValueError("V value must remain finite")
                trace[cell] *= decay
                if not math.isfinite(trace[cell]):
                    raise ValueError("eligibility trace must remain finite")
            if done:
                break
            state = next_state
    return v


def off_policy_td(
    env,
    behavior,
    target,
    episodes=500,
    alpha=0.1,
    gamma=0.9,
    lambda_=0.9,
    seed=0,
    max_steps=1000,
) -> dict:
    """按行为策略采样、对目标策略做离策略 TD(λ) 评估，返回 V 表。

    behavior、target 均为 dict，键恰为从 S 可达的非 G 格（二元非 bool
    整数 tuple），值为 U/R/D/L 序的四项 list，各项为有限非负 float 且
    行和为 1.0；target 取正概率处 behavior 也必须为正。余参数校验、
    V 键域与初值、G 格 V 恒 0.0、reset 与 done/max_steps 边界均同
    td_lambda_prediction。每回合非 G 格 E 清零，每步仅调用一次
    random()，按 behavior 的 URDL 累计概率取首个严格大于样本的动作，
    未命中取 L。step 得 (s2, r, done)，rho=target[s][a]/behavior[s][a]；
    done 时 δ=r-V[s]，否则 δ=r+gamma*V[s2]-V[s]（截断末步仍自举）。
    随后按坐标序同步置 E[x]=rho*(gamma*lambda_*E[x]+I[x=s])，再按
    坐标序作 V[x]+=alpha*δ*E[x]。rho、δ、E 或 V 非有限即抛
    ValueError。全部随机性来自一个 random.Random(seed)。
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
    if not _is_finite_number(gamma) or gamma < 0 or gamma >= 1:
        raise ValueError("gamma must be finite and in [0, 1)")
    if not _is_finite_number(lambda_) or lambda_ < 0 or lambda_ > 1:
        raise ValueError("lambda_ must be finite and in [0, 1]")

    actions = tuple(_ACTIONS)  # U, R, D, L
    states = sorted(
        state
        for state in _reachable_cells(env)
        if env._cell(state) != "G"
    )
    all_states = sorted(_reachable_cells(env))

    def _validate_policy(policy, name):
        if not isinstance(policy, dict):
            raise TypeError(name + " must be a dict")
        for key in policy:
            if (
                not isinstance(key, tuple)
                or len(key) != 2
                or any(
                    isinstance(v, bool) or not isinstance(v, int)
                    for v in key
                )
            ):
                raise TypeError(
                    name + " keys must be tuples of two non-bool ints"
                )
        if set(policy) != set(states):
            raise ValueError(
                name
                + " keys must exactly be the reachable non-goal cells"
            )
        for cell in states:
            row = policy[cell]
            if not isinstance(row, list):
                raise TypeError(name + " rows must be lists")
            if len(row) != 4:
                raise ValueError(
                    name
                    + " rows must have four entries in U, R, D, L order"
                )
            for probability in row:
                if isinstance(probability, bool) or not isinstance(
                    probability, float
                ):
                    raise TypeError(name + " probabilities must be floats")
            for probability in row:
                if not math.isfinite(probability):
                    raise ValueError(
                        name + " probabilities must be finite"
                    )
                if probability < 0:
                    raise ValueError(
                        name + " probabilities must be non-negative"
                    )
            if sum(row, 0.0) != 1.0:
                raise ValueError(name + " rows must sum to 1.0")

    _validate_policy(behavior, "behavior policy")
    _validate_policy(target, "target policy")
    for cell in states:
        for index in range(4):
            if target[cell][index] > 0.0 and behavior[cell][index] <= 0.0:
                raise ValueError(
                    "behavior policy must have positive probability wherever "
                    "target policy does"
                )

    v = {cell: 0.0 for cell in all_states}
    rng = random.Random(seed)
    decay = gamma * lambda_

    for _ in range(episodes):
        state = env.reset()
        trace = {cell: 0.0 for cell in states}
        for _ in range(max_steps):
            sample = rng.random()
            cumulative = 0.0
            action = "L"
            action_index = 3
            for index, probability in enumerate(behavior[state]):
                cumulative += probability
                if cumulative > sample:
                    action = actions[index]
                    action_index = index
                    break
            rho = (
                target[state][action_index]
                / behavior[state][action_index]
            )
            if not math.isfinite(rho):
                raise ValueError("importance ratio must remain finite")
            next_state, reward, done = env.step(action)
            if done:
                delta = reward - v[state]
            else:
                delta = reward + gamma * v[next_state] - v[state]
            if not math.isfinite(delta):
                raise ValueError("delta must remain finite")
            for cell in states:
                indicator = 1.0 if cell == state else 0.0
                new_trace = rho * (decay * trace[cell] + indicator)
                if not math.isfinite(new_trace):
                    raise ValueError("eligibility trace must remain finite")
                trace[cell] = new_trace
            for cell in states:
                v[cell] += alpha * delta * trace[cell]
                if not math.isfinite(v[cell]):
                    raise ValueError("V value must remain finite")
            if done:
                break
            state = next_state
    return v


def emphatic_td_lambda(
    env,
    behavior,
    target,
    interest,
    episodes=500,
    alpha=0.1,
    gamma=0.9,
    lambda_=0.9,
    seed=0,
    max_steps=1000,
) -> dict:
    """强调（emphatic）离策略 TD(λ) 策略评估，返回 V 表。

    behavior、target 与其余参数的公开契约均同 off_policy_td。interest
    为 dict，键恰为从 S 可达的非 G 格（二元非 bool 整数 tuple），值为
    非 bool 的有限 int/float 且均 ≥ 0，至少一项 > 0；类型错误抛
    TypeError，其余违约抛 ValueError。每回合非 G 格 E 清零，且
    F=ρprev=0.0。每步先算 F=interest[s]+gamma*ρprev*F、
    M=lambda_*interest[s]+(1-lambda_)*F；再仅调用一次 random() 按
    behavior 的 URDL 累计概率取首个严格大于样本的动作（未命中取
    URDL 中最后一个正概率动作），令
    rho=target[s][a]/behavior[s][a]；step 得 (s2, r, done)，done 时
    δ=r-V[s]，否则 δ=r+gamma*V[s2]-V[s]（截断末步仍自举）。随后按坐标
    序同步置 E[x]=rho*(gamma*lambda_*E[x]+(M if x==s else 0.0))，再按
    坐标序作 V[x]+=alpha*δ*E[x]。F、M、rho、δ、E 或 V 非有限即抛
    ValueError。done 即停，否则置 s,ρprev=s2,rho。全部随机性来自一个
    random.Random(seed)。
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
    if not _is_finite_number(gamma) or gamma < 0 or gamma >= 1:
        raise ValueError("gamma must be finite and in [0, 1)")
    if not _is_finite_number(lambda_) or lambda_ < 0 or lambda_ > 1:
        raise ValueError("lambda_ must be finite and in [0, 1]")

    actions = tuple(_ACTIONS)  # U, R, D, L
    states = sorted(
        state
        for state in _reachable_cells(env)
        if env._cell(state) != "G"
    )
    all_states = sorted(_reachable_cells(env))

    def _validate_policy(policy, name):
        if not isinstance(policy, dict):
            raise TypeError(name + " must be a dict")
        for key in policy:
            if (
                not isinstance(key, tuple)
                or len(key) != 2
                or any(
                    isinstance(v, bool) or not isinstance(v, int)
                    for v in key
                )
            ):
                raise TypeError(
                    name + " keys must be tuples of two non-bool ints"
                )
        if set(policy) != set(states):
            raise ValueError(
                name
                + " keys must exactly be the reachable non-goal cells"
            )
        for cell in states:
            row = policy[cell]
            if not isinstance(row, list):
                raise TypeError(name + " rows must be lists")
            if len(row) != 4:
                raise ValueError(
                    name
                    + " rows must have four entries in U, R, D, L order"
                )
            for probability in row:
                if isinstance(probability, bool) or not isinstance(
                    probability, float
                ):
                    raise TypeError(name + " probabilities must be floats")
            for probability in row:
                if not math.isfinite(probability):
                    raise ValueError(
                        name + " probabilities must be finite"
                    )
                if probability < 0:
                    raise ValueError(
                        name + " probabilities must be non-negative"
                    )
            if sum(row, 0.0) != 1.0:
                raise ValueError(name + " rows must sum to 1.0")

    _validate_policy(behavior, "behavior policy")
    _validate_policy(target, "target policy")
    for cell in states:
        for index in range(4):
            if target[cell][index] > 0.0 and behavior[cell][index] <= 0.0:
                raise ValueError(
                    "behavior policy must have positive probability wherever "
                    "target policy does"
                )

    if not isinstance(interest, dict):
        raise TypeError("interest must be a dict")
    for key in interest:
        if (
            not isinstance(key, tuple)
            or len(key) != 2
            or any(
                isinstance(v, bool) or not isinstance(v, int) for v in key
            )
        ):
            raise TypeError(
                "interest keys must be tuples of two non-bool ints"
            )
    if set(interest) != set(states):
        raise ValueError(
            "interest keys must exactly be the reachable non-goal cells"
        )
    any_positive = False
    for cell in states:
        value = interest[cell]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("interest values must be ints or floats")
        if not _is_finite_number(value):
            raise ValueError("interest values must be finite")
        if value < 0:
            raise ValueError("interest values must be non-negative")
        if value > 0:
            any_positive = True
    if not any_positive:
        raise ValueError("interest must have at least one positive value")

    v = {cell: 0.0 for cell in all_states}
    rng = random.Random(seed)
    decay = gamma * lambda_

    for _ in range(episodes):
        state = env.reset()
        trace = {cell: 0.0 for cell in states}
        followon = 0.0
        rho_prev = 0.0
        for _ in range(max_steps):
            followon = interest[state] + gamma * rho_prev * followon
            if not math.isfinite(followon):
                raise ValueError("follow-on trace must remain finite")
            emphasis = (
                lambda_ * interest[state] + (1.0 - lambda_) * followon
            )
            if not math.isfinite(emphasis):
                raise ValueError("emphasis must remain finite")
            sample = rng.random()
            cumulative = 0.0
            action_index = 3
            for index, probability in enumerate(behavior[state]):
                if probability > 0.0:
                    action_index = index
                cumulative += probability
                if cumulative > sample:
                    break
            action = actions[action_index]
            rho = (
                target[state][action_index]
                / behavior[state][action_index]
            )
            if not math.isfinite(rho):
                raise ValueError("importance ratio must remain finite")
            next_state, reward, done = env.step(action)
            if done:
                delta = reward - v[state]
            else:
                delta = reward + gamma * v[next_state] - v[state]
            if not math.isfinite(delta):
                raise ValueError("delta must remain finite")
            for cell in states:
                increment = emphasis if cell == state else 0.0
                new_trace = rho * (decay * trace[cell] + increment)
                if not math.isfinite(new_trace):
                    raise ValueError("eligibility trace must remain finite")
                trace[cell] = new_trace
            for cell in states:
                v[cell] += alpha * delta * trace[cell]
                if not math.isfinite(v[cell]):
                    raise ValueError("V value must remain finite")
            if done:
                break
            state = next_state
            rho_prev = rho
    return v


def off_policy_actor_critic(
    env,
    behavior,
    episodes=500,
    alpha=0.05,
    beta=0.1,
    gamma=0.9,
    rho_clip=1.0,
    seed=0,
    max_steps=1000,
) -> dict:
    """离策略在线一步 actor-critic，以 softmax 策略为目标、behavior 为
    行为策略，返回 h/v 表与逐回合统计。

    behavior 为 dict，单侧行为策略契约同 off_policy_td：键恰为从 S
    可达的非 G 格（二元非 bool 整数 tuple），值为 U/R/D/L 序的四项
    list，各项为有限非负 float 且行和为 1.0。episodes、alpha、beta、
    gamma、seed、max_steps 的校验，H/V 键域与初值，reset 与
    done/max_steps 边界，以及返回契约均同 actor_critic。rho_clip 为非
    bool 的有限 int/float 且 > 0，错型抛 TypeError，非有限或越界抛
    ValueError。

    每步以旧 H 按稳定 softmax 得 URDL 概率 p；仅以一个
    random.Random(seed) 调用一次 random()，按 behavior 的 URDL 累计
    概率取首个严格大于样本的动作，未命中取 URDL 中最后一个正概率动作
    （故 behavior[s][a] 必为正）。step 得 (s2, r, done)：done 时
    δ=r-V[s]，否则 δ=r+gamma*V[s2]-V[s]，达到步限且未 done 的末步同样
    按后式自举；ρ=min(rho_clip, p[a]/behavior[s][a])。先以旧 p 对各 b
    同步作 H[s,b]+=alpha*ρ*δ*(I[b=a]-p[b])，再作
    V[s]+=beta*ρ*δ。ρ、δ、梯度中间量或任一更新值非有限即抛
    ValueError。全部随机性来自一个 random.Random(seed)。

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
    if isinstance(rho_clip, bool) or not isinstance(
        rho_clip, (int, float)
    ):
        raise TypeError("rho_clip must be an int or float")
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
    if not _is_finite_number(rho_clip) or rho_clip <= 0:
        raise ValueError("rho_clip must be finite and positive")

    actions = tuple(_ACTIONS)  # U, R, D, L
    states = sorted(
        state
        for state in _reachable_cells(env)
        if env._cell(state) != "G"
    )

    if not isinstance(behavior, dict):
        raise TypeError("behavior policy must be a dict")
    for key in behavior:
        if (
            not isinstance(key, tuple)
            or len(key) != 2
            or any(
                isinstance(v, bool) or not isinstance(v, int)
                for v in key
            )
        ):
            raise TypeError(
                "behavior policy keys must be tuples of two non-bool ints"
            )
    if set(behavior) != set(states):
        raise ValueError(
            "behavior policy keys must exactly be the reachable "
            "non-goal cells"
        )
    for cell in states:
        row = behavior[cell]
        if not isinstance(row, list):
            raise TypeError("behavior policy rows must be lists")
        if len(row) != 4:
            raise ValueError(
                "behavior policy rows must have four entries in "
                "U, R, D, L order"
            )
        for probability in row:
            if isinstance(probability, bool) or not isinstance(
                probability, float
            ):
                raise TypeError(
                    "behavior policy probabilities must be floats"
                )
        for probability in row:
            if not math.isfinite(probability):
                raise ValueError(
                    "behavior policy probabilities must be finite"
                )
            if probability < 0:
                raise ValueError(
                    "behavior policy probabilities must be non-negative"
                )
        if sum(row, 0.0) != 1.0:
            raise ValueError("behavior policy rows must sum to 1.0")

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
            sample = rng.random()
            cumulative = 0.0
            action_index = 3
            for index, probability in enumerate(behavior[state]):
                if probability > 0.0:
                    action_index = index
                cumulative += probability
                if cumulative > sample:
                    break
            action = actions[action_index]
            value = v[state]
            next_state, reward, done = env.step(action)
            steps += 1
            total_reward += reward
            if done:
                delta = reward - value
            else:
                delta = reward + gamma * v[next_state] - value
            if not math.isfinite(delta):
                raise ValueError("delta must remain finite")
            rho = min(
                rho_clip,
                probs[action_index] / behavior[state][action_index],
            )
            if not math.isfinite(rho):
                raise ValueError("importance ratio must remain finite")
            for b, p_b in zip(actions, probs):
                indicator = 1.0 if b == action else 0.0
                update = alpha * rho * delta * (indicator - p_b)
                if not math.isfinite(update):
                    raise ValueError("policy gradient must remain finite")
                new_h = h[(state, b)] + update
                if not math.isfinite(new_h):
                    raise ValueError("H value must remain finite")
                h[(state, b)] = new_h
            new_v = v[state] + beta * rho * delta
            if not math.isfinite(new_v):
                raise ValueError("V value must remain finite")
            v[state] = new_v
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


def off_policy_reinforce(
    env,
    behavior,
    episodes=500,
    alpha=0.05,
    beta=0.1,
    gamma=0.9,
    rho_clip=10.0,
    seed=0,
    max_steps=1000,
) -> dict:
    """离策略 Monte Carlo REINFORCE（带状态值 V 基线），目标策略为
    softmax 偏好 H、behavior 为行为策略，返回 h/v 表与逐回合统计。

    behavior 为 dict，契约同 off_policy_actor_critic：键恰为从 S 可达的
    非 G 格（二元非 bool 整数 tuple），值为 U/R/D/L 序的四项 list，
    各项为有限 float、行和为 1.0；此外各项概率必须严格为正，零值抛
    ValueError。episodes、alpha、beta、gamma、rho_clip、seed、
    max_steps 的校验同 off_policy_actor_critic，H/V 键域与零初值、
    reset 与 done/max_steps 边界及返回契约均同 reinforce_v。

    每步以旧 H 按稳定 softmax 得 URDL 概率 p；仅以一个
    random.Random(seed) 调用一次 random()，按 behavior 的 URDL 累计
    概率取首个严格大于样本的动作，未命中取 L，并记录
    (s, a, r, p, 行为概率, 旧 V)。回合在 done 或 max_steps 结束后
    （截断不自举，截断步 done=False）皆令 G=0.0，逆序作
    G=r+gamma*G；再正序从 W=1.0 作
    W=min(rho_clip, W*p[a]/行为概率)、A=G-旧V，以所存 p 对各 b 同步
    作 H[s,b]+=alpha*W*A*(I[b=a]-p[b])，再作 V[s]+=beta*W*A。G、W、
    A 或任一更新后的 H/V 非有限即抛 ValueError。全部随机性来自一个
    random.Random(seed)。

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
    if isinstance(rho_clip, bool) or not isinstance(
        rho_clip, (int, float)
    ):
        raise TypeError("rho_clip must be an int or float")
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
    if not _is_finite_number(rho_clip) or rho_clip <= 0:
        raise ValueError("rho_clip must be finite and positive")

    actions = tuple(_ACTIONS)  # U, R, D, L
    states = sorted(
        state
        for state in _reachable_cells(env)
        if env._cell(state) != "G"
    )

    if not isinstance(behavior, dict):
        raise TypeError("behavior policy must be a dict")
    for key in behavior:
        if (
            not isinstance(key, tuple)
            or len(key) != 2
            or any(
                isinstance(v, bool) or not isinstance(v, int)
                for v in key
            )
        ):
            raise TypeError(
                "behavior policy keys must be tuples of two non-bool ints"
            )
    if set(behavior) != set(states):
        raise ValueError(
            "behavior policy keys must exactly be the reachable "
            "non-goal cells"
        )
    for cell in states:
        row = behavior[cell]
        if not isinstance(row, list):
            raise TypeError("behavior policy rows must be lists")
        if len(row) != 4:
            raise ValueError(
                "behavior policy rows must have four entries in "
                "U, R, D, L order"
            )
        for probability in row:
            if isinstance(probability, bool) or not isinstance(
                probability, float
            ):
                raise TypeError(
                    "behavior policy probabilities must be floats"
                )
        for probability in row:
            if not math.isfinite(probability):
                raise ValueError(
                    "behavior policy probabilities must be finite"
                )
            if probability <= 0.0:
                raise ValueError(
                    "behavior policy probabilities must be positive"
                )
        if sum(row, 0.0) != 1.0:
            raise ValueError("behavior policy rows must sum to 1.0")

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
            sample = rng.random()
            cumulative = 0.0
            action_index = 3
            for index, probability in enumerate(behavior[state]):
                cumulative += probability
                if cumulative > sample:
                    action_index = index
                    break
            action = actions[action_index]
            prob = dict(zip(actions, probs))
            behavior_prob = behavior[state][action_index]
            value = v[state]
            next_state, reward, done = env.step(action)
            trajectory.append(
                (state, action, reward, prob, behavior_prob, value)
            )
            total_reward += reward
            if done:
                break
            state = next_state
        episode_results.append([len(trajectory), total_reward, done])

        g = 0.0
        returns = [0.0] * len(trajectory)
        for t in range(len(trajectory) - 1, -1, -1):
            g = trajectory[t][2] + gamma * g
            if not math.isfinite(g):
                raise ValueError("return must remain finite")
            returns[t] = g

        weight = 1.0
        for t, (
            st,
            action,
            _reward,
            prob,
            behavior_prob,
            value,
        ) in enumerate(trajectory):
            weight = min(
                rho_clip,
                weight * prob[action] / behavior_prob,
            )
            if not math.isfinite(weight):
                raise ValueError("importance weight must remain finite")
            advantage = returns[t] - value
            if not math.isfinite(advantage):
                raise ValueError("advantage must remain finite")
            for b in actions:
                indicator = 1.0 if b == action else 0.0
                update = (
                    alpha
                    * weight
                    * advantage
                    * (indicator - prob[b])
                )
                if not math.isfinite(update):
                    raise ValueError("policy gradient must remain finite")
                new_h = h[(st, b)] + update
                if not math.isfinite(new_h):
                    raise ValueError("H value must remain finite")
                h[(st, b)] = new_h
            new_v = v[st] + beta * weight * advantage
            if not math.isfinite(new_v):
                raise ValueError("V value must remain finite")
            v[st] = new_v

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


def vtrace_actor_critic(
    env,
    behavior,
    episodes=500,
    alpha=0.05,
    beta=0.1,
    gamma=0.9,
    rho_clip=1.0,
    c_clip=1.0,
    seed=0,
    max_steps=1000,
) -> dict:
    """整回合 V-trace 离策略 actor-critic，以 softmax 策略为目标、
    behavior 为行为策略，返回 h/v 表与逐回合统计。

    behavior 单侧行为策略契约、episodes、alpha、beta、gamma、seed、
    max_steps 的校验，H/V 键域与初值，reset 与 done/max_steps 边界，
    以及返回契约均同 off_policy_actor_critic。rho_clip 契约亦同
    off_policy_actor_critic；c_clip 为非 bool 的有限 int/float 且 > 0，
    错型抛 TypeError，非有限或越界抛 ValueError。

    每步以旧 H 按稳定 softmax 得 URDL 概率 p；仅以一个
    random.Random(seed) 调用一次 random()，按 behavior 的 URDL 累计
    概率取首个严格大于样本的动作，未命中取 URDL 中最后一个正概率动作
    （故 behavior[s][a] 必为正）。step 得 (s2, r, done)，记录 s、a、
    r、p、b=behavior[s][a]、v=V[s]、n=0.0 if done else V[s2] 及
    term=done；仅未终止末步的 trunc 为 True。回合末调用
    segmented_vtrace(r, v, n, log(b), log(p[a]), term, trunc, gamma,
    rho_clip, c_clip) 取得 values 与 advantages，再按原步序以所存 p
    先对各 b 同步作 H[s,b]+=alpha*advantages[t]*(I[b=a]-p[b])，再作
    V[s]+=beta*(values[t]-v)。softmax、对数、优势递推中间量或任一
    更新值非有限即抛 ValueError。全部随机性来自一个
    random.Random(seed)。

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
    if isinstance(rho_clip, bool) or not isinstance(
        rho_clip, (int, float)
    ):
        raise TypeError("rho_clip must be an int or float")
    if isinstance(c_clip, bool) or not isinstance(c_clip, (int, float)):
        raise TypeError("c_clip must be an int or float")
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
    if not _is_finite_number(rho_clip) or rho_clip <= 0:
        raise ValueError("rho_clip must be finite and positive")
    if not _is_finite_number(c_clip) or c_clip <= 0:
        raise ValueError("c_clip must be finite and positive")

    actions = tuple(_ACTIONS)  # U, R, D, L
    states = sorted(
        state
        for state in _reachable_cells(env)
        if env._cell(state) != "G"
    )

    if not isinstance(behavior, dict):
        raise TypeError("behavior policy must be a dict")
    for key in behavior:
        if (
            not isinstance(key, tuple)
            or len(key) != 2
            or any(
                isinstance(v, bool) or not isinstance(v, int)
                for v in key
            )
        ):
            raise TypeError(
                "behavior policy keys must be tuples of two non-bool ints"
            )
    if set(behavior) != set(states):
        raise ValueError(
            "behavior policy keys must exactly be the reachable "
            "non-goal cells"
        )
    for cell in states:
        row = behavior[cell]
        if not isinstance(row, list):
            raise TypeError("behavior policy rows must be lists")
        if len(row) != 4:
            raise ValueError(
                "behavior policy rows must have four entries in "
                "U, R, D, L order"
            )
        for probability in row:
            if isinstance(probability, bool) or not isinstance(
                probability, float
            ):
                raise TypeError(
                    "behavior policy probabilities must be floats"
                )
        for probability in row:
            if not math.isfinite(probability):
                raise ValueError(
                    "behavior policy probabilities must be finite"
                )
            if probability < 0:
                raise ValueError(
                    "behavior policy probabilities must be non-negative"
                )
        if sum(row, 0.0) != 1.0:
            raise ValueError("behavior policy rows must sum to 1.0")

    h = {(state, action): 0.0 for state in states for action in actions}
    v = {state: 0.0 for state in states}
    rng = random.Random(seed)
    episode_results = []

    for _ in range(episodes):
        state = env.reset()
        trajectory = []
        steps = 0
        total_reward = 0
        done = False
        for _ in range(max_steps):
            m = max(h[(state, a)] for a in actions)
            weights = [math.exp(h[(state, a)] - m) for a in actions]
            total = sum(weights)
            probs = [weight / total for weight in weights]
            if not all(math.isfinite(p_b) for p_b in probs):
                raise ValueError("softmax probabilities must remain finite")
            sample = rng.random()
            cumulative = 0.0
            action_index = 3
            for index, probability in enumerate(behavior[state]):
                if probability > 0.0:
                    action_index = index
                cumulative += probability
                if cumulative > sample:
                    break
            action = actions[action_index]
            value = v[state]
            next_state, reward, done = env.step(action)
            steps += 1
            total_reward += reward
            next_value = 0.0 if done else v[next_state]
            behavior_prob = behavior[state][action_index]
            trajectory.append(
                (
                    state,
                    action,
                    reward,
                    probs,
                    probs[action_index],
                    behavior_prob,
                    value,
                    next_value,
                    bool(done),
                )
            )
            if done:
                break
            state = next_state

        truncs = [False] * len(trajectory)
        if not done:
            truncs[-1] = True
        rewards = [record[2] for record in trajectory]
        old_values = [record[6] for record in trajectory]
        next_values = [record[7] for record in trajectory]
        log_behavior = []
        log_target = []
        for record in trajectory:
            log_b = math.log(record[5])
            log_p = math.log(record[4])
            if not math.isfinite(log_b) or not math.isfinite(log_p):
                raise ValueError("log probabilities must remain finite")
            log_behavior.append(log_b)
            log_target.append(log_p)
        terms = [record[8] for record in trajectory]

        vtrace_result = segmented_vtrace(
            rewards,
            old_values,
            next_values,
            log_behavior,
            log_target,
            terms,
            truncs,
            gamma,
            rho_clip,
            c_clip,
        )
        targets = vtrace_result["values"]
        advantages = vtrace_result["advantages"]

        for t, record in enumerate(trajectory):
            st, action, _reward, probs, _p_a, _b, old_v, _n, _term = (
                record
            )
            advantage = advantages[t]
            if not math.isfinite(advantage):
                raise ValueError("advantage must remain finite")
            new_h_values = {}
            for b_action, p_b in zip(actions, probs):
                indicator = 1.0 if b_action == action else 0.0
                update = alpha * advantage * (indicator - p_b)
                if not math.isfinite(update):
                    raise ValueError("policy gradient must remain finite")
                new_h = h[(st, b_action)] + update
                if not math.isfinite(new_h):
                    raise ValueError("H value must remain finite")
                new_h_values[b_action] = new_h
            for b_action in actions:
                h[(st, b_action)] = new_h_values[b_action]
            critic_delta = targets[t] - old_v
            if not math.isfinite(critic_delta):
                raise ValueError("critic target delta must remain finite")
            new_v = v[st] + beta * critic_delta
            if not math.isfinite(new_v):
                raise ValueError("V value must remain finite")
            v[st] = new_v

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


def retrace_actor_critic(
    env,
    behavior,
    episodes=500,
    alpha=0.05,
    beta=0.1,
    gamma=0.9,
    lambda_=0.9,
    seed=0,
    max_steps=1000,
) -> dict:
    """整回合 Retrace(λ) 离策略 actor-critic，以 softmax 策略为目标、
    behavior 为行为策略，返回 h/v 表与逐回合统计。

    behavior 单侧行为策略契约、episodes、alpha、beta、gamma、seed、
    max_steps 的校验，H/V 键域与初值，reset 与 done/max_steps 边界，
    以及返回契约均同 vtrace_actor_critic。lambda_ 为非 bool 的有限
    int/float 且在 [0, 1]，错型抛 TypeError，转 float 溢出、非有限或
    越界抛 ValueError。

    每步以旧 H 按稳定 softmax 得 URDL 概率 p；仅以一个
    random.Random(seed) 调用一次 random()，按 behavior 的 URDL 累计
    概率取首个严格大于样本的动作，未命中取 URDL 中最后一个正概率动作
    （故 behavior[s][a] 必为正）。step 得 (s2, r, done)，记录 s、a、
    r、p、b=behavior[s][a]、v=V[s]、n=0.0 if done else V[s2]。
    done 即停；截断末步仍以 V[s2] 自举。回合末将各步
    [r, v, n, log(b), log(p[a]), done] 交给
    retrace(transitions, gamma, lambda_) 取得 targets 与
    advantages，再按原步序顺序令 rho=min(1.0, exp(log(p[a])-
    log(b)))，先以所存 p 对各 x 同步作
    H[s,x]+=alpha*rho*advantages[t]*(I[x=a]-p[x])，再作
    V[s]+=beta*(targets[t]-v)。softmax、对数、rho、优势递推中间量或
    任一更新值溢出或非有限即抛 ValueError。全部随机性来自一个
    random.Random(seed)。

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
    if not _is_finite_number(gamma) or gamma < 0 or gamma >= 1:
        raise ValueError("gamma must be finite and in [0, 1)")
    try:
        lambda_value = float(lambda_)
    except OverflowError:
        raise ValueError("lambda_ must convert to a finite float")
    if not math.isfinite(lambda_value):
        raise ValueError("lambda_ must be finite")
    if lambda_value < 0.0 or lambda_value > 1.0:
        raise ValueError("lambda_ must be finite and in [0, 1]")

    actions = tuple(_ACTIONS)  # U, R, D, L
    states = sorted(
        state
        for state in _reachable_cells(env)
        if env._cell(state) != "G"
    )

    if not isinstance(behavior, dict):
        raise TypeError("behavior policy must be a dict")
    for key in behavior:
        if (
            not isinstance(key, tuple)
            or len(key) != 2
            or any(
                isinstance(v, bool) or not isinstance(v, int)
                for v in key
            )
        ):
            raise TypeError(
                "behavior policy keys must be tuples of two non-bool ints"
            )
    if set(behavior) != set(states):
        raise ValueError(
            "behavior policy keys must exactly be the reachable "
            "non-goal cells"
        )
    for cell in states:
        row = behavior[cell]
        if not isinstance(row, list):
            raise TypeError("behavior policy rows must be lists")
        if len(row) != 4:
            raise ValueError(
                "behavior policy rows must have four entries in "
                "U, R, D, L order"
            )
        for probability in row:
            if isinstance(probability, bool) or not isinstance(
                probability, float
            ):
                raise TypeError(
                    "behavior policy probabilities must be floats"
                )
        for probability in row:
            if not math.isfinite(probability):
                raise ValueError(
                    "behavior policy probabilities must be finite"
                )
            if probability < 0:
                raise ValueError(
                    "behavior policy probabilities must be non-negative"
                )
        if sum(row, 0.0) != 1.0:
            raise ValueError("behavior policy rows must sum to 1.0")

    h = {(state, action): 0.0 for state in states for action in actions}
    v = {state: 0.0 for state in states}
    rng = random.Random(seed)
    episode_results = []

    for _ in range(episodes):
        state = env.reset()
        trajectory = []
        steps = 0
        total_reward = 0
        done = False
        for _ in range(max_steps):
            m = max(h[(state, a)] for a in actions)
            weights = [math.exp(h[(state, a)] - m) for a in actions]
            total = sum(weights)
            probs = [weight / total for weight in weights]
            if not all(math.isfinite(p_b) for p_b in probs):
                raise ValueError("softmax probabilities must remain finite")
            sample = rng.random()
            cumulative = 0.0
            action_index = 3
            for index, probability in enumerate(behavior[state]):
                if probability > 0.0:
                    action_index = index
                cumulative += probability
                if cumulative > sample:
                    break
            action = actions[action_index]
            value = v[state]
            next_state, reward, done = env.step(action)
            steps += 1
            total_reward += reward
            next_value = 0.0 if done else v[next_state]
            behavior_prob = behavior[state][action_index]
            trajectory.append(
                (
                    state,
                    action,
                    reward,
                    probs,
                    probs[action_index],
                    behavior_prob,
                    value,
                    next_value,
                )
            )
            if done:
                break
            state = next_state

        terms = [False] * len(trajectory)
        if done:
            terms[-1] = True
        transitions = []
        log_behavior = []
        log_target = []
        for t, record in enumerate(trajectory):
            _st, _action, reward, _probs, p_a, b, old_v, next_v = record
            log_b = math.log(b)
            log_p = math.log(p_a)
            if not math.isfinite(log_b) or not math.isfinite(log_p):
                raise ValueError("log probabilities must remain finite")
            log_behavior.append(log_b)
            log_target.append(log_p)
            transitions.append(
                [reward, old_v, next_v, log_b, log_p, terms[t]]
            )

        retrace_result = retrace(transitions, gamma, lambda_)
        targets = retrace_result["targets"]
        advantages = retrace_result["advantages"]

        for t, record in enumerate(trajectory):
            st, action, _reward, probs, _p_a, _b, old_v, _n = record
            advantage = advantages[t]
            if not math.isfinite(advantage):
                raise ValueError("advantage must remain finite")
            diff = log_target[t] - log_behavior[t]
            if not math.isfinite(diff):
                raise ValueError("log-prob difference must remain finite")
            try:
                ratio = math.exp(diff)
            except OverflowError:
                raise ValueError("importance ratio exp overflow")
            rho = min(1.0, ratio)
            if not math.isfinite(rho):
                raise ValueError("retrace rho must remain finite")
            new_h_values = {}
            for x_action, p_x in zip(actions, probs):
                indicator = 1.0 if x_action == action else 0.0
                update = alpha * rho * advantage * (indicator - p_x)
                if not math.isfinite(update):
                    raise ValueError("policy gradient must remain finite")
                new_h = h[(st, x_action)] + update
                if not math.isfinite(new_h):
                    raise ValueError("H value must remain finite")
                new_h_values[x_action] = new_h
            for x_action in actions:
                h[(st, x_action)] = new_h_values[x_action]
            critic_delta = targets[t] - old_v
            if not math.isfinite(critic_delta):
                raise ValueError("critic target delta must remain finite")
            new_v = v[st] + beta * critic_delta
            if not math.isfinite(new_v):
                raise ValueError("V value must remain finite")
            v[st] = new_v

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


def true_online_td_lambda_prediction(
    env,
    policy,
    episodes=500,
    alpha=0.1,
    gamma=0.9,
    lambda_=0.9,
    seed=0,
    max_steps=1000,
) -> dict:
    """按给定策略做 true online TD(λ) 策略评估（荷兰迹），返回 V 表。

    参数校验、V/E 键域与初值、G 格 V 恒 0.0、策略按 URDL 采样、单一
    random.Random(seed) 每步一次 random()、reset 与 done/max_steps
    边界均同 td_lambda_prediction。每回合非 G 格 E 清零，v_old=0.0。
    每步 step 前记 v=V[s]；done 时 v2=0.0，否则 v2 取更新前 V[s2]，
    故截断末步仍自举。d=r+gamma*v2-v；先按坐标序对各非 G 格 x 作
    E[x]*=gamma*lambda_，再作 E[s]+=1-alpha*E[s]；随后按坐标序作
    V[x]+=alpha*(d+v-v_old)*E[x]，再作 V[s]-=alpha*(v-v_old)。
    E、d 或任一 V 非有限即抛 ValueError。done 即停，否则置
    s、v_old 为 s2、v2。
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
    if not _is_finite_number(gamma) or gamma < 0 or gamma >= 1:
        raise ValueError("gamma must be finite and in [0, 1)")
    if not _is_finite_number(lambda_) or lambda_ < 0 or lambda_ > 1:
        raise ValueError("lambda_ must be finite and in [0, 1]")

    actions = tuple(_ACTIONS)  # U, R, D, L
    states = sorted(
        state
        for state in _reachable_cells(env)
        if env._cell(state) != "G"
    )
    all_states = sorted(_reachable_cells(env))
    if not isinstance(policy, dict):
        raise TypeError("policy must be a dict")
    for key in policy:
        if (
            not isinstance(key, tuple)
            or len(key) != 2
            or any(
                isinstance(v, bool) or not isinstance(v, int) for v in key
            )
        ):
            raise TypeError(
                "policy keys must be tuples of two non-bool ints"
            )
    if set(policy) != set(states):
        raise ValueError(
            "policy keys must exactly be the reachable non-goal cells"
        )
    for state in states:
        row = policy[state]
        if not isinstance(row, list):
            raise TypeError("policy rows must be lists")
        if len(row) != 4:
            raise ValueError(
                "policy rows must have four entries in U, R, D, L order"
            )
        for probability in row:
            if isinstance(probability, bool) or not isinstance(
                probability, float
            ):
                raise TypeError("policy probabilities must be floats")
        for probability in row:
            if not math.isfinite(probability):
                raise ValueError("policy probabilities must be finite")
            if probability < 0:
                raise ValueError("policy probabilities must be non-negative")
        if sum(row, 0.0) != 1.0:
            raise ValueError("policy rows must sum to 1.0")

    v = {cell: 0.0 for cell in all_states}
    rng = random.Random(seed)

    for _ in range(episodes):
        state = env.reset()
        trace = {cell: 0.0 for cell in states}
        v_old = 0.0
        for _ in range(max_steps):
            sample = rng.random()
            cumulative = 0.0
            action = "L"
            for candidate, probability in zip(actions, policy[state]):
                cumulative += probability
                if cumulative > sample:
                    action = candidate
                    break
            value = v[state]
            next_state, reward, done = env.step(action)
            next_value = 0.0 if done else v[next_state]
            delta = reward + gamma * next_value - value
            if not math.isfinite(delta):
                raise ValueError("TD error must remain finite")
            decay = gamma * lambda_
            for cell in states:
                trace[cell] *= decay
                if not math.isfinite(trace[cell]):
                    raise ValueError("eligibility trace must remain finite")
            trace[state] += 1.0 - alpha * trace[state]
            if not math.isfinite(trace[state]):
                raise ValueError("eligibility trace must remain finite")
            coefficient = delta + value - v_old
            for cell in states:
                v[cell] += alpha * coefficient * trace[cell]
                if not math.isfinite(v[cell]):
                    raise ValueError("V value must remain finite")
            v[state] -= alpha * (value - v_old)
            if not math.isfinite(v[state]):
                raise ValueError("V value must remain finite")
            if done:
                break
            state = next_state
            v_old = next_value
    return v


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


def vtrace(
    rewards,
    values,
    behavior,
    target,
    gamma=0.9,
    rho_clip=1.0,
    c_clip=1.0,
    bootstrap=0.0,
) -> dict:
    """V-trace 离线策略价值目标，返回固定键序 values、advantages 的 dict。

    rewards、values、behavior（行为策略对数概率）、target（目标策略
    对数概率）须为等长非空的 list/tuple，各元素为非 bool 的
    int/float；gamma、rho_clip、c_clip、bootstrap 为非 bool 的
    int/float 标量，其中 gamma 须在 [0, 1]，rho_clip、c_clip 须
    > 0。所有输入先复制并转换为 float（R、V、B、P、g、u、c、b；
    转换溢出或结果非有限均抛 ValueError），不修改原序列。逐项令
    z[t]=exp(P[t]-B[t])（上溢抛 ValueError，下溢为 0.0 合法）、
    rho[t]=min(z[t],u)、C[t]=min(z[t],c)。令 T 为长度并设
    V[T]=vs[T]=b；t 逆序计算
    d=rho[t]*(R[t]+g*V[t+1]-V[t])、
    vs[t]=V[t]+d+g*C[t]*(vs[t+1]-V[t+1])，再原序计算
    advantages[t]=rho[t]*(R[t]+g*vs[t+1]-V[t])。对数差、中间量或
    输出非有限均抛 ValueError。values 为 vs[0..T-1]，两个列表均按
    原时序排列为 float 新 list。
    """
    if not isinstance(rewards, (list, tuple)):
        raise TypeError("rewards must be a list or tuple")
    if not isinstance(values, (list, tuple)):
        raise TypeError("values must be a list or tuple")
    if not isinstance(behavior, (list, tuple)):
        raise TypeError("behavior must be a list or tuple")
    if not isinstance(target, (list, tuple)):
        raise TypeError("target must be a list or tuple")
    if len(rewards) == 0 or len(values) == 0 or len(behavior) == 0 or len(
        target
    ) == 0:
        raise ValueError(
            "rewards, values, behavior and target must be non-empty"
        )
    if not (
        len(rewards)
        == len(values)
        == len(behavior)
        == len(target)
    ):
        raise ValueError(
            "rewards, values, behavior and target must have equal length"
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

    R = [_to_float(item, "rewards") for item in rewards]
    V = [_to_float(item, "values") for item in values]
    B = [_to_float(item, "behavior") for item in behavior]
    P = [_to_float(item, "target") for item in target]

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
    u = _to_scalar(rho_clip, "rho_clip")
    c = _to_scalar(c_clip, "c_clip")
    b = _to_scalar(bootstrap, "bootstrap")
    if g < 0.0 or g > 1.0:
        raise ValueError("gamma must be in [0, 1]")
    if u <= 0.0:
        raise ValueError("rho_clip must be > 0")
    if c <= 0.0:
        raise ValueError("c_clip must be > 0")

    T = len(R)
    rho = [0.0] * T
    clipped_c = [0.0] * T
    for t in range(T):
        diff = P[t] - B[t]
        if not math.isfinite(diff):
            raise ValueError("log-prob difference must be finite")
        try:
            z = math.exp(diff)
        except OverflowError:
            raise ValueError("importance ratio exp overflow")
        rho[t] = min(z, u)
        clipped_c[t] = min(z, c)

    vs = [0.0] * T
    vs_next = b
    v_next = b
    for t in range(T - 1, -1, -1):
        delta = R[t] + g * v_next - V[t]
        if not math.isfinite(delta):
            raise ValueError("vtrace delta must be finite")
        d = rho[t] * delta
        if not math.isfinite(d):
            raise ValueError("clipped delta term must be finite")
        gap = vs_next - v_next
        if not math.isfinite(gap):
            raise ValueError("value gap must be finite")
        vs_t = V[t] + d + g * clipped_c[t] * gap
        if not math.isfinite(vs_t):
            raise ValueError("vtrace target must be finite")
        vs[t] = vs_t
        vs_next = vs_t
        v_next = V[t]

    advantages = [0.0] * T
    for t in range(T):
        vs_next = b if t == T - 1 else vs[t + 1]
        delta = R[t] + g * vs_next - V[t]
        if not math.isfinite(delta):
            raise ValueError("vtrace advantage delta must be finite")
        advantage = rho[t] * delta
        if not math.isfinite(advantage):
            raise ValueError("advantage must be finite")
        advantages[t] = advantage
    return {"values": vs, "advantages": advantages}


def retrace(transitions, gamma=0.9, lambda_=0.9) -> dict:
    """Retrace(λ) 离轨优势递推，返回固定键序 targets、advantages 的 dict。

    transitions 须为非空 list/tuple，每项为六项 list
    [r, v, n, b, p, d]：奖励 r、当前值 v、下一值 n、行为动作对数
    概率 b、目标动作对数概率 p、终止标记 d。前五项为非 bool 的
    int/float，d 为 bool；gamma、lambda_ 为非 bool 的 int/float
    标量且均须在 [0, 1]。前五项先复制并转换为 float（转换溢出或
    结果非有限均抛 ValueError），不修改输入。逐项令
    z=exp(p-b)（上溢抛 ValueError，下溢为 0.0 合法）、
    c=lambda_*min(1.0, z)。自 a=0.0、nc=0.0 逆序递推
    delta=r+gamma*(0.0 if d else n)-v、
    a=delta+gamma*(0.0 if d else nc*a)，保存优势 a 与目标 v+a，
    再令 nc=c。对数差、中间量或输出非有限均抛 ValueError。两个
    列表均按原时序排列为 float 新 list。
    """
    if not isinstance(transitions, (list, tuple)):
        raise TypeError("transitions must be a list or tuple")
    if len(transitions) == 0:
        raise ValueError("transitions must be non-empty")

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

    rows = []
    for row in transitions:
        if not isinstance(row, list):
            raise TypeError("every transition must be a list")
        if len(row) != 6:
            raise ValueError(
                "every transition must have exactly six elements"
            )
        r = _to_float(row[0], "reward")
        v = _to_float(row[1], "value")
        n = _to_float(row[2], "next_value")
        b = _to_float(row[3], "behavior")
        p = _to_float(row[4], "target")
        d = row[5]
        if not isinstance(d, bool):
            raise TypeError("done flag must be a bool")
        rows.append((r, v, n, b, p, d))

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
    if g < 0.0 or g > 1.0:
        raise ValueError("gamma must be in [0, 1]")
    if l < 0.0 or l > 1.0:
        raise ValueError("lambda_ must be in [0, 1]")

    T = len(rows)
    c_values = [0.0] * T
    for t in range(T):
        _r, _v, _n, b, p, _d = rows[t]
        diff = p - b
        if not math.isfinite(diff):
            raise ValueError("log-prob difference must be finite")
        try:
            z = math.exp(diff)
        except OverflowError:
            raise ValueError("importance ratio exp overflow")
        c_values[t] = l * min(1.0, z)
        if not math.isfinite(c_values[t]):
            raise ValueError("retrace trace-cutting coefficient must be finite")

    targets = [0.0] * T
    advantages = [0.0] * T
    a = 0.0
    nc = 0.0
    for t in range(T - 1, -1, -1):
        r, v, n, _b, _p, d = rows[t]
        delta = r + g * (0.0 if d else n) - v
        if not math.isfinite(delta):
            raise ValueError("retrace delta must be finite")
        a = delta + g * (0.0 if d else nc * a)
        if not math.isfinite(a):
            raise ValueError("advantage must be finite")
        target = v + a
        if not math.isfinite(target):
            raise ValueError("target must be finite")
        advantages[t] = a
        targets[t] = target
        nc = c_values[t]
    return {"targets": targets, "advantages": advantages}


def segmented_gae(r, v, n, term, trunc, gamma=0.9, lambda_=0.95) -> dict:
    """分段轨迹的广义优势估计，区分真正终止与时间截断。

    返回固定键序 deltas、advantages、returns 的 dict。r（奖励）、
    v（当前值）、n（后继值）、term（终止标记）、trunc（截断标记）
    须为等长非空的 list/tuple；r、v、n 各元素为非 bool 的
    int/float，term、trunc 各元素为 bool，且同一位置不得两个标记
    同时为 True。gamma、lambda_ 为非 bool 的 int/float 标量且须在
    [0, 1]。数值输入先复制并转换为 float（转换溢出或结果非有限均
    抛 ValueError），不修改原序列。按 t 逆序计算
    delta=r[t]+gamma*(0.0 if term[t] else n[t])-v[t]（真正终止不
    自举，时间截断仍以 n[t] 自举）；令后继优势初值 0.0，
    A[t]=delta+gamma*lambda_*(0.0 if term[t] or trunc[t] else
    A[t+1])，终止与截断两类边界均切断优势递推。returns[t]=A[t]
    +v[t]；中间量或输出非有限均抛 ValueError。三个列表均按原时序
    排列为 float 新 list。
    """
    for name, seq in (
        ("r", r),
        ("v", v),
        ("n", n),
        ("term", term),
        ("trunc", trunc),
    ):
        if not isinstance(seq, (list, tuple)):
            raise TypeError(f"{name} must be a list or tuple")
    if len(r) == 0:
        raise ValueError("r, v, n, term and trunc must be non-empty")
    if not (len(r) == len(v) == len(n) == len(term) == len(trunc)):
        raise ValueError(
            "r, v, n, term and trunc must have equal length"
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

    rewards = [_to_float(item, "r") for item in r]
    values = [_to_float(item, "v") for item in v]
    next_values = [_to_float(item, "n") for item in n]
    for name, seq in (("term", term), ("trunc", trunc)):
        for item in seq:
            if not isinstance(item, bool):
                raise TypeError(f"{name} must contain only bool")
    for t in range(len(term)):
        if term[t] and trunc[t]:
            raise ValueError(
                "term and trunc must not both be True at the same index"
            )

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
    if g < 0.0 or g > 1.0:
        raise ValueError("gamma must be in [0, 1]")
    if l < 0.0 or l > 1.0:
        raise ValueError("lambda_ must be in [0, 1]")

    T = len(rewards)
    deltas = [0.0] * T
    advantages = [0.0] * T
    next_advantage = 0.0
    for t in range(T - 1, -1, -1):
        delta = rewards[t] + g * (
            0.0 if term[t] else next_values[t]
        ) - values[t]
        if not math.isfinite(delta):
            raise ValueError("segmented gae delta must be finite")
        if term[t] or trunc[t]:
            advantage = delta
        else:
            advantage = delta + g * l * next_advantage
        if not math.isfinite(advantage):
            raise ValueError("advantage must be finite")
        deltas[t] = delta
        advantages[t] = advantage
        next_advantage = advantage
    returns = [0.0] * T
    for t in range(T):
        ret = advantages[t] + values[t]
        if not math.isfinite(ret):
            raise ValueError("return must be finite")
        returns[t] = ret
    return {
        "deltas": deltas,
        "advantages": advantages,
        "returns": returns,
    }


def segmented_vtrace(
    r, v, n, b, p, term, trunc, gamma=0.9, rho_clip=1.0, c_clip=1.0
) -> dict:
    """分段轨迹的 V-trace 离轨价值目标，区分真正终止与时间截断。

    返回固定键序 values、advantages 的 dict。r（奖励）、v（当前值）、
    n（后继值）、b（行为策略对数概率）、p（目标策略对数概率）、
    term（终止标记）、trunc（截断标记）须为等长非空的 list/tuple；
    r、v、n、b、p 各元素为非 bool 的 int/float，term、trunc 各元素
    为 bool；同一位置不得两个标记同时为 True，且末位两个标记不得
    同时为 False（轨迹必须终于终止或截断边界）。gamma、rho_clip、
    c_clip 为非 bool 的 int/float 标量，其中 gamma 须在 [0, 1]，
    rho_clip、c_clip 须 > 0。数值输入先复制并转换为 float（转换
    溢出或结果非有限均抛 ValueError），不修改原序列。逐项令
    z[t]=exp(p[t]-b[t])（上溢抛 ValueError，下溢为 0.0 合法）、
    rho[t]=min(z[t],rho_clip)、c[t]=min(z[t],c_clip)。逆序令
    base=0.0 if term[t] else n[t]、
    d=rho[t]*(r[t]+gamma*base-v[t])；term[t] 或 trunc[t] 时
    x[t]=v[t]+d，否则
    x[t]=v[t]+d+gamma*c[t]*(x[t+1]-n[t])。再令
    boot=0.0 if term[t]、n[t] if trunc[t]、否则 x[t+1]，
    A[t]=rho[t]*(r[t]+gamma*boot-v[t])。对数差、中间量或输出非
    有限均抛 ValueError。values 为 x[0..T-1]，两个列表均按原时序
    排列为 float 新 list。
    """
    for name, seq in (
        ("r", r),
        ("v", v),
        ("n", n),
        ("b", b),
        ("p", p),
        ("term", term),
        ("trunc", trunc),
    ):
        if not isinstance(seq, (list, tuple)):
            raise TypeError(f"{name} must be a list or tuple")
    if len(r) == 0:
        raise ValueError(
            "r, v, n, b, p, term and trunc must be non-empty"
        )
    if not (
        len(r)
        == len(v)
        == len(n)
        == len(b)
        == len(p)
        == len(term)
        == len(trunc)
    ):
        raise ValueError(
            "r, v, n, b, p, term and trunc must have equal length"
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

    rewards = [_to_float(item, "r") for item in r]
    values = [_to_float(item, "v") for item in v]
    next_values = [_to_float(item, "n") for item in n]
    behavior = [_to_float(item, "b") for item in b]
    target = [_to_float(item, "p") for item in p]
    for name, seq in (("term", term), ("trunc", trunc)):
        for item in seq:
            if not isinstance(item, bool):
                raise TypeError(f"{name} must contain only bool")
    for t in range(len(term)):
        if term[t] and trunc[t]:
            raise ValueError(
                "term and trunc must not both be True at the same index"
            )
    if not (term[-1] or trunc[-1]):
        raise ValueError(
            "term and trunc must not both be False at the last index"
        )

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
    u = _to_scalar(rho_clip, "rho_clip")
    clip_c = _to_scalar(c_clip, "c_clip")
    if g < 0.0 or g > 1.0:
        raise ValueError("gamma must be in [0, 1]")
    if u <= 0.0:
        raise ValueError("rho_clip must be > 0")
    if clip_c <= 0.0:
        raise ValueError("c_clip must be > 0")

    T = len(rewards)
    rho = [0.0] * T
    c_values = [0.0] * T
    for t in range(T):
        diff = target[t] - behavior[t]
        if not math.isfinite(diff):
            raise ValueError("log-prob difference must be finite")
        try:
            z = math.exp(diff)
        except OverflowError:
            raise ValueError("importance ratio exp overflow")
        rho[t] = min(z, u)
        c_values[t] = min(z, clip_c)

    x = [0.0] * T
    for t in range(T - 1, -1, -1):
        base = 0.0 if term[t] else next_values[t]
        delta = rewards[t] + g * base - values[t]
        if not math.isfinite(delta):
            raise ValueError("segmented vtrace delta must be finite")
        d = rho[t] * delta
        if not math.isfinite(d):
            raise ValueError("clipped delta term must be finite")
        if term[t] or trunc[t]:
            x_t = values[t] + d
        else:
            gap = x[t + 1] - next_values[t]
            if not math.isfinite(gap):
                raise ValueError("value gap must be finite")
            x_t = values[t] + d + g * c_values[t] * gap
        if not math.isfinite(x_t):
            raise ValueError("segmented vtrace target must be finite")
        x[t] = x_t

    advantages = [0.0] * T
    for t in range(T):
        if term[t]:
            boot = 0.0
        elif trunc[t]:
            boot = next_values[t]
        else:
            boot = x[t + 1]
        delta = rewards[t] + g * boot - values[t]
        if not math.isfinite(delta):
            raise ValueError("segmented vtrace advantage delta must be finite")
        advantage = rho[t] * delta
        if not math.isfinite(advantage):
            raise ValueError("advantage must be finite")
        advantages[t] = advantage
    return {"values": x, "advantages": advantages}


def categorical_projection(
    rewards, dones, next_probs, gamma=0.99, v_min=-10.0, v_max=10.0, atoms=51
) -> dict:
    """C51 分类分布投影，返回固定键序 support、probabilities 的 dict。

    rewards、dones、next_probs 须为等长非空的 list/tuple：rewards 各
    元素为非 bool 的有限 int/float；dones 各元素为 bool；next_probs
    每行须为 list/tuple 且恰有 atoms 项，各项为非 bool 的有限
    int/float 且非负，每行按 sum(row, 0.0) 计算的行和须恰为 1.0。
    gamma、v_min、v_max 为非 bool 的 int/float 标量：gamma 须在
    [0, 1]，v_min < v_max；atoms 为非 bool 的 int 且 >= 2。类型不
    符抛 TypeError，空表、长度/形状不符、非有限、越界或行和不符抛
    ValueError。所有数值先复制并转换为 float（转换溢出或结果非有限
    均抛 ValueError），不修改原序列。令
    delta=(v_max-v_min)/(atoms-1)、support[i]=v_min+i*delta。逐 t、
    逐 j 令 z=clip(reward+(0.0 if dones[t] else gamma*support[j]))、
    b=(z-v_min)/delta、l=floor(b)、u=ceil(b)；l==u 时向该格累加 p，
    否则先后累加 p*(u-b)、p*(b-l)，各行从 0.0 开始累加。任何中间
    运算或输出非有限均抛 ValueError。support 为按 i 序的 float
    列表，probabilities 为按 t 序的 float 行列表，计算顺序固定。
    """
    if not isinstance(rewards, (list, tuple)):
        raise TypeError("rewards must be a list or tuple")
    if not isinstance(dones, (list, tuple)):
        raise TypeError("dones must be a list or tuple")
    if not isinstance(next_probs, (list, tuple)):
        raise TypeError("next_probs must be a list or tuple")
    if len(rewards) == 0 or len(dones) == 0 or len(next_probs) == 0:
        raise ValueError("rewards, dones and next_probs must be non-empty")
    if not (len(rewards) == len(dones) == len(next_probs)):
        raise ValueError(
            "rewards, dones and next_probs must have equal length"
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

    if isinstance(atoms, bool) or not isinstance(atoms, int):
        raise TypeError("atoms must be an int")
    if atoms < 2:
        raise ValueError("atoms must be >= 2")

    g = _to_scalar(gamma, "gamma")
    lo = _to_scalar(v_min, "v_min")
    hi = _to_scalar(v_max, "v_max")
    if g < 0.0 or g > 1.0:
        raise ValueError("gamma must be in [0, 1]")
    if not lo < hi:
        raise ValueError("v_min must be less than v_max")

    r = [_to_float(item, "rewards") for item in rewards]

    d_flags = []
    for item in dones:
        if not isinstance(item, bool):
            raise TypeError("dones must contain only bool")
        d_flags.append(item)

    probs = []
    for row in next_probs:
        if not isinstance(row, (list, tuple)):
            raise TypeError(
                "every row of next_probs must be a list or tuple"
            )
        if len(row) != atoms:
            raise ValueError(
                "every row of next_probs must have exactly atoms items"
            )
        values = [_to_float(item, "next_probs") for item in row]
        for value in values:
            if value < 0.0:
                raise ValueError("next_probs must be non-negative")
        total = sum(values, 0.0)
        if not math.isfinite(total) or total != 1.0:
            raise ValueError("each row of next_probs must sum to 1.0")
        probs.append(values)

    delta = (hi - lo) / (atoms - 1)
    if not math.isfinite(delta):
        raise ValueError("support spacing must be finite")
    support = []
    for i in range(atoms):
        node = lo + i * delta
        if not math.isfinite(node):
            raise ValueError("support must be finite")
        support.append(node)

    T = len(r)
    projected = []
    for t in range(T):
        row = [0.0] * atoms
        reward = r[t]
        done = d_flags[t]
        for j in range(atoms):
            p = probs[t][j]
            if done:
                z = reward
            else:
                z = reward + g * support[j]
            if not math.isfinite(z):
                raise ValueError("projected Bellman value must be finite")
            if z < lo:
                z = lo
            elif z > hi:
                z = hi
            b = (z - lo) / delta
            if not math.isfinite(b):
                raise ValueError("projection position must be finite")
            l = math.floor(b)
            u = math.ceil(b)
            if l == u:
                k = min(max(l, 0), atoms - 1)
                row[k] += p
                if not math.isfinite(row[k]):
                    raise ValueError("projected probability must be finite")
            else:
                kl = min(max(l, 0), atoms - 1)
                ku = min(max(u, 0), atoms - 1)
                lower = p * (u - b)
                if not math.isfinite(lower):
                    raise ValueError("projection weight must be finite")
                row[kl] += lower
                if not math.isfinite(row[kl]):
                    raise ValueError("projected probability must be finite")
                upper = p * (b - l)
                if not math.isfinite(upper):
                    raise ValueError("projection weight must be finite")
                row[ku] += upper
                if not math.isfinite(row[ku]):
                    raise ValueError("projected probability must be finite")
        for value in row:
            if not math.isfinite(value):
                raise ValueError("projected probabilities must be finite")
        projected.append(row)
    return {"support": support, "probabilities": projected}


def categorical_q_learning(
    env,
    episodes=500,
    alpha=0.5,
    gamma=0.99,
    epsilon=0.1,
    v_min=-10.0,
    v_max=10.0,
    atoms=51,
    seed=0,
    max_steps=1000,
) -> dict:
    """C51 式分类 Q-learning，返回 {"q": ..., "distributions": ...}。

    支撑与投影规则同 categorical_projection：delta=(v_max-v_min)/(atoms-1)，
    support[i]=v_min+i*delta，且 0 须落在 [v_min, v_max] 内。分布覆盖从 S
    可达的非 G 格与 U/R/D/L 的全部组合，初始时距 0 最近的 atom 置 1
    （等距取较小索引），其余为 0.0。每回合 reset，单回合最多 max_steps
    步；全部随机性来自一个 random.Random(seed)。

    每步先调一次 random()：其值 < epsilon 时按 URDL 调 randrange(4)
    取探索动作，否则取期望 Q（按 atom 序从 0.0 累加 support[i]*p[i]）
    最大且 URDL 中首个的动作。step 后 done 时目标为 reward 的单点投
    影，否则取下一状态期望 Q 最大动作的分布，按
    reward+gamma*support[j] 投影；投影行从 0.0 累加，随后逐 atom 执
    行 p+=alpha*(target-p)。任何中间或输出值非有限抛 ValueError。
    done 即停；到达步限且未终止的末步仍照常更新。返回 q 为
    {((row, col), action): float}，distributions 为
    {((row, col), action): [float]}，均按状态坐标升序、动作 URDL
    键序。
    """
    if not isinstance(env, GridWorld):
        raise TypeError("env must be a GridWorld")
    if isinstance(episodes, bool) or not isinstance(episodes, int):
        raise TypeError("episodes must be an int")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an int")
    if isinstance(atoms, bool) or not isinstance(atoms, int):
        raise TypeError("atoms must be an int")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int):
        raise TypeError("max_steps must be an int")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise TypeError("alpha must be an int or float")
    if isinstance(gamma, bool) or not isinstance(gamma, (int, float)):
        raise TypeError("gamma must be an int or float")
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)):
        raise TypeError("epsilon must be an int or float")
    if isinstance(v_min, bool) or not isinstance(v_min, (int, float)):
        raise TypeError("v_min must be an int or float")
    if isinstance(v_max, bool) or not isinstance(v_max, (int, float)):
        raise TypeError("v_max must be an int or float")
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if atoms < 2:
        raise ValueError("atoms must be >= 2")
    if not _is_finite_number(alpha) or alpha <= 0 or alpha > 1:
        raise ValueError("alpha must be finite and in (0, 1]")
    if not _is_finite_number(gamma) or gamma < 0 or gamma >= 1:
        raise ValueError("gamma must be finite and in [0, 1)")
    if not _is_finite_number(epsilon) or epsilon < 0 or epsilon > 1:
        raise ValueError("epsilon must be finite and in [0, 1]")

    def _to_bound(item, name):
        try:
            result = float(item)
        except OverflowError:
            raise ValueError(f"{name} must convert to a finite float")
        if not math.isfinite(result):
            raise ValueError(f"{name} must be finite")
        return result

    lo = _to_bound(v_min, "v_min")
    hi = _to_bound(v_max, "v_max")
    if not lo < hi:
        raise ValueError("v_min must be less than v_max")
    if not lo <= 0.0 <= hi:
        raise ValueError("0 must be within [v_min, v_max]")

    delta = (hi - lo) / (atoms - 1)
    if not math.isfinite(delta):
        raise ValueError("support spacing must be finite")
    support = []
    for i in range(atoms):
        node = lo + i * delta
        if not math.isfinite(node):
            raise ValueError("support must be finite")
        support.append(node)

    actions = tuple(_ACTIONS)  # U, R, D, L
    states = sorted(
        state
        for state in _reachable_cells(env)
        if env._cell(state) != "G"
    )
    nearest = 0
    for i in range(1, atoms):
        if abs(support[i]) < abs(support[nearest]):
            nearest = i
    distributions = {}
    for state in states:
        for action in actions:
            probs = [0.0] * atoms
            probs[nearest] = 1.0
            distributions[(state, action)] = probs
    rng = random.Random(seed)

    def expected_q(probs):
        total = 0.0
        for i in range(atoms):
            total += support[i] * probs[i]
        return total

    for _ in range(episodes):
        state = env.reset()
        for _ in range(max_steps):
            if rng.random() < epsilon:
                action = actions[rng.randrange(4)]
            else:
                action = max(
                    actions,
                    key=lambda a: expected_q(distributions[(state, a)]),
                )
            next_state, reward, done = env.step(action)
            target = [0.0] * atoms
            if done:
                sources = ((float(reward), 1.0),)
            else:
                best = max(
                    actions,
                    key=lambda a: expected_q(
                        distributions[(next_state, a)]
                    ),
                )
                next_probs = distributions[(next_state, best)]
                sources = tuple(
                    (float(reward) + gamma * support[j], next_probs[j])
                    for j in range(atoms)
                )
            for z, p in sources:
                if not math.isfinite(z):
                    raise ValueError("projected Bellman value must be finite")
                if z < lo:
                    z = lo
                elif z > hi:
                    z = hi
                b = (z - lo) / delta
                if not math.isfinite(b):
                    raise ValueError("projection position must be finite")
                l = math.floor(b)
                u = math.ceil(b)
                if l == u:
                    k = min(max(l, 0), atoms - 1)
                    target[k] += p
                    if not math.isfinite(target[k]):
                        raise ValueError(
                            "projected probability must be finite"
                        )
                else:
                    kl = min(max(l, 0), atoms - 1)
                    ku = min(max(u, 0), atoms - 1)
                    lower = p * (u - b)
                    if not math.isfinite(lower):
                        raise ValueError("projection weight must be finite")
                    target[kl] += lower
                    if not math.isfinite(target[kl]):
                        raise ValueError(
                            "projected probability must be finite"
                        )
                    upper = p * (b - l)
                    if not math.isfinite(upper):
                        raise ValueError("projection weight must be finite")
                    target[ku] += upper
                    if not math.isfinite(target[ku]):
                        raise ValueError(
                            "projected probability must be finite"
                        )
            probs = distributions[(state, action)]
            for i in range(atoms):
                new_p = probs[i] + alpha * (target[i] - probs[i])
                if not math.isfinite(new_p):
                    raise ValueError("probabilities must remain finite")
                probs[i] = new_p
            if done:
                break
            state = next_state

    q = {}
    out_distributions = {}
    for state in states:
        for action in actions:
            probs = distributions[(state, action)]
            value = 0.0
            for i in range(atoms):
                value += support[i] * probs[i]
            if not math.isfinite(value):
                raise ValueError("Q value must be finite")
            q[(state, action)] = float(value)
            out_distributions[(state, action)] = [float(p) for p in probs]
    return {"q": q, "distributions": out_distributions}


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


def ppo_value_loss(old, new, returns, clip=0.2) -> dict:
    """PPO 截断值函数损失，返回固定键序 clipped、losses、mean 的 dict。

    old、new、returns 须为等长非空的 list/tuple，各元素为非 bool 的
    int/float；clip 为非 bool 的 int/float 标量且须在 [0, 1)。容器或
    元素类型错抛 TypeError，空或长度不等抛 ValueError。全部类型与结构
    校验通过后再复制并转换为 float 列表 o、n、r 及标量 e（转 float
    溢出或结果非有限抛 ValueError），不修改原序列。按 i 升序计算
    d=n[i]-o[i]、q=o[i]+min(max(d,-e),e)、u=(n[i]-r[i])**2、
    v=(q-r[i])**2、z=0.5*max(u,v)；任一运算溢出或 d、q、u、v、z
    非有限均抛 ValueError。total 自 0.0 起按 i 升序累加 z，total 或
    total/len(losses) 非有限抛 ValueError。clipped、losses 依次为 q、
    z 的 float 新列表，mean 为上述 float 均值；不改写负零，同输入
    逐值一致。
    """
    if not isinstance(old, (list, tuple)):
        raise TypeError("old must be a list or tuple")
    if not isinstance(new, (list, tuple)):
        raise TypeError("new must be a list or tuple")
    if not isinstance(returns, (list, tuple)):
        raise TypeError("returns must be a list or tuple")
    if len(old) == 0 or len(new) == 0 or len(returns) == 0:
        raise ValueError("old, new and returns must be non-empty")
    if not len(old) == len(new) == len(returns):
        raise ValueError("old, new and returns must have equal length")
    for name, seq in (("old", old), ("new", new), ("returns", returns)):
        for item in seq:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise TypeError(f"{name} must contain only int or float")
    if isinstance(clip, bool) or not isinstance(clip, (int, float)):
        raise TypeError("clip must be an int or float")

    def _to_float(item, name):
        try:
            result = float(item)
        except OverflowError:
            raise ValueError(f"{name} must convert to a finite float")
        if not math.isfinite(result):
            raise ValueError(f"{name} must contain only finite numbers")
        return result

    o = [_to_float(item, "old") for item in old]
    n = [_to_float(item, "new") for item in new]
    r = [_to_float(item, "returns") for item in returns]
    try:
        e = float(clip)
    except OverflowError:
        raise ValueError("clip must convert to a finite float")
    if not math.isfinite(e):
        raise ValueError("clip must be finite")
    if e < 0.0 or e >= 1.0:
        raise ValueError("clip must be in [0, 1)")

    clipped = []
    losses = []
    for i in range(len(o)):
        try:
            d = n[i] - o[i]
        except OverflowError:
            raise ValueError("value difference overflow")
        if not math.isfinite(d):
            raise ValueError("value difference must be finite")
        try:
            q = o[i] + min(max(d, -e), e)
            u = (n[i] - r[i]) ** 2
            v = (q - r[i]) ** 2
            z = 0.5 * max(u, v)
        except OverflowError:
            raise ValueError("value loss terms must remain finite")
        if not math.isfinite(q):
            raise ValueError("clipped value must be finite")
        if not math.isfinite(u) or not math.isfinite(v):
            raise ValueError("squared value terms must be finite")
        if not math.isfinite(z):
            raise ValueError("value loss must be finite")
        clipped.append(q)
        losses.append(z)

    total = 0.0
    for z in losses:
        try:
            total += z
        except OverflowError:
            raise ValueError("loss sum overflow")
        if not math.isfinite(total):
            raise ValueError("loss sum must be finite")
    mean = total / len(losses)
    if not math.isfinite(mean):
        raise ValueError("mean must be finite")
    return {"clipped": clipped, "losses": losses, "mean": mean}


def ppo_value_update(values, batch, lr=0.1, clip=0.2, epochs=4) -> dict:
    """PPO 截断值函数多轮更新，返回固定键序 values 的 dict。

    values 须为非空 list 且仅含有限 float；batch 须为非空 list，每
    项恰为三项 list [s, o, t]，其中 s 为非 bool 的 int 且为 values
    的有效索引，o、t 为有限 float；lr、clip 为有限 float 且依次属
    于 (0, 1]、[0, 1)；epochs 为非 bool 正 int。容器、项或标量类
    型错抛 TypeError，空、行长、索引、范围或有限性错抛 ValueError。
    先全量校验再复制 values 为 V，不修改输入。每轮冻结 X 为 V 的副
    本，g 为同长全 0.0 列表；按 batch 序算 d=X[s]-o、
    q=o+min(max(d,-clip),clip)、u=(X[s]-t)**2、v=(q-t)**2。若
    u>=v 则 a=X[s]-t；否则仅当 -clip<d<clip 时 a=q-t，其余 a=0.0；
    按序累加 g[s]。全批后按索引同步令 V[i]=X[i]-lr*g[i]/len(batch)。
    任一运算溢出或中间量、新值非有限均抛 ValueError，不返回部分结
    果。返回键仅 values，值为最终 float 新 list；同输入逐值一致。
    """
    if not isinstance(values, list):
        raise TypeError("values must be a list")
    if not values:
        raise ValueError("values must be non-empty")
    for item in values:
        if not isinstance(item, float):
            raise TypeError("values must contain only float")
        if not math.isfinite(item):
            raise ValueError("values must contain only finite float")
    n = len(values)

    if not isinstance(batch, list):
        raise TypeError("batch must be a list")
    if not batch:
        raise ValueError("batch must be non-empty")
    samples = []
    for item in batch:
        if not isinstance(item, list):
            raise TypeError("every batch item must be a list")
        if len(item) != 3:
            raise ValueError("every batch item must have exactly three elements")
        s, o, t = item
        if isinstance(s, bool) or not isinstance(s, int):
            raise TypeError("state index must be a non-bool int")
        if not 0 <= s < n:
            raise ValueError("state index out of range")
        if not isinstance(o, float):
            raise TypeError("old value must be a float")
        if not math.isfinite(o):
            raise ValueError("old value must be finite")
        if not isinstance(t, float):
            raise TypeError("target value must be a float")
        if not math.isfinite(t):
            raise ValueError("target value must be finite")
        samples.append((s, o, t))

    if not isinstance(lr, float):
        raise TypeError("lr must be a float")
    if not math.isfinite(lr):
        raise ValueError("lr must be finite")
    if lr <= 0.0 or lr > 1.0:
        raise ValueError("lr must be in (0, 1]")
    if not isinstance(clip, float):
        raise TypeError("clip must be a float")
    if not math.isfinite(clip):
        raise ValueError("clip must be finite")
    if clip < 0.0 or clip >= 1.0:
        raise ValueError("clip must be in [0, 1)")
    if isinstance(epochs, bool) or not isinstance(epochs, int):
        raise TypeError("epochs must be a non-bool int")
    if epochs <= 0:
        raise ValueError("epochs must be positive")

    V = list(values)
    n_batch = len(samples)
    for _ in range(epochs):
        X = list(V)
        g = [0.0] * n
        for s, o, t in samples:
            x = X[s]
            try:
                d = x - o
            except OverflowError:
                raise ValueError("value difference overflow")
            if not math.isfinite(d):
                raise ValueError("value difference must be finite")
            try:
                q = o + min(max(d, -clip), clip)
                u = (x - t) ** 2
                v = (q - t) ** 2
            except OverflowError:
                raise ValueError("clipped value terms must remain finite")
            if not math.isfinite(q):
                raise ValueError("clipped value must be finite")
            if not math.isfinite(u) or not math.isfinite(v):
                raise ValueError("squared value terms must be finite")
            if u >= v:
                a = x - t
            elif -clip < d < clip:
                a = q - t
            else:
                a = 0.0
            if not math.isfinite(a):
                raise ValueError("gradient term must be finite")
            try:
                g[s] += a
            except OverflowError:
                raise ValueError("gradient sum overflow")
            if not math.isfinite(g[s]):
                raise ValueError("gradient sum must be finite")
        for i in range(n):
            try:
                step = lr * g[i] / n_batch
                new_value = X[i] - step
            except OverflowError:
                raise ValueError("value update overflow")
            if not math.isfinite(step):
                raise ValueError("update step must be finite")
            if not math.isfinite(new_value):
                raise ValueError("updated values must be finite")
            V[i] = new_value
    return {"values": V}


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


def advantage_weighted_update(
    logits, batch, lr=0.05, temp=1.0, cap=20.0, epochs=4
) -> dict:
    """优势加权策略更新，返回固定键序 logits、objectives、probabilities、
    weights 的 dict。

    logits 须为非空矩形 list，每行为非空 list 且元素为有限 float；
    batch 须为非空 list，每项恰为三项 list [s, a, A]，其中 s、a 为
    非 bool 的 int 且分别为有效行、列索引，A 为有限 float；lr、temp、
    cap 为有限 float，lr 属于 (0, 1]，temp、cap 均大于 0；epochs 为
    非 bool 正 int。容器、成员或标量类型不符抛 TypeError，空、形状、
    索引、非有限或越界抛 ValueError。全量校验后复制 L。每轮冻结 L，
    按列序取 m=max(L) 算稳定 log-softmax：
    q_j=L_j-m-log(Σexp(L_k-m))、p_j=exp(q_j)；令 x=A/temp，当
    x>=log(cap) 时 w=cap，否则 w=exp(x)。按 batch、列序从 0.0 累加
    w*(I[j==a]-p_j)，除批长后同步 L+=lr*g；目标均值为
    sum(w*q[a], 0.0)/批长。运算溢出或中间量、新值非有限均抛
    ValueError。返回最终 L、逐轮目标均值 list、最终 softmax 概率矩阵、
    按 batch 序的权重 list，数值均为 float 且为新容器；不修改输入，
    同输入逐值一致。
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
        if len(item) != 3:
            raise ValueError(
                "every batch item must have exactly three elements"
            )
        s, a, adv = item
        if isinstance(s, bool) or not isinstance(s, int):
            raise TypeError("state index must be a non-bool int")
        if isinstance(a, bool) or not isinstance(a, int):
            raise TypeError("action index must be a non-bool int")
        if not 0 <= s < n_rows:
            raise ValueError("state index out of range")
        if not 0 <= a < n_cols:
            raise ValueError("action index out of range")
        if not isinstance(adv, float):
            raise TypeError("advantage must be a float")
        if not math.isfinite(adv):
            raise ValueError("advantage must be finite")
        samples.append((s, a, adv))

    if not isinstance(lr, float):
        raise TypeError("lr must be a float")
    if not math.isfinite(lr):
        raise ValueError("lr must be finite")
    if lr <= 0.0 or lr > 1.0:
        raise ValueError("lr must be in (0, 1]")
    if not isinstance(temp, float):
        raise TypeError("temp must be a float")
    if not math.isfinite(temp):
        raise ValueError("temp must be finite")
    if temp <= 0.0:
        raise ValueError("temp must be positive")
    if not isinstance(cap, float):
        raise TypeError("cap must be a float")
    if not math.isfinite(cap):
        raise ValueError("cap must be finite")
    if cap <= 0.0:
        raise ValueError("cap must be positive")
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

    log_cap = math.log(cap)
    weights = []
    for _s, _a, adv in samples:
        x = adv / temp
        if not math.isfinite(x):
            raise ValueError("scaled advantage must be finite")
        if x >= log_cap:
            w = cap
        else:
            w = math.exp(x)
        if not math.isfinite(w):
            raise ValueError("weight must be finite")
        weights.append(w)

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
        objective_sum = 0.0
        for (s, a, _adv), w in zip(samples, weights):
            objective_sum += w * qs[s][a]
            if not math.isfinite(objective_sum):
                raise ValueError("objective sum must be finite")
            row_g = g[s]
            p_row = ps[s]
            for j in range(n_cols):
                row_g[j] += w * ((1.0 if j == a else 0.0) - p_row[j])
                if not math.isfinite(row_g[j]):
                    raise ValueError("gradient must be finite")
        mean_objective = objective_sum / n_batch
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
        "weights": weights,
    }


def natural_gradient(
    logits, batch, lr=0.05, damping=0.001, tol=1e-10, max_iter=100
) -> dict:
    """自然梯度 logits 更新（状态块对角 Fisher 矩阵 + 共轭梯度），返回键 logits 的 dict。

    logits、lr 沿用 ppo_update 契约：logits 须为非空矩形 list，每行
    为非空 list 且元素为有限 float；lr 为有限 float 且属于 (0, 1]。
    batch 须为非空 list，每项恰为三项 list [s, a, A]，s、a 为非 bool
    的 int 且分别为有效行、列索引，A 为有限 float；容器或字段错型抛
    TypeError，空表、行长、索引越界、非有限抛 ValueError。damping、
    tol 须为非 bool 的 int/float，转换为 float 后有限且大于 0；
    max_iter 须为非 bool 的正 int。错型抛 TypeError，转换溢出或取值
    非法抛 ValueError。先复制 logits 冻结，各行以稳定 softmax 求 p。
    按 batch、列序从 0.0 累加 g[s,j]=ΣA(I[j==a]-p[s,j])/N；状态 s
    的 Fisher 块为 F_s=damping*I+n_s/N*(diag(p_s)-p_s*p_s^T)，n_s
    为该状态样本数。行列展平后以共轭梯度求解 Fx=g：置 x=0、r=g、
    d=r，||r||≤tol 时不迭代；否则每轮令 ρ=r·r、q=F(d)、
    a=ρ/(d·q)、x+=a*d、r-=a*q，残差达阈值即停，否则
    d=r+(r·r/ρ)*d。max_iter 轮仍未达阈值抛 RuntimeError；分母不
    大于 0 或任一运算结果非有限抛 ValueError。返回
    {"logits": 旧值+lr*x}，数值均为 float；不修改输入，同输入逐值
    一致。
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
        if len(item) != 3:
            raise ValueError("every batch item must have exactly three elements")
        s, a, adv = item
        if isinstance(s, bool) or not isinstance(s, int):
            raise TypeError("state index must be a non-bool int")
        if isinstance(a, bool) or not isinstance(a, int):
            raise TypeError("action index must be a non-bool int")
        if not 0 <= s < n_rows:
            raise ValueError("state index out of range")
        if not 0 <= a < n_cols:
            raise ValueError("action index out of range")
        if not isinstance(adv, float):
            raise TypeError("advantage must be a float")
        if not math.isfinite(adv):
            raise ValueError("advantage must be finite")
        samples.append((s, a, adv))

    if not isinstance(lr, float):
        raise TypeError("lr must be a float")
    if not math.isfinite(lr):
        raise ValueError("lr must be finite")
    if lr <= 0.0 or lr > 1.0:
        raise ValueError("lr must be in (0, 1]")
    if isinstance(damping, bool) or not isinstance(damping, (int, float)):
        raise TypeError("damping must be a non-bool int or float")
    try:
        damping = float(damping)
    except OverflowError:
        raise ValueError("damping must convert to a finite float")
    if not math.isfinite(damping) or damping <= 0.0:
        raise ValueError("damping must be finite and positive")
    if isinstance(tol, bool) or not isinstance(tol, (int, float)):
        raise TypeError("tol must be a non-bool int or float")
    try:
        tol = float(tol)
    except OverflowError:
        raise ValueError("tol must convert to a finite float")
    if not math.isfinite(tol) or tol <= 0.0:
        raise ValueError("tol must be finite and positive")
    if isinstance(max_iter, bool) or not isinstance(max_iter, int):
        raise TypeError("max_iter must be a non-bool int")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")

    def _softmax(row):
        m = max(row)
        exps = []
        total = 0.0
        for value in row:
            term = math.exp(value - m)
            if not math.isfinite(term):
                raise ValueError("softmax terms must be finite")
            exps.append(term)
            total += term
        if not math.isfinite(total) or total <= 0.0:
            raise ValueError("softmax normalizer must be finite and positive")
        p = []
        for term in exps:
            p_j = term / total
            if not math.isfinite(p_j):
                raise ValueError("probabilities must be finite")
            p.append(p_j)
        return p

    L = [list(row) for row in logits]
    n_batch = len(samples)
    ps = [_softmax(row) for row in L]

    g = [[0.0] * n_cols for _ in range(n_rows)]
    counts = [0] * n_rows
    for s, a, adv in samples:
        counts[s] += 1
        row_g = g[s]
        p_row = ps[s]
        for j in range(n_cols):
            term = adv * ((1.0 if j == a else 0.0) - p_row[j])
            row_g[j] += term
            if not math.isfinite(row_g[j]):
                raise ValueError("gradient must be finite")
    for i in range(n_rows):
        for j in range(n_cols):
            g[i][j] /= n_batch
            if not math.isfinite(g[i][j]):
                raise ValueError("gradient must be finite")

    dim = n_rows * n_cols
    g_flat = [0.0] * dim
    for s in range(n_rows):
        base = s * n_cols
        for j in range(n_cols):
            g_flat[base + j] = g[s][j]

    def _dot(u, v):
        total = 0.0
        for i in range(dim):
            total += u[i] * v[i]
            if not math.isfinite(total):
                raise ValueError("dot product must be finite")
        return total

    def _norm(v):
        norm_v = math.sqrt(_dot(v, v))
        if not math.isfinite(norm_v):
            raise ValueError("residual norm must be finite")
        return norm_v

    def _apply_F(vec):
        out = [0.0] * dim
        for s in range(n_rows):
            base = s * n_cols
            p_row = ps[s]
            pd = 0.0
            for j in range(n_cols):
                pd += p_row[j] * vec[base + j]
            if not math.isfinite(pd):
                raise ValueError("Fisher product must be finite")
            scale = counts[s] / n_batch
            for j in range(n_cols):
                v = vec[base + j]
                q_j = damping * v + scale * (p_row[j] * v - p_row[j] * pd)
                if not math.isfinite(q_j):
                    raise ValueError("Fisher product must be finite")
                out[base + j] = q_j
        return out

    x = [0.0] * dim
    r = list(g_flat)
    d = list(r)
    if _norm(r) > tol:
        converged = False
        for _ in range(max_iter):
            rho = _dot(r, r)
            if rho <= 0.0:
                raise ValueError("CG denominator must be positive")
            q = _apply_F(d)
            denom = _dot(d, q)
            if not math.isfinite(denom) or denom <= 0.0:
                raise ValueError("CG denominator must be positive and finite")
            alpha = rho / denom
            if not math.isfinite(alpha):
                raise ValueError("CG step must be finite")
            for i in range(dim):
                new_x = x[i] + alpha * d[i]
                new_r = r[i] - alpha * q[i]
                if not math.isfinite(new_x) or not math.isfinite(new_r):
                    raise ValueError("CG state must be finite")
                x[i] = new_x
                r[i] = new_r
            if _norm(r) <= tol:
                converged = True
                break
            rr = _dot(r, r)
            beta = rr / rho
            if not math.isfinite(beta):
                raise ValueError("CG coefficient must be finite")
            for i in range(dim):
                new_d = r[i] + beta * d[i]
                if not math.isfinite(new_d):
                    raise ValueError("CG direction must be finite")
                d[i] = new_d
        if not converged:
            raise RuntimeError("natural gradient did not converge")

    updated = []
    for s in range(n_rows):
        base = s * n_cols
        row = []
        for j in range(n_cols):
            value = L[s][j] + lr * x[base + j]
            if not math.isfinite(value):
                raise ValueError("updated logits must be finite")
            row.append(value)
        updated.append(row)
    return {"logits": updated}


def ppo_kl_penalty_update(logits, batch, lr=0.05, beta=1.0, epochs=4) -> dict:
    """PPO KL 惩罚多轮 logits 更新，返回固定键序 logits、objectives、kls 的 dict。

    logits 须为非空矩形 list，每行为非空 list 且元素为有限 float；
    batch 须为非空 list，每项为四项 list [s, a, o, A]，其中 s、a 为
    非 bool 的 int 且分别为有效行、列索引，o、A 为有限 float；lr 为
    有限 float 且属于 (0, 1]；beta 为非 bool 的 int/float，转换溢出、
    非有限或不大于 0 抛 ValueError；epochs 为非 bool 正 int。类型不符
    抛 TypeError，其余约束不符抛 ValueError。先完成全部校验再复制
    logits 为 L，不修改输入。每轮冻结 L，以稳定 log-softmax 按行列序
    算 q、p；按 batch 序取 [s, a, o, A]，算 d=q[s][a]-o、r=exp(d)、
    k=o-q[s][a]、z=r*A-beta*k，objective、KL 分别为 z、k 从 0.0 累加
    后除以 batch 长度。梯度从全 0.0 矩阵开始，按样本、列序累加
    (r*A+beta)*(I[j==a]-p[s][j])，除 batch 长度后同步 L+=lr*g。任一
    新量溢出或非有限均抛 ValueError 且无部分结果。返回最终 L、逐轮
    objective 与 KL 的 float list，均为新容器，同输入逐值一致。
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
    if isinstance(beta, bool) or not isinstance(beta, (int, float)):
        raise TypeError("beta must be a non-bool int or float")
    try:
        beta = float(beta)
    except OverflowError:
        raise ValueError("beta must convert to a finite float")
    if not math.isfinite(beta):
        raise ValueError("beta must be finite")
    if beta <= 0.0:
        raise ValueError("beta must be positive")
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
    kls = []
    for _ in range(epochs):
        frozen = [list(row) for row in L]
        qs = []
        ps = []
        for row in frozen:
            q, p = _log_softmax(row)
            qs.append(q)
            ps.append(p)
        g = [[0.0] * n_cols for _ in range(n_rows)]
        objective_total = 0.0
        kl_total = 0.0
        for s, a, o, adv in samples:
            q_sa = qs[s][a]
            d = q_sa - o
            if not math.isfinite(d):
                raise ValueError("log-prob difference must be finite")
            try:
                r = math.exp(d)
            except OverflowError:
                raise ValueError("probability ratio exp overflow")
            k = o - q_sa
            if not math.isfinite(k):
                raise ValueError("KL term must be finite")
            z = r * adv - beta * k
            if not math.isfinite(z):
                raise ValueError("objective term must be finite")
            objective_total += z
            if not math.isfinite(objective_total):
                raise ValueError("objective sum must be finite")
            kl_total += k
            if not math.isfinite(kl_total):
                raise ValueError("KL sum must be finite")
            coeff = r * adv + beta
            if not math.isfinite(coeff):
                raise ValueError("gradient coefficient must be finite")
            row_g = g[s]
            p_row = ps[s]
            for j in range(n_cols):
                grad = coeff * ((1.0 if j == a else 0.0) - p_row[j])
                row_g[j] += grad
                if not math.isfinite(row_g[j]):
                    raise ValueError("gradient must be finite")
        mean_objective = objective_total / n_batch
        if not math.isfinite(mean_objective):
            raise ValueError("epoch mean objective must be finite")
        mean_kl = kl_total / n_batch
        if not math.isfinite(mean_kl):
            raise ValueError("epoch mean KL must be finite")
        objectives.append(mean_objective)
        kls.append(mean_kl)
        for i in range(n_rows):
            for j in range(n_cols):
                g[i][j] /= n_batch
                new_value = L[i][j] + lr * g[i][j]
                if not math.isfinite(new_value):
                    raise ValueError("updated logits must be finite")
                L[i][j] = new_value

    return {
        "logits": L,
        "objectives": objectives,
        "kls": kls,
    }


def ppo_adaptive_kl_update(
    logits,
    batch,
    lr=0.05,
    beta=1.0,
    target=0.01,
    factor=2.0,
    lo=1e-4,
    hi=1e4,
    epochs=4,
) -> dict:
    """自适应 KL 惩罚系数的 PPO 多轮 logits 更新，返回固定键序 logits、
    objectives、kls、betas 的 dict。

    logits、batch、lr、epochs 的校验与异常逐项沿用
    ppo_kl_penalty_update；beta、target、factor、lo、hi 须为非 bool 的
    int/float，类型不符抛 TypeError；转 float 溢出或非有限，
    beta、target、lo、hi <= 0，factor <= 1，或 beta 不在 [lo, hi] 内，
    均抛 ValueError。先完成全部校验再转 float 并复制 logits 为 L，不
    修改输入。记 b=beta，逐轮调用 ppo_kl_penalty_update(L, batch, lr,
    b, 1)，以返回的 logits 续训，并收集该轮 objective、KL。若
    KL>1.5*target 则 b=min(hi, b*factor)；若 KL<target/1.5 则
    b=max(lo, b/factor)；否则 b 不变；每轮记录调整后的新 b。任一运算
    溢出或结果非有限均抛 ValueError，失败时无部分结果。返回最终
    float 矩阵及逐轮 objective、KL、beta 的 float list，均为新容器，
    同输入逐值一致。
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

    numeric_params = (
        ("beta", beta),
        ("target", target),
        ("factor", factor),
        ("lo", lo),
        ("hi", hi),
    )
    for name, value in numeric_params:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a non-bool int or float")

    numeric_values = {}
    for name, value in numeric_params:
        try:
            converted = float(value)
        except OverflowError:
            raise ValueError(f"{name} must convert to a finite float")
        if not math.isfinite(converted):
            raise ValueError(f"{name} must be finite")
        numeric_values[name] = converted
    b = numeric_values["beta"]
    t = numeric_values["target"]
    factor_value = numeric_values["factor"]
    lo_value = numeric_values["lo"]
    hi_value = numeric_values["hi"]

    if b <= 0.0:
        raise ValueError("beta must be positive")
    if t <= 0.0:
        raise ValueError("target must be positive")
    if factor_value <= 1.0:
        raise ValueError("factor must be greater than 1")
    if lo_value <= 0.0:
        raise ValueError("lo must be positive")
    if hi_value <= 0.0:
        raise ValueError("hi must be positive")
    if not lo_value <= b <= hi_value:
        raise ValueError("beta must be within [lo, hi]")

    if isinstance(epochs, bool) or not isinstance(epochs, int):
        raise TypeError("epochs must be a non-bool int")
    if epochs <= 0:
        raise ValueError("epochs must be positive")

    L = [list(row) for row in logits]
    objectives = []
    kls = []
    betas = []
    for _ in range(epochs):
        result = ppo_kl_penalty_update(L, batch, lr, b, 1)
        L = result["logits"]
        objective = result["objectives"][0]
        kl = result["kls"][0]
        objectives.append(objective)
        kls.append(kl)
        try:
            upper = 1.5 * t
            lower = t / 1.5
        except OverflowError:
            raise ValueError("KL threshold must be finite")
        if not math.isfinite(upper) or not math.isfinite(lower):
            raise ValueError("KL threshold must be finite")
        try:
            if kl > upper:
                candidate = b * factor_value
                if not math.isfinite(candidate):
                    raise ValueError("updated beta must be finite")
                b = min(hi_value, candidate)
            elif kl < lower:
                candidate = b / factor_value
                if not math.isfinite(candidate):
                    raise ValueError("updated beta must be finite")
                b = max(lo_value, candidate)
        except OverflowError:
            raise ValueError("updated beta must be finite")
        if not math.isfinite(b):
            raise ValueError("updated beta must be finite")
        betas.append(b)

    return {
        "logits": L,
        "objectives": objectives,
        "kls": kls,
        "betas": betas,
    }


def ppo_entropy_update(logits, lr=0.05, entropy_coef=0.01, epochs=4) -> dict:
    """PPO 熵正则多轮 logits 更新，返回固定键序 logits、entropies、
    probabilities 的 dict。

    logits 须为非空矩形 list，每行为非空 list 且元素满足
    type(x) is float 且为有限值；容器或元素类型错抛 TypeError，
    空、非矩形或非有限抛 ValueError。lr、entropy_coef 为有限
    float 且依次属于 (0, 1]、[0, 1]，类型错抛 TypeError，非有限
    或越界抛 ValueError；epochs 为非 bool 正 int，类型错抛
    TypeError，非正抛 ValueError。先全量校验再复制 logits 为 L，
    不修改输入。每轮冻结 L，逐行取 m=max(row)，按
    p_j=exp(L_j-m)/sum(exp(L_k-m), 0.0) 求稳定 softmax；
    E=-sum(p_j*log(p_j), 0.0)，p_j=0 项以 0 计；按列序梯度
    g_j=entropy_coef*p_j*(-log(p_j)-E)，同步令
    L_j+=lr*g_j。各轮熵均值按行序 sum(E, 0.0)/行数 记入列表。
    任一 exp/log/乘加/更新或最终输出非有限均抛 ValueError，不
    返回部分结果。返回最终 float 矩阵 L、逐轮熵均值 list 及最终
    稳定 softmax 二维 float list；同输入逐值一致。
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
            if type(item) is not float:
                raise TypeError("logits must contain only float")
            if not math.isfinite(item):
                raise ValueError("logits must contain only finite float")
    n_rows = len(logits)
    n_cols = width

    if not isinstance(lr, float):
        raise TypeError("lr must be a float")
    if not math.isfinite(lr):
        raise ValueError("lr must be finite")
    if lr <= 0.0 or lr > 1.0:
        raise ValueError("lr must be in (0, 1]")
    if not isinstance(entropy_coef, float):
        raise TypeError("entropy_coef must be a float")
    if not math.isfinite(entropy_coef):
        raise ValueError("entropy_coef must be finite")
    if entropy_coef < 0.0 or entropy_coef > 1.0:
        raise ValueError("entropy_coef must be in [0, 1]")
    if isinstance(epochs, bool) or not isinstance(epochs, int):
        raise TypeError("epochs must be a non-bool int")
    if epochs <= 0:
        raise ValueError("epochs must be positive")

    L = [list(row) for row in logits]
    entropies = []
    for _ in range(epochs):
        frozen = [list(row) for row in L]
        new_rows = [[0.0] * n_cols for _ in range(n_rows)]
        row_entropies = []
        for i in range(n_rows):
            row = frozen[i]
            m = max(row)
            exps = []
            for j in range(n_cols):
                try:
                    e = math.exp(row[j] - m)
                except OverflowError:
                    raise ValueError("softmax exp overflow")
                if not math.isfinite(e):
                    raise ValueError("softmax exp term must be finite")
                exps.append(e)
            total = sum(exps, 0.0)
            if not math.isfinite(total):
                raise ValueError("softmax normalizer must be finite")
            probs = []
            for j in range(n_cols):
                p = exps[j] / total
                if not math.isfinite(p):
                    raise ValueError("probabilities must be finite")
                probs.append(p)
            terms = []
            for j in range(n_cols):
                p = probs[j]
                if p == 0.0:
                    terms.append(0.0)
                    continue
                try:
                    term = p * math.log(p)
                except OverflowError:
                    raise ValueError("entropy term overflow")
                if not math.isfinite(term):
                    raise ValueError("entropy term must be finite")
                terms.append(term)
            entropy = -sum(terms, 0.0)
            if not math.isfinite(entropy):
                raise ValueError("entropy must be finite")
            row_entropies.append(entropy)
            for j in range(n_cols):
                p = probs[j]
                if p == 0.0:
                    g = 0.0
                else:
                    try:
                        g = entropy_coef * p * (-math.log(p) - entropy)
                    except OverflowError:
                        raise ValueError("entropy gradient overflow")
                if not math.isfinite(g):
                    raise ValueError("entropy gradient must be finite")
                try:
                    new_value = row[j] + lr * g
                except OverflowError:
                    raise ValueError("logit update overflow")
                if not math.isfinite(new_value):
                    raise ValueError("updated logits must be finite")
                new_rows[i][j] = new_value
        mean_entropy = sum(row_entropies, 0.0) / n_rows
        if not math.isfinite(mean_entropy):
            raise ValueError("epoch mean entropy must be finite")
        entropies.append(mean_entropy)
        L = new_rows

    probabilities = []
    for row in L:
        m = max(row)
        exps = []
        for j in range(n_cols):
            try:
                e = math.exp(row[j] - m)
            except OverflowError:
                raise ValueError("softmax exp overflow")
            if not math.isfinite(e):
                raise ValueError("softmax exp term must be finite")
            exps.append(e)
        total = sum(exps, 0.0)
        if not math.isfinite(total):
            raise ValueError("softmax normalizer must be finite")
        probs = []
        for j in range(n_cols):
            p = exps[j] / total
            if not math.isfinite(p):
                raise ValueError("probabilities must be finite")
            probs.append(p)
        probabilities.append(probs)
    return {
        "logits": L,
        "entropies": entropies,
        "probabilities": probabilities,
    }


def ppo_update_entropy(
    logits, batch, lr=0.05, c=0.2, entropy_coef=0.01, epochs=4
) -> dict:
    """PPO 裁剪目标叠加熵正则的多轮 logits 更新，返回固定键序 logits、
    objectives、entropies、probabilities 的 dict。

    logits、batch、lr、c、epochs 的校验与异常逐项沿用 ppo_update；
    entropy_coef 须为非 bool 的 int/float，类型不符抛 TypeError，转
    float 溢出、非有限或越界 [0, 1] 抛 ValueError，校验后记为浮点系数。
    全部参数先校验完毕且不修改输入。

    每轮冻结 L，逐行取 m=max(row)，按列序求 e_j=exp(L_j-m)；PPO 部分
    沿用 ppo_update 的顺序累加 normalizer T（T+=e_j）与
    q_j=L_j-m-log(T)，概率取 p'_j=exp(q_j)；熵部分 normalizer 为
    S=sum(e_j, 0.0)，概率取 p_j=e_j/S。PPO 部分按 batch 序以 o、q_a 求
    r=exp(q_a-o)，objective 为 sum(min(r*A, clip(r)*A), 0.0)/len(batch)；
    若 A>0 且 r>1+c 或 A<0 且 r<1-c，该样本梯度为 0，否则行 s 列 j
    梯度为 r*A*(I[j==a]-p'_j)，依样本、列序累加后除批量长度。熵部分
    逐行令 E=-sum(p_j*log(p_j), 0.0)（p_j=0 项以 0 计），列序梯度
    h_j=entropy_coef*p_j*(-log(p_j)-E)，各轮熵均值为
    sum(E, 0.0)/行数。两份梯度均取自冻结 L，再同步令
    L_j+=lr*(ppo_j+entropy_j)；entropy_coef=0 时 logits 与 objectives
    与 ppo_update 逐值相同。任一 exp/log/乘加/更新或最终输出非有限均
    抛 ValueError。返回最终 float 矩阵 L、逐轮 objective 均值 list、
    逐轮熵均值 list 及最终稳定 softmax（按 p_j 配方）二维 float list；
    同输入逐值一致。
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
    if isinstance(entropy_coef, bool) or not isinstance(
        entropy_coef, (int, float)
    ):
        raise TypeError("entropy_coef must be an int or float")
    try:
        entropy_coef = float(entropy_coef)
    except OverflowError:
        raise ValueError("entropy_coef must convert to a finite float")
    if not math.isfinite(entropy_coef):
        raise ValueError("entropy_coef must be finite")
    if entropy_coef < 0.0 or entropy_coef > 1.0:
        raise ValueError("entropy_coef must be in [0, 1]")

    def _frozen(row):
        m = max(row)
        exps = []
        total_ppo = 0.0
        for j in range(n_cols):
            try:
                e = math.exp(row[j] - m)
            except OverflowError:
                raise ValueError("softmax exp overflow")
            if not math.isfinite(e):
                raise ValueError("softmax exp term must be finite")
            exps.append(e)
            total_ppo += e
            if not math.isfinite(total_ppo):
                raise ValueError("softmax normalizer must be finite")
        total = sum(exps, 0.0)
        if not math.isfinite(total):
            raise ValueError("softmax normalizer must be finite")
        log_total = math.log(total_ppo)
        if not math.isfinite(log_total):
            raise ValueError("softmax log normalizer must be finite")
        q = []
        p_ppo = []
        p_entropy = []
        for j in range(n_cols):
            q_j = row[j] - m - log_total
            if not math.isfinite(q_j):
                raise ValueError("log-probabilities must be finite")
            pp_j = math.exp(q_j)
            if not math.isfinite(pp_j):
                raise ValueError("probabilities must be finite")
            pe_j = exps[j] / total
            if not math.isfinite(pe_j):
                raise ValueError("probabilities must be finite")
            q.append(q_j)
            p_ppo.append(pp_j)
            p_entropy.append(pe_j)
        return q, p_ppo, p_entropy

    def _final_probs(row):
        m = max(row)
        exps = []
        for j in range(n_cols):
            try:
                e = math.exp(row[j] - m)
            except OverflowError:
                raise ValueError("softmax exp overflow")
            if not math.isfinite(e):
                raise ValueError("softmax exp term must be finite")
            exps.append(e)
        total = sum(exps, 0.0)
        if not math.isfinite(total):
            raise ValueError("softmax normalizer must be finite")
        probs = []
        for j in range(n_cols):
            p_j = exps[j] / total
            if not math.isfinite(p_j):
                raise ValueError("probabilities must be finite")
            probs.append(p_j)
        return probs

    L = [list(row) for row in logits]
    n_batch = len(samples)
    objectives = []
    entropies = []
    for _ in range(epochs):
        frozen = [list(row) for row in L]
        qs = []
        ps_ppo = []
        ps_entropy = []
        for row in frozen:
            q, p_ppo, p_entropy = _frozen(row)
            qs.append(q)
            ps_ppo.append(p_ppo)
            ps_entropy.append(p_entropy)

        g = [[0.0] * n_cols for _ in range(n_rows)]
        objective_terms = []
        for s, a, o, adv in samples:
            d = qs[s][a] - o
            if not math.isfinite(d):
                raise ValueError("log-prob difference must be finite")
            try:
                r = math.exp(d)
            except OverflowError:
                raise ValueError("probability ratio exp overflow")
            if not math.isfinite(r):
                raise ValueError("probability ratio must be finite")
            clipped = min(max(r, 1.0 - c), 1.0 + c)
            u = r * adv
            v = clipped * adv
            if not math.isfinite(u) or not math.isfinite(v):
                raise ValueError("clipped objective terms must be finite")
            objective_terms.append(min(u, v))
            if (adv > 0.0 and r > 1.0 + c) or (
                adv < 0.0 and r < 1.0 - c
            ):
                continue
            row_g = g[s]
            p_row = ps_ppo[s]
            coeff = r * adv
            for j in range(n_cols):
                grad = coeff * ((1.0 if j == a else 0.0) - p_row[j])
                row_g[j] += grad
                if not math.isfinite(row_g[j]):
                    raise ValueError("gradient must be finite")
        objective_total = sum(objective_terms, 0.0)
        if not math.isfinite(objective_total):
            raise ValueError("objective sum must be finite")
        mean_objective = objective_total / n_batch
        if not math.isfinite(mean_objective):
            raise ValueError("epoch mean objective must be finite")
        objectives.append(mean_objective)

        entropy_grads = [[0.0] * n_cols for _ in range(n_rows)]
        row_entropies = []
        for i in range(n_rows):
            probs = ps_entropy[i]
            terms = []
            for j in range(n_cols):
                p = probs[j]
                if p == 0.0:
                    terms.append(0.0)
                    continue
                try:
                    term = p * math.log(p)
                except OverflowError:
                    raise ValueError("entropy term overflow")
                if not math.isfinite(term):
                    raise ValueError("entropy term must be finite")
                terms.append(term)
            entropy = -sum(terms, 0.0)
            if not math.isfinite(entropy):
                raise ValueError("entropy must be finite")
            row_entropies.append(entropy)
            for j in range(n_cols):
                p = probs[j]
                if p == 0.0:
                    h = 0.0
                else:
                    try:
                        h = entropy_coef * p * (-math.log(p) - entropy)
                    except OverflowError:
                        raise ValueError("entropy gradient overflow")
                if not math.isfinite(h):
                    raise ValueError("entropy gradient must be finite")
                entropy_grads[i][j] = h
        mean_entropy = sum(row_entropies, 0.0) / n_rows
        if not math.isfinite(mean_entropy):
            raise ValueError("epoch mean entropy must be finite")
        entropies.append(mean_entropy)

        for i in range(n_rows):
            for j in range(n_cols):
                ppo_grad = g[i][j] / n_batch
                if not math.isfinite(ppo_grad):
                    raise ValueError("mean gradient must be finite")
                total_grad = ppo_grad + entropy_grads[i][j]
                if not math.isfinite(total_grad):
                    raise ValueError("combined gradient must be finite")
                new_value = L[i][j] + lr * total_grad
                if not math.isfinite(new_value):
                    raise ValueError("updated logits must be finite")
                L[i][j] = new_value

    probabilities = [_final_probs(row) for row in L]
    return {
        "logits": L,
        "objectives": objectives,
        "entropies": entropies,
        "probabilities": probabilities,
    }


def ppo_grad_clip(
    logits, batch, lr=0.05, c=0.2, epochs=4, max_norm=1.0
) -> dict:
    """带梯度范数裁剪的 PPO 多轮 logits 更新，返回固定键序 logits、
    objectives、probabilities、norms、scales 的 dict。

    logits、batch、lr、c、epochs 的校验、异常、输入复制及逐轮目标/
    梯度算法完全沿用 ppo_update；max_norm 须为非 bool 的 int/float，
    类型不符抛 TypeError，转 float 溢出、非有限或 <=0 抛 ValueError，
    校验后取 float M。全部参数先校验完毕且不修改输入。

    每轮冻结 L，按 ppo_update 算更新前 objective 及累计梯度并除以
    batch 长度；写 L 前按行列序从 0.0 累加 g[i][j]**2，令
    norm=sqrt(sum)，scale=1.0 if norm<=M else M/norm，再同步作
    L[i][j]+=lr*scale*g[i][j]，下一轮使用新 L。平方、累加、norm、
    scale 或新 L 溢出/非有限均抛 ValueError；零梯度得 norm=0.0、
    scale=1.0。返回最终 L、各轮均值 list、最终 softmax 二维 list、
    逐轮裁剪前范数 float 列表与逐轮缩放因子 float 列表；均为新容器，
    同输入逐值一致。
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
    if isinstance(max_norm, bool) or not isinstance(max_norm, (int, float)):
        raise TypeError("max_norm must be an int or float")
    try:
        M = float(max_norm)
    except OverflowError:
        raise ValueError("max_norm must convert to a finite float")
    if not math.isfinite(M) or M <= 0.0:
        raise ValueError("max_norm must be finite and > 0")

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
    norms = []
    scales = []
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
        sq_total = 0.0
        for i in range(n_rows):
            for j in range(n_cols):
                try:
                    sq = g[i][j] ** 2
                except OverflowError:
                    raise ValueError("squared gradient must be finite")
                if not math.isfinite(sq):
                    raise ValueError("squared gradient must be finite")
                sq_total += sq
                if not math.isfinite(sq_total):
                    raise ValueError("gradient norm sum must be finite")
        norm = math.sqrt(sq_total)
        if not math.isfinite(norm):
            raise ValueError("gradient norm must be finite")
        scale = 1.0 if norm <= M else M / norm
        if not math.isfinite(scale):
            raise ValueError("gradient scale must be finite")
        norms.append(norm)
        scales.append(scale)
        for i in range(n_rows):
            for j in range(n_cols):
                new_value = L[i][j] + lr * scale * g[i][j]
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
        "norms": norms,
        "scales": scales,
    }


def ppo_minibatch(
    logits, batch, batch_size=32, lr=0.05, c=0.2, epochs=4, seed=0
) -> dict:
    """小批量 PPO 多轮 logits 更新，返回固定键序 logits、objectives、
    probabilities 的 dict。

    logits、batch、lr、c 的校验与异常逐项沿用 ppo_update；
    batch_size、epochs、seed 须为非 bool 的 int，batch_size、epochs
    还须为正，类型不符抛 TypeError，非正抛 ValueError。全部参数先
    校验完毕且不修改输入，随后仅建一个 random.Random(seed)，复制
    logits 为 L。

    每轮令 p=list(range(len(batch)))，按 i 从 n-1 降至 1 取
    j=rng.randrange(i+1) 并交换 p[i]、p[j]（Fisher–Yates）；再按 p
    连续切为至多 batch_size 项的小批，尾批保留。依 p 取 batch 行，
    对每个小批调用 ppo_update(L, sub, lr, c, 1)，以返回的 logits
    续训；失败原样抛出且无部分结果。返回的 logits 为最终 L，
    objectives 为按轮排列的各小批 objectives[0]（float），
    probabilities 为最终 L 的稳定 softmax 二维 float list。
    同输入同 seed 逐值一致。
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
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError("batch_size must be a non-bool int")
    if isinstance(epochs, bool) or not isinstance(epochs, int):
        raise TypeError("epochs must be a non-bool int")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be a non-bool int")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if epochs <= 0:
        raise ValueError("epochs must be positive")

    rng = random.Random(seed)
    L = [list(row) for row in logits]
    n_batch = len(batch)
    objectives = []
    last_result = None
    for _ in range(epochs):
        order = list(range(n_batch))
        for i in range(n_batch - 1, 0, -1):
            j = rng.randrange(i + 1)
            order[i], order[j] = order[j], order[i]
        for start in range(0, n_batch, batch_size):
            sub = [batch[order[k]] for k in range(
                start, min(start + batch_size, n_batch)
            )]
            last_result = ppo_update(L, sub, lr, c, 1)
            L = last_result["logits"]
            objectives.append(last_result["objectives"][0])

    return {
        "logits": L,
        "objectives": objectives,
        "probabilities": last_result["probabilities"],
    }


def ppo_minibatch_kl(
    logits,
    batch,
    batch_size=32,
    lr=0.05,
    c=0.2,
    epochs=4,
    target_kl=0.01,
    seed=0,
) -> dict:
    """带 KL 早停的小批量 PPO 多轮 logits 更新，返回固定键序 logits、
    objectives、probabilities、kls、stopped 的 dict。

    logits、batch、batch_size、lr、c、epochs、seed 的校验、洗牌、切批
    及随机消费逐项沿用 ppo_minibatch；target_kl 须为非 bool 的
    int/float，类型不符抛 TypeError，转 float 溢出、非有限或 <=0 抛
    ValueError，校验后记为 t。全部参数先校验完毕且不修改输入，随后
    仅建一个 random.Random(seed)，复制 logits 为 L。

    每轮 Fisher–Yates 洗牌后连续切批（尾批保留），依序对每个小批调用
    ppo_update(L, sub, lr, c, 1)，以返回的 logits 续训，并顺序收集各
    小批 objectives[0]。轮末按 batch 原序，以当前 L 的稳定
    log-softmax 重算各样本所选动作的 new_logp，从 0.0 顺序累加
    old_logp-new_logp，除以样本数得 float KL 并收集；任一新量非有限
    抛 ValueError。若 KL>t，保留本轮更新并立即结束（stopped=True），
    否则最多执行 epochs 轮。返回最终 L、展平的各小批目标 float 列表、
    最终稳定 softmax 二维 float 表、逐轮 KL float 列表与是否出现
    KL>t 的 bool；均为新容器，同输入同 seed 逐值一致。
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
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError("batch_size must be a non-bool int")
    if isinstance(epochs, bool) or not isinstance(epochs, int):
        raise TypeError("epochs must be a non-bool int")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be a non-bool int")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if isinstance(target_kl, bool) or not isinstance(
        target_kl, (int, float)
    ):
        raise TypeError("target_kl must be an int or float")
    try:
        t = float(target_kl)
    except OverflowError:
        raise ValueError("target_kl must convert to a finite float")
    if not math.isfinite(t) or t <= 0.0:
        raise ValueError("target_kl must be finite and > 0")

    def _selected_logp(row, action_index):
        """与 ppo_update 相同的稳定 log-softmax，返回所选动作列的 logp。"""
        m = max(row)
        total = 0.0
        for value in row:
            total += math.exp(value - m)
        if not math.isfinite(total):
            raise ValueError("softmax normalizer must be finite")
        log_total = math.log(total)
        q_a = row[action_index] - m - log_total
        if not math.isfinite(q_a):
            raise ValueError("log-probabilities must be finite")
        return q_a

    rng = random.Random(seed)
    L = [list(row) for row in logits]
    n_batch = len(batch)
    objectives = []
    kls = []
    probabilities = []
    stopped = False
    for _ in range(epochs):
        order = list(range(n_batch))
        for i in range(n_batch - 1, 0, -1):
            j = rng.randrange(i + 1)
            order[i], order[j] = order[j], order[i]
        for start in range(0, n_batch, batch_size):
            sub = [batch[order[k]] for k in range(
                start, min(start + batch_size, n_batch)
            )]
            result = ppo_update(L, sub, lr, c, 1)
            L = result["logits"]
            probabilities = result["probabilities"]
            objectives.append(result["objectives"][0])
        kl_sum = 0.0
        for s, a, o, _adv in samples:
            new_logp = _selected_logp(L[s], a)
            diff = o - new_logp
            if not math.isfinite(diff):
                raise ValueError("KL term must be finite")
            kl_sum += diff
            if not math.isfinite(kl_sum):
                raise ValueError("KL sum must be finite")
        kl = kl_sum / len(samples)
        if not math.isfinite(kl):
            raise ValueError("KL must be finite")
        kls.append(kl)
        if kl > t:
            stopped = True
            break

    return {
        "logits": L,
        "objectives": objectives,
        "probabilities": probabilities,
        "kls": kls,
        "stopped": stopped,
    }


def ppo_minibatch_adaptive_kl(
    logits,
    batch,
    batch_size=32,
    lr=0.05,
    beta=1.0,
    target=0.01,
    factor=2.0,
    lo=1e-4,
    hi=1e4,
    epochs=4,
    seed=0,
) -> dict:
    """自适应 KL 惩罚系数的小批量 PPO 多轮 logits 更新，返回固定键序
    logits、objectives、kls、betas 的 dict。

    logits、batch、lr 的校验与异常逐项沿用 ppo_kl_penalty_update；
    beta、target、factor、lo、hi 五个 KL 参数的校验与异常沿用
    ppo_adaptive_kl_update；batch_size、epochs、seed 三个整数的校验
    与异常沿用 ppo_minibatch。全部参数先校验完毕且不修改输入，随后
    仅建一个 random.Random(seed)，复制 logits 为 L，令 b=float(beta)。

    每轮令 p=list(range(len(batch)))，按 i 从 n-1 降至 1 取
    j=rng.randrange(i+1) 并交换 p[i]、p[j]（Fisher–Yates）；再按 p
    连续切为至多 batch_size 项的小批，尾批保留。依 p 取 batch 行，
    对每个小批调用 ppo_adaptive_kl_update(L, sub, lr, b, target,
    factor, lo, hi, 1)，以返回的 logits 续训；其 objectives[0]、
    kls[0]、betas[0] 按批序展平收集，返回的新 beta 跨批跨轮传递。
    任一运算溢出或非有限由子调用抛 ValueError；失败原样抛出且无
    部分结果。返回最终 float 矩阵 L 及按批序展平的 objective、KL、
    beta float 列表，均为新容器，同输入同 seed 逐值一致。
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

    if not isinstance(lr, float):
        raise TypeError("lr must be a float")
    if not math.isfinite(lr):
        raise ValueError("lr must be finite")
    if lr <= 0.0 or lr > 1.0:
        raise ValueError("lr must be in (0, 1]")

    numeric_params = (
        ("beta", beta),
        ("target", target),
        ("factor", factor),
        ("lo", lo),
        ("hi", hi),
    )
    for name, value in numeric_params:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a non-bool int or float")

    numeric_values = {}
    for name, value in numeric_params:
        try:
            converted = float(value)
        except OverflowError:
            raise ValueError(f"{name} must convert to a finite float")
        if not math.isfinite(converted):
            raise ValueError(f"{name} must be finite")
        numeric_values[name] = converted
    b0 = numeric_values["beta"]
    t = numeric_values["target"]
    factor_value = numeric_values["factor"]
    lo_value = numeric_values["lo"]
    hi_value = numeric_values["hi"]

    if b0 <= 0.0:
        raise ValueError("beta must be positive")
    if t <= 0.0:
        raise ValueError("target must be positive")
    if factor_value <= 1.0:
        raise ValueError("factor must be greater than 1")
    if lo_value <= 0.0:
        raise ValueError("lo must be positive")
    if hi_value <= 0.0:
        raise ValueError("hi must be positive")
    if not lo_value <= b0 <= hi_value:
        raise ValueError("beta must be within [lo, hi]")

    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError("batch_size must be a non-bool int")
    if isinstance(epochs, bool) or not isinstance(epochs, int):
        raise TypeError("epochs must be a non-bool int")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be a non-bool int")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if epochs <= 0:
        raise ValueError("epochs must be positive")

    rng = random.Random(seed)
    L = [list(row) for row in logits]
    b = float(beta)
    n_batch = len(batch)
    objectives = []
    kls = []
    betas = []
    for _ in range(epochs):
        order = list(range(n_batch))
        for i in range(n_batch - 1, 0, -1):
            j = rng.randrange(i + 1)
            order[i], order[j] = order[j], order[i]
        for start in range(0, n_batch, batch_size):
            sub = [batch[order[k]] for k in range(
                start, min(start + batch_size, n_batch)
            )]
            result = ppo_adaptive_kl_update(
                L, sub, lr, b, t, factor_value, lo_value, hi_value, 1
            )
            L = result["logits"]
            objectives.append(result["objectives"][0])
            kls.append(result["kls"][0])
            b = result["betas"][0]
            betas.append(b)

    return {
        "logits": L,
        "objectives": objectives,
        "kls": kls,
        "betas": betas,
    }


def ppo_update_kl(
    logits, batch, lr=0.05, c=0.2, epochs=4, target_kl=0.01
) -> dict:
    """带 KL 早停的 PPO 多轮 logits 更新，返回固定键序 logits、
    objectives、probabilities、kls、stopped 的 dict。

    logits、batch、lr、c、epochs 的校验与异常逐项沿用 ppo_update；
    target_kl 须为非 bool 的 int/float，类型不符抛 TypeError，转 float
    溢出、非有限或 <=0 抛 ValueError，校验后记为 t。复制 logits 为 L，
    逐轮调用 ppo_update(L, batch, lr, c, 1)，用返回 logits 续训，并收集
    每轮唯一 objective。每轮后按 ppo_update 的稳定 log-softmax 重算
    batch 各样本所选动作的新 logp；按 batch 序从 0.0 累加
    old_logp-new_logp，除批量长度得 float KL 并收集；任一新结果非有限
    抛 ValueError。若 KL>t，保留本轮更新并立即结束（stopped=True），
    否则最多执行 epochs 轮。返回最终 float 矩阵、已执行轮目标列表、
    最终 softmax 矩阵、KL 列表与是否出现 KL>t 的 bool；返回新容器且
    不修改输入，同输入逐值一致。
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
    if isinstance(target_kl, bool) or not isinstance(
        target_kl, (int, float)
    ):
        raise TypeError("target_kl must be an int or float")
    try:
        t = float(target_kl)
    except OverflowError:
        raise ValueError("target_kl must convert to a finite float")
    if not math.isfinite(t) or t <= 0.0:
        raise ValueError("target_kl must be finite and > 0")

    def _selected_logp(row, action_index):
        """与 ppo_update 相同的稳定 log-softmax，返回所选动作列的 logp。"""
        m = max(row)
        total = 0.0
        for value in row:
            total += math.exp(value - m)
        if not math.isfinite(total):
            raise ValueError("softmax normalizer must be finite")
        log_total = math.log(total)
        q_a = row[action_index] - m - log_total
        if not math.isfinite(q_a):
            raise ValueError("log-probabilities must be finite")
        return q_a

    L = [list(row) for row in logits]
    objectives = []
    kls = []
    probabilities = []
    stopped = False
    for _ in range(epochs):
        result = ppo_update(L, batch, lr, c, 1)
        L = result["logits"]
        probabilities = result["probabilities"]
        objectives.append(result["objectives"][0])
        kl_sum = 0.0
        for s, a, o, _adv in samples:
            new_logp = _selected_logp(L[s], a)
            diff = o - new_logp
            if not math.isfinite(diff):
                raise ValueError("KL term must be finite")
            kl_sum += diff
            if not math.isfinite(kl_sum):
                raise ValueError("KL sum must be finite")
        kl = kl_sum / len(samples)
        if not math.isfinite(kl):
            raise ValueError("KL must be finite")
        kls.append(kl)
        if kl > t:
            stopped = True
            break

    return {
        "logits": L,
        "objectives": objectives,
        "probabilities": probabilities,
        "kls": kls,
        "stopped": stopped,
    }


def ppo_normalized_update(
    logits, batch, lr=0.05, c=0.2, epochs=4, epsilon=1e-8
) -> dict:
    """优势规范化的 PPO 多轮 logits 更新，返回固定键序 logits、
    objectives、probabilities、mean、std、advantages 的 dict。

    logits、batch、lr、c、epochs 的校验与异常逐项沿用 ppo_update；
    epsilon 须为非 bool 的 int/float，类型不符抛 TypeError，转 float
    溢出、非有限或 <=0 抛 ValueError，校验后记为 e。全部参数先校验
    完毕且不修改输入。

    随后按 batch 序从 0.0 累加 A 并除以项数得均值 m，再从 0.0 按同序
    累加 (A-m)**2 并除以项数得总体方差 v，取 d=math.sqrt(v)。v==0.0
    时规范化优势均为正 0.0，否则按原序取 (A-m)/(d+e)；上述任一运算
    溢出或中间量、结果非有限均抛 ValueError。以原 s/a/o 与规范化优势
    构造新行，调用 ppo_update(logits, new_batch, lr, c, epochs)，其
    异常原样抛出。返回其 logits、objectives、probabilities 及 m、d、
    规范化优势 float 列表；不修改输入，同输入逐值一致。
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
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)):
        raise TypeError("epsilon must be an int or float")
    try:
        e = float(epsilon)
    except OverflowError:
        raise ValueError("epsilon must convert to a finite float")
    if not math.isfinite(e) or e <= 0.0:
        raise ValueError("epsilon must be finite and > 0")

    n_batch = len(samples)
    total = 0.0
    for _s, _a, _o, adv in samples:
        total += adv
        if not math.isfinite(total):
            raise ValueError("advantage sum must be finite")
    m = total / n_batch
    if not math.isfinite(m):
        raise ValueError("advantage mean must be finite")
    squared = 0.0
    for _s, _a, _o, adv in samples:
        diff = adv - m
        if not math.isfinite(diff):
            raise ValueError("centered advantage must be finite")
        try:
            term = diff ** 2
        except OverflowError:
            raise ValueError("squared advantage must be finite")
        if not math.isfinite(term):
            raise ValueError("squared advantage must be finite")
        squared += term
        if not math.isfinite(squared):
            raise ValueError("variance sum must be finite")
    v = squared / n_batch
    if not math.isfinite(v):
        raise ValueError("advantage variance must be finite")
    d = math.sqrt(v)
    if not math.isfinite(d):
        raise ValueError("advantage std must be finite")

    if v == 0.0:
        advantages = [0.0] * n_batch
    else:
        denom = d + e
        if not math.isfinite(denom):
            raise ValueError("normalization denominator must be finite")
        advantages = []
        for _s, _a, _o, adv in samples:
            diff = adv - m
            if not math.isfinite(diff):
                raise ValueError("centered advantage must be finite")
            normalized = diff / denom
            if not math.isfinite(normalized):
                raise ValueError("normalized advantage must be finite")
            advantages.append(normalized)

    new_batch = [
        [s, a, o, normalized]
        for (s, a, o, _adv), normalized in zip(samples, advantages)
    ]
    result = ppo_update(logits, new_batch, lr, c, epochs)
    return {
        "logits": result["logits"],
        "objectives": result["objectives"],
        "probabilities": result["probabilities"],
        "mean": m,
        "std": d,
        "advantages": advantages,
    }


def ppo_policy_kl(old_logits, new_logits) -> dict:
    """新旧策略 logits 的逐行 KL 散度，返回固定键序 rows、mean、max 的
    dict。

    old_logits、new_logits 均须为非空矩形 list，每行须为非空 list 且
    元素恰为有限 float（type(x) is float，float 子类亦抛 TypeError）；参数、行或元素类型不符抛 TypeError，空表、
    空行、非矩形、两侧形状不同或元素非有限抛 ValueError。先对两侧做
    全量校验再计算，且不修改输入。逐行按原列序作稳定 log-softmax：
    m=max(row)、z=sum(exp(x-m), 0.0)、q[j]=row[j]-m-log(z)；旧策略另取
    p[j]=exp(q_old[j])，再从 0.0 按列序累加
    KL=Σp[j]*(q_old[j]-q_new[j])，不夹断负零或舍入产生的微小负值。
    m、z、q、p、乘积、累加或汇总非有限，或 z<=0，均抛 ValueError。
    rows 为按行序的 float KL 列表，mean 为 sum(rows, 0.0)/len(rows)，
    max 为内置 max 所得 float；计算顺序固定，同输入逐值一致。
    """

    def _validate_logits(value, name):
        if not isinstance(value, list):
            raise TypeError(name + " must be a list")
        if not value:
            raise ValueError(name + " must be non-empty")
        width = None
        for row in value:
            if not isinstance(row, list):
                raise TypeError("every row of " + name + " must be a list")
            if not row:
                raise ValueError("every row of " + name + " must be non-empty")
            if width is None:
                width = len(row)
            elif len(row) != width:
                raise ValueError(name + " must be rectangular")
            for item in row:
                if type(item) is not float:
                    raise TypeError(name + " must contain only float")
                if not math.isfinite(item):
                    raise ValueError(
                        name + " must contain only finite float"
                    )
        return len(value), width

    old_shape = _validate_logits(old_logits, "old_logits")
    new_shape = _validate_logits(new_logits, "new_logits")
    if old_shape != new_shape:
        raise ValueError("old_logits and new_logits must have the same shape")

    def _log_softmax(row):
        m = max(row)
        if not math.isfinite(m):
            raise ValueError("row maximum must be finite")
        z = sum((math.exp(x - m) for x in row), 0.0)
        if not math.isfinite(z):
            raise ValueError("softmax normalizer must be finite")
        if z <= 0.0:
            raise ValueError("softmax normalizer must be positive")
        log_z = math.log(z)
        q = []
        for x in row:
            value = x - m - log_z
            if not math.isfinite(value):
                raise ValueError("log-probabilities must be finite")
            q.append(value)
        return q

    rows = []
    for old_row, new_row in zip(old_logits, new_logits):
        q_old = _log_softmax(old_row)
        q_new = _log_softmax(new_row)
        kl = 0.0
        for j in range(len(q_old)):
            p = math.exp(q_old[j])
            if not math.isfinite(p):
                raise ValueError("old-policy probabilities must be finite")
            product = p * (q_old[j] - q_new[j])
            if not math.isfinite(product):
                raise ValueError("KL term must be finite")
            kl += product
            if not math.isfinite(kl):
                raise ValueError("KL sum must be finite")
        rows.append(kl)

    mean = sum(rows, 0.0) / len(rows)
    if not math.isfinite(mean):
        raise ValueError("mean KL must be finite")
    maximum = max(rows)
    if not math.isfinite(maximum):
        raise ValueError("max KL must be finite")
    return {"rows": rows, "mean": mean, "max": maximum}


def ppo_line_search(
    logits, batch, lr=0.05, c=0.2, target=0.01, steps=8
) -> dict:
    """按几何衰减学习率做 PPO 单轮候选搜索，返回固定键序 logits、
    kls、accepted、lr 的 dict。

    logits、batch、lr、c 的校验与异常逐项沿用 ppo_update；target 须为
    非 bool 的 int/float，类型不符抛 TypeError，转 float 溢出、非有限
    或 <=0 抛 ValueError，校验后记为 t；steps 须为非 bool 的 int 且
    属于 [1, 61]，类型不符抛 TypeError，越界抛 ValueError。全部参数
    先校验完毕且不修改输入。

    按 i=0..steps-1 令 rate=lr/(2**i)，每次均从原 logits 调用
    ppo_update(logits, batch, rate, c, 1)，候选之间不串接；以
    ppo_policy_kl 计算原 logits 到候选 logits 的 mean KL，将该 float
    依序加入 kls，首个 KL<=t 即接受并停止。新中间量非有限抛
    ValueError 且无部分结果。成功时 logits 取该候选的新矩阵、
    accepted 为 True、lr 为对应 rate；全部拒绝时 logits 为原矩阵的
    新副本、accepted 为 False、lr 为 None；kls 为已尝试的 KL float
    列表。同输入逐值一致。
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
    if isinstance(target, bool) or not isinstance(target, (int, float)):
        raise TypeError("target must be an int or float")
    try:
        t = float(target)
    except OverflowError:
        raise ValueError("target must convert to a finite float")
    if not math.isfinite(t) or t <= 0.0:
        raise ValueError("target must be finite and > 0")
    if isinstance(steps, bool) or not isinstance(steps, int):
        raise TypeError("steps must be a non-bool int")
    if steps < 1 or steps > 61:
        raise ValueError("steps must be in [1, 61]")

    kls = []
    accepted_logits = None
    accepted_rate = None
    for i in range(steps):
        rate = lr / (2 ** i)
        if not math.isfinite(rate):
            raise ValueError("candidate learning rate must be finite")
        candidate = ppo_update(logits, batch, rate, c, 1)["logits"]
        kl = ppo_policy_kl(logits, candidate)["mean"]
        if not math.isfinite(kl):
            raise ValueError("KL must be finite")
        kls.append(kl)
        if kl <= t:
            accepted_logits = candidate
            accepted_rate = rate
            break

    if accepted_logits is None:
        return {
            "logits": [list(row) for row in logits],
            "kls": kls,
            "accepted": False,
            "lr": None,
        }
    return {
        "logits": accepted_logits,
        "kls": kls,
        "accepted": True,
        "lr": accepted_rate,
    }


def trpo(logits, batch, max_kl=0.01, steps=10) -> dict:
    """TRPO 回溯线搜索：沿自然梯度方向按几何衰减倍率取候选，返回固定
    键序 logits、objectives、kls、accepted、step 的 dict。

    logits、batch 契约沿用 natural_gradient：logits 须为非空矩形
    list，每行为非空 list 且元素为有限 float；batch 须为非空 list，
    每项恰为三项 list [s, a, A]，s、a 为非 bool 的 int 且分别为有效
    行、列索引，A 为有限 float。max_kl 须为非 bool 的 int/float，
    类型不符抛 TypeError，转 float 溢出、非有限或 <=0 抛 ValueError；
    steps 须为非 bool 的 int 且属于 [1, 61]，类型不符抛 TypeError，
    越界抛 ValueError。全部参数先校验完毕且不修改输入。

    先以 natural_gradient(logits, batch, 1.0)["logits"] 取满步长目标
    D，方向 d=D-logits；对原 logits 作稳定 log-softmax 得 q0，基线
    J0=sum(A, 0.0)/len(batch)。依 i=0..steps-1，每次均从原 logits 造
    X=logits+(2.0**-i)*d，候选之间不串接；对 X 作稳定 log-softmax
    得 qX，目标
    J(X)=sum(A*exp(qX[s,a]-q0[s,a]), 0.0)/len(batch)，并将原 logits 与
    X 逐项 float() 转为普通 float（兼容 float 子类，不修改输入）后以
    ppo_policy_kl(plain, plain_X)["mean"] 取旧策略到新策略的 mean KL 记
    为 K(X)，将该次的 J、K float 依序加入 objectives、kls。首个满足
    K<=max_kl 且 J>=J0 的候选即接受并停止，列表截止于命中项；全部
    拒绝时两列表各含 steps 项。exp 向下溢出为 0 合法，向上溢出或任
    一中间结果非有限抛 ValueError。成功时 logits 取命中候选矩阵、
    accepted 为 True、step 为对应倍率 2.0**-i；全部拒绝时 logits 为
    原矩阵的新副本、accepted 为 False、step 为 None。同输入逐值
    一致。
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
        if len(item) != 3:
            raise ValueError("every batch item must have exactly three elements")
        s, a, adv = item
        if isinstance(s, bool) or not isinstance(s, int):
            raise TypeError("state index must be a non-bool int")
        if isinstance(a, bool) or not isinstance(a, int):
            raise TypeError("action index must be a non-bool int")
        if not 0 <= s < n_rows:
            raise ValueError("state index out of range")
        if not 0 <= a < n_cols:
            raise ValueError("action index out of range")
        if not isinstance(adv, float):
            raise TypeError("advantage must be a float")
        if not math.isfinite(adv):
            raise ValueError("advantage must be finite")
        samples.append((s, a, adv))

    if isinstance(max_kl, bool) or not isinstance(max_kl, (int, float)):
        raise TypeError("max_kl must be a non-bool int or float")
    try:
        max_kl = float(max_kl)
    except OverflowError:
        raise ValueError("max_kl must convert to a finite float")
    if not math.isfinite(max_kl) or max_kl <= 0.0:
        raise ValueError("max_kl must be finite and > 0")
    if isinstance(steps, bool) or not isinstance(steps, int):
        raise TypeError("steps must be a non-bool int")
    if steps < 1 or steps > 61:
        raise ValueError("steps must be in [1, 61]")

    def _log_softmax(matrix_row):
        m = max(matrix_row)
        if not math.isfinite(m):
            raise ValueError("row maximum must be finite")
        z = sum((math.exp(x - m) for x in matrix_row), 0.0)
        if not math.isfinite(z):
            raise ValueError("softmax normalizer must be finite")
        if z <= 0.0:
            raise ValueError("softmax normalizer must be positive")
        log_z = math.log(z)
        q_row = []
        for x in matrix_row:
            value = x - m - log_z
            if not math.isfinite(value):
                raise ValueError("log-probabilities must be finite")
            q_row.append(value)
        return q_row

    n_batch = len(samples)
    target = natural_gradient(logits, batch, 1.0)["logits"]
    direction = [
        [target[s][j] - logits[s][j] for j in range(n_cols)]
        for s in range(n_rows)
    ]
    q0 = [_log_softmax(row) for row in logits]
    plain_logits = [
        [float(value) for value in row] for row in logits
    ]
    baseline = sum((adv for _s, _a, adv in samples), 0.0) / n_batch
    if not math.isfinite(baseline):
        raise ValueError("baseline objective must be finite")

    objectives = []
    kls = []
    accepted_logits = None
    accepted_step = None
    for i in range(steps):
        factor = 2.0 ** -i
        if not math.isfinite(factor) or factor <= 0.0:
            raise ValueError("step factor must be finite and positive")
        candidate = []
        for s in range(n_rows):
            row = []
            for j in range(n_cols):
                value = logits[s][j] + factor * direction[s][j]
                if not math.isfinite(value):
                    raise ValueError("candidate logits must be finite")
                row.append(value)
            candidate.append(row)
        qx = [_log_softmax(row) for row in candidate]
        plain_candidate = [
            [float(value) for value in row] for row in candidate
        ]

        objective = 0.0
        for s, a, adv in samples:
            delta = qx[s][a] - q0[s][a]
            try:
                ratio = math.exp(delta)
            except OverflowError:
                raise ValueError("importance ratio must be finite")
            if not math.isfinite(ratio):
                raise ValueError("importance ratio must be finite")
            term = adv * ratio
            if not math.isfinite(term):
                raise ValueError("objective term must be finite")
            objective += term
            if not math.isfinite(objective):
                raise ValueError("objective sum must be finite")
        objective /= n_batch
        if not math.isfinite(objective):
            raise ValueError("objective must be finite")

        kl = ppo_policy_kl(plain_logits, plain_candidate)["mean"]
        if not math.isfinite(kl):
            raise ValueError("KL must be finite")

        objectives.append(objective)
        kls.append(kl)
        if kl <= max_kl and objective >= baseline:
            accepted_logits = candidate
            accepted_step = factor
            break

    if accepted_logits is None:
        return {
            "logits": [list(row) for row in logits],
            "objectives": objectives,
            "kls": kls,
            "accepted": False,
            "step": None,
        }
    return {
        "logits": accepted_logits,
        "objectives": objectives,
        "kls": kls,
        "accepted": True,
        "step": accepted_step,
    }


def _ppo_training_setup(env, seed):
    """初始化共享 PPO 训练状态：URDL 状态编号、零 logits/值表与随机源。"""
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
    return actions, states, state_index, h, values, rng


def _ppo_run_episode(
    env,
    actions,
    state_index,
    h,
    values,
    rng,
    max_steps,
    gamma,
    lambda_,
    alpha,
    c,
    epochs,
):
    """执行一个 PPO 回合（三个 PPO 入口共享的逐回合内核）。

    reset 后至 done 或 max_steps 步：每步按稳定 softmax(H[s,·])
    （列序 U/R/D/L）采样动作，仅消费一次 rng.random()，记录
    (s, a, r, old_logp, v, v2, done) 与 [r, c, action, reward, done]
    步项；done 时 v2=0.0，步限截断仍自举。回合末自 A=0.0 逆序递推
    GAE，调用 ppo_update 多轮截断更新，再按步序更新 V。返回
    (更新后的 h, 步表, objectives, 末步 done, 回报和)。
    """
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
    objectives = result["objectives"]

    for (s, _a, _r, _old_logp, value, _v2, _done), adv in zip(
        trajectory, advantages
    ):
        new_value = values[s] + alpha * (
            adv + value - values[s]
        )
        if not math.isfinite(new_value):
            raise ValueError("updated value must be finite")
        values[s] = new_value

    episode_done = trajectory[-1][6]
    episode_return = 0.0
    for (_s, _a, reward, _ol, _v, _v2, _d) in trajectory:
        episode_return += reward
    return h, step_records, objectives, episode_done, episode_return


def _ppo_h_table(h, states):
    """将 logits 表连同按坐标升序的状态编号整理为最终 h 表。"""
    return [
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


class _PpoEpisodeCollector:
    """固定回合收集器：累积步表与 objectives，停止谓词恒为假。"""

    def __init__(self):
        self.episodes = []
        self.objectives = []

    def record(self, step_records, objectives, done, episode_return):
        self.episodes.append(step_records)
        self.objectives.append(objectives)

    def should_stop(self):
        return False


class _PpoSuccessCollector:
    """成功率早停收集器：累积成功序列并逐回合调用 ppo_success_streak。"""

    def __init__(self, window, threshold, patience):
        self.episodes = []
        self.objectives = []
        self.successes = []
        self.convergence = None
        self._window = window
        self._threshold = threshold
        self._patience = patience

    def record(self, step_records, objectives, done, episode_return):
        self.episodes.append(step_records)
        self.objectives.append(objectives)
        self.successes.append(step_records[-1][4])
        self.convergence = ppo_success_streak(
            self.successes,
            self._window,
            self._threshold,
            self._patience,
        )

    def should_stop(self):
        return self.convergence["converged"]


class _PpoConvergeCollector:
    """成功率/回报双阈值早停收集器，维护完整滑窗与命中回合。"""

    def __init__(self, window, success, reward, patience, min_ep):
        self.episodes = []
        self.objectives = []
        self.dones = []
        self.returns = []
        self.windows = []
        self.converged_episode = None
        self._window = window
        self._success = success
        self._reward = reward
        self._patience = patience
        self._min_ep = min_ep
        self._streak = 0

    def record(self, step_records, objectives, done, episode_return):
        self.episodes.append(step_records)
        self.objectives.append(objectives)
        self.dones.append(done)
        self.returns.append(episode_return)

        end = len(self.returns)
        if end >= self._window:
            start = end - self._window + 1
            hits = 0
            for flag in self.dones[start - 1:end]:
                if flag:
                    hits += 1
            success_rate = hits / self._window
            window_return = 0.0
            for past_return in self.returns[start - 1:end]:
                window_return += past_return
            reward_mean = window_return / self._window
            passed = (
                success_rate >= self._success
                and reward_mean >= self._reward
            )
            self.windows.append(
                [start, end, success_rate, reward_mean, passed]
            )
            if passed:
                self._streak += 1
            else:
                self._streak = 0
            if end >= self._min_ep and self._streak >= self._patience:
                self.converged_episode = end

    def should_stop(self):
        return self.converged_episode is not None


def _ppo_run_episodes(
    env,
    limit,
    actions,
    state_index,
    h,
    values,
    rng,
    max_steps,
    gamma,
    lambda_,
    alpha,
    c,
    epochs,
    collector,
):
    """固定回合上限的逐回合驱动：每回合运行私有内核，再交由收集器
    record，并以其停止谓词 should_stop 决定是否早停。返回更新后的 h。"""
    for _ in range(limit):
        h, step_records, objectives, done, episode_return = (
            _ppo_run_episode(
                env,
                actions,
                state_index,
                h,
                values,
                rng,
                max_steps,
                gamma,
                lambda_,
                alpha,
                c,
                epochs,
            )
        )
        collector.record(
            step_records, objectives, done, episode_return
        )
        if collector.should_stop():
            break
    return h


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

    actions, states, state_index, h, values, rng = _ppo_training_setup(
        env, seed
    )
    collector = _PpoEpisodeCollector()
    h = _ppo_run_episodes(
        env,
        episodes,
        actions,
        state_index,
        h,
        values,
        rng,
        max_steps,
        gamma,
        lambda_,
        alpha,
        c,
        epochs,
        collector,
    )

    return {
        "h": _ppo_h_table(h, states),
        "episodes": collector.episodes,
        "objectives": collector.objectives,
    }


def trpo_train(
    env,
    episodes=100,
    alpha=0.05,
    gamma=0.9,
    lambda_=0.95,
    max_kl=0.01,
    steps=10,
    seed=0,
    max_steps=1000,
) -> dict:
    """TRPO 训练（每回合 GAE 采样后作信赖域回溯线搜索更新），返回 h 表、
    v 表、逐回合步表与目标/KL/接受记录。

    状态编号、H/V 初值、采样消费随机数、轨迹记录、GAE 递推、V 更新以及
    终止/截断处理逐项沿用 ppo_train：从 S 可达的非 G 格按坐标升序编号
    0..n-1，H 为 n×4 的零 logits 表（列序 U/R/D/L），V 为长度 n 的零
    值表；每回合 reset，至 done 或 max_steps 步；每步按稳定
    softmax(H[s,·]) 采样动作，仅消费一次 random()，记录
    (s, a, r, v, v2, done)，done 时 v2=0.0，步限截断仍自举且回合末自
    A=0.0 逆序递推，不跨回合。回合末按步序将 [s, a, float(A)] 批次传给
    trpo(H, batch, max_kl, steps)，以其返回 logits 续训，再按步序作
    V[s]+=alpha*(A+v-V[s])。全部随机性来自一个 random.Random(seed)。

    env、episodes、alpha、gamma、lambda_、seed、max_steps 的同名参数
    校验顺序与 TypeError/ValueError 沿用 ppo_train；max_kl、steps 沿用
    trpo（max_kl 为非 bool 的有限正 int/float；steps 为属于 [1, 61] 的
    非 bool int）。全部校验在首次 reset 前完成。更新器抛出的异常原样
    透传，且不返回部分结果。

    返回键依次为 h、v、episodes、objectives、kls、accepted、
    step_factors；h、v 行按坐标升序分别为
    [r, c, lU, lR, lD, lL] 与 [r, c, value]，均为 float；episodes 为
    各回合的 [r, c, action, reward, done] 步列表；后四项逐回合记录
    trpo 返回：objectives、kls 为 float 列表，accepted 为 bool，
    step_factors 为命中倍率 float 或 None。新运算结果非有限均抛
    ValueError。同参同 seed 逐值一致，仅用标准库。
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
    if isinstance(max_kl, bool) or not isinstance(
        max_kl, (int, float)
    ):
        raise TypeError("max_kl must be a non-bool int or float")
    if isinstance(steps, bool) or not isinstance(steps, int):
        raise TypeError("steps must be a non-bool int")
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
    try:
        max_kl = float(max_kl)
    except OverflowError:
        raise ValueError("max_kl must convert to a finite float")
    if not math.isfinite(max_kl) or max_kl <= 0.0:
        raise ValueError("max_kl must be finite and > 0")
    if steps < 1 or steps > 61:
        raise ValueError("steps must be in [1, 61]")

    actions, states, state_index, h, values, rng = _ppo_training_setup(
        env, seed
    )
    all_episodes = []
    all_objectives = []
    all_kls = []
    all_accepted = []
    all_step_factors = []

    for _ in range(episodes):
        state = env.reset()
        trajectory = []
        step_records = []
        for _ in range(max_steps):
            s = state_index[state]
            row = h[s]
            m = max(row)
            softmax_weights = [
                math.exp(logit - m) for logit in row
            ]
            total = sum(softmax_weights)
            if not math.isfinite(total) or total <= 0.0:
                raise ValueError(
                    "softmax normalizer must be finite and positive"
                )
            probs = [weight / total for weight in softmax_weights]
            u = rng.random()
            cumulative = 0.0
            a = 3
            for k, p_k in enumerate(probs):
                cumulative += p_k
                if cumulative > u:
                    a = k
                    break
            action = actions[a]
            value = values[s]
            next_state, reward, done = env.step(action)
            value_next = (
                0.0 if done else values[state_index[next_state]]
            )
            trajectory.append(
                (s, a, reward, value, value_next, done)
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
            [s, a, float(adv)]
            for (s, a, _r, _v, _v2, _done), adv in zip(
                trajectory, advantages
            )
        ]
        result = trpo(h, batch, max_kl, steps)
        h = result["logits"]
        all_objectives.append(result["objectives"])
        all_kls.append(result["kls"])
        all_accepted.append(result["accepted"])
        all_step_factors.append(result["step"])

        for (s, _a, _r, value, _v2, _done), adv in zip(
            trajectory, advantages
        ):
            new_value = values[s] + alpha * (
                adv + value - values[s]
            )
            if not math.isfinite(new_value):
                raise ValueError("updated value must be finite")
            values[s] = new_value

        all_episodes.append(step_records)

    v_table = [
        [float(r), float(c), values[i]]
        for i, (r, c) in enumerate(states)
    ]
    return {
        "h": _ppo_h_table(h, states),
        "v": v_table,
        "episodes": all_episodes,
        "objectives": all_objectives,
        "kls": all_kls,
        "accepted": all_accepted,
        "step_factors": all_step_factors,
    }


def advantage_weighted_train(
    env,
    episodes=100,
    alpha=0.05,
    gamma=0.9,
    lambda_=0.95,
    temp=1.0,
    cap=20.0,
    epochs=4,
    seed=0,
    max_steps=1000,
) -> dict:
    """优势加权训练（每回合 GAE 采样后作优势加权更新），返回 h 表、逐回合
    步表、目标值与权重。

    状态编号、H/V 初值、采样消费随机数、轨迹记录、GAE 递推、V 更新以及
    终止/截断处理逐项沿用 ppo_train：从 S 可达的非 G 格按坐标升序编号
    0..n-1，H 为 n×4 的零 logits 表（列序 U/R/D/L），V 为长度 n 的零
    值表；每回合 reset，至 done 或 max_steps 步；每步按稳定
    softmax(H[s,·]) 采样动作，仅消费一次 random()，记录
    (s, a, r, v, v2, done)，done 时 v2=0.0，步限截断仍自举且回合末自
    A=0.0 逆序递推，不跨回合。回合末按步序将 [s, a, float(A)] 批次传给
    advantage_weighted_update(H, batch, float(alpha), float(temp),
    float(cap), epochs)，以其返回 logits 续训，再按步序作
    V[s]+=alpha*(A+v-V[s])。全部随机性来自一个 random.Random(seed)。

    同名参数的校验顺序与 TypeError/ValueError 沿用 ppo_train；
    temp、cap 沿用 advantage_weighted_update（有限正 float），全部校验
    在首次 reset 前完成。更新器抛出的异常原样透传，且不返回部分结果。

    返回键依次为 h、episodes、objectives、weights；前三键结构沿用
    ppo_train（h 行为 [r, c, lU, lR, lD, lL]，episodes 为各回合
    [r, c, action, reward, done] 步列表，objectives 为各回合更新器返回
    的逐轮目标均值），weights 逐回合保留更新器返回的完整 float 权重
    列表。全部容器均为新建。
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
    if not isinstance(temp, float):
        raise TypeError("temp must be a float")
    if not math.isfinite(temp):
        raise ValueError("temp must be finite")
    if temp <= 0.0:
        raise ValueError("temp must be positive")
    if not isinstance(cap, float):
        raise TypeError("cap must be a float")
    if not math.isfinite(cap):
        raise ValueError("cap must be finite")
    if cap <= 0.0:
        raise ValueError("cap must be positive")
    if isinstance(epochs, bool) or not isinstance(epochs, int):
        raise TypeError("epochs must be a non-bool int")
    if epochs <= 0:
        raise ValueError("epochs must be positive")

    actions, states, state_index, h, values, rng = _ppo_training_setup(
        env, seed
    )
    all_episodes = []
    all_objectives = []
    all_weights = []

    for _ in range(episodes):
        state = env.reset()
        trajectory = []
        step_records = []
        for _ in range(max_steps):
            s = state_index[state]
            row = h[s]
            m = max(row)
            softmax_weights = [
                math.exp(logit - m) for logit in row
            ]
            total = sum(softmax_weights)
            if not math.isfinite(total) or total <= 0.0:
                raise ValueError(
                    "softmax normalizer must be finite and positive"
                )
            probs = [weight / total for weight in softmax_weights]
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
            value = values[s]
            next_state, reward, done = env.step(action)
            value_next = (
                0.0 if done else values[state_index[next_state]]
            )
            trajectory.append(
                (s, a, reward, value, value_next, done)
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
            [s, a, float(adv)]
            for (s, a, _r, _v, _v2, _done), adv in zip(
                trajectory, advantages
            )
        ]
        result = advantage_weighted_update(
            h,
            batch,
            float(alpha),
            float(temp),
            float(cap),
            epochs,
        )
        h = result["logits"]
        all_objectives.append(result["objectives"])
        all_weights.append(result["weights"])

        for (s, _a, _r, value, _v2, _done), adv in zip(
            trajectory, advantages
        ):
            new_value = values[s] + alpha * (
                adv + value - values[s]
            )
            if not math.isfinite(new_value):
                raise ValueError("updated value must be finite")
            values[s] = new_value

        all_episodes.append(step_records)

    return {
        "h": _ppo_h_table(h, states),
        "episodes": all_episodes,
        "objectives": all_objectives,
        "weights": all_weights,
    }


def awr_until(
    env,
    max_episodes=100,
    seed=0,
    max_steps=1000,
    window=20,
    threshold=.9,
    patience=3,
) -> dict:
    """优势加权训练至成功序列收敛或达回合上限，返回 h 表、逐回合步表、
    目标值、权重与收敛报告。

    env、seed、max_steps 的校验与异常沿用 advantage_weighted_train；
    max_episodes 按其 episodes 契约（非 bool 正 int）；window、
    patience 须为非 bool 正 int；threshold 须为非 bool 的 int/float，
    类型错抛 TypeError，转 float 溢出、非有限或越出 [0, 1] 抛
    ValueError。全部校验在首次 reset 前完成。

    以 advantage_weighted_train 默认训练超参（alpha=0.05、gamma=0.9、
    lambda_=0.95、temp=1.0、cap=20.0、epochs=4）及给定 seed、
    max_steps 连续训练：采样、GAE、advantage_weighted_update、H/V 更新、
    随机数消费及终止/截断处理与该函数逐值一致，全部随机性来自一个
    random.Random(seed)。

    每回合更新后以该回合末步 done 为成功标记（步限截断记 False），
    追加进累计成功序列并调用 ppo_success_streak(successes, window,
    threshold, patience)；首次 converged 为真即停止，否则训练
    max_episodes 回合。

    返回键依次为 h、episodes、objectives、weights、report；实际完成 k
    回合时，前四项逐值等于 advantage_weighted_train(env, episodes=k,
    seed=seed, max_steps=max_steps) 的对应返回，report 为成功序列的
    完整 ppo_success_streak 结果。不额外消费随机数；更新器异常原样
    透传且不返回部分结果，同参同 seed 逐值一致。仅用标准库。
    """
    if not isinstance(env, GridWorld):
        raise TypeError("env must be a GridWorld")
    if isinstance(max_episodes, bool) or not isinstance(max_episodes, int):
        raise TypeError("max_episodes must be an int")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an int")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int):
        raise TypeError("max_steps must be an int")
    if isinstance(window, bool) or not isinstance(window, int):
        raise TypeError("window must be a non-bool int")
    if isinstance(patience, bool) or not isinstance(patience, int):
        raise TypeError("patience must be a non-bool int")
    if isinstance(threshold, bool) or not isinstance(
        threshold, (int, float)
    ):
        raise TypeError("threshold must be an int or float")
    if max_episodes <= 0:
        raise ValueError("max_episodes must be positive")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if window <= 0:
        raise ValueError("window must be positive")
    if patience <= 0:
        raise ValueError("patience must be positive")
    try:
        float(threshold)
    except OverflowError:
        raise ValueError("threshold must convert to a finite float")
    if not math.isfinite(float(threshold)):
        raise ValueError("threshold must be finite")
    if float(threshold) < 0.0 or float(threshold) > 1.0:
        raise ValueError("threshold must be in [0, 1]")

    alpha = 0.05
    gamma = 0.9
    lambda_ = 0.95
    temp = 1.0
    cap = 20.0
    epochs = 4

    actions, states, state_index, h, values, rng = _ppo_training_setup(
        env, seed
    )
    all_episodes = []
    all_objectives = []
    all_weights = []
    successes = []
    report = None

    for _ in range(max_episodes):
        state = env.reset()
        trajectory = []
        step_records = []
        for _ in range(max_steps):
            s = state_index[state]
            row = h[s]
            m = max(row)
            softmax_weights = [
                math.exp(logit - m) for logit in row
            ]
            total = sum(softmax_weights)
            if not math.isfinite(total) or total <= 0.0:
                raise ValueError(
                    "softmax normalizer must be finite and positive"
                )
            probs = [weight / total for weight in softmax_weights]
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
            value = values[s]
            next_state, reward, done = env.step(action)
            value_next = (
                0.0 if done else values[state_index[next_state]]
            )
            trajectory.append(
                (s, a, reward, value, value_next, done)
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
            [s, a, float(adv)]
            for (s, a, _r, _v, _v2, _done), adv in zip(
                trajectory, advantages
            )
        ]
        result = advantage_weighted_update(
            h,
            batch,
            float(alpha),
            float(temp),
            float(cap),
            epochs,
        )
        h = result["logits"]
        all_objectives.append(result["objectives"])
        all_weights.append(result["weights"])

        for (s, _a, _r, value, _v2, _done), adv in zip(
            trajectory, advantages
        ):
            new_value = values[s] + alpha * (
                adv + value - values[s]
            )
            if not math.isfinite(new_value):
                raise ValueError("updated value must be finite")
            values[s] = new_value

        all_episodes.append(step_records)

        successes.append(step_records[-1][4])
        report = ppo_success_streak(
            successes, window, threshold, patience
        )
        if report["converged"]:
            break

    return {
        "h": _ppo_h_table(h, states),
        "episodes": all_episodes,
        "objectives": all_objectives,
        "weights": all_weights,
        "report": report,
    }


def ppo_converge(
    env,
    episodes=100,
    seed=0,
    max_steps=1000,
    window=20,
    success=.9,
    reward=-20.,
    patience=3,
    min_ep=20,
) -> dict:
    """PPO 训练带收敛早停，返回 h 表、逐回合步表、目标值与收敛报告。

    训练过程与 ppo_train(env, episodes, 0.05, 0.9, 0.95, 0.2, 4, seed,
    max_steps) 完全一致：同一 random.Random(seed)，不增随机数调用，
    实际运行 k 回合时，返回前三个键与 ppo_train 的 k 回合结果逐值一致。
    window、patience、min_ep 须为非 bool 的 int，且为正，否则按情形抛
    TypeError 或 ValueError；success、reward 须为非 bool 的 int/float，
    转 float 溢出或非有限抛 ValueError，success 还须落在 [0, 1]。全部
    校验在首次 reset 前完成。

    每回合更新后记录该回合是否 done（步限截断记 False）与回报和。
    end 回合（1 基）满 window 个时记窗
    [start, end, success_rate, reward_mean, passed]：start 为
    end-window+1，success_rate 为窗内 done 比例，reward_mean 自 0.0
    按序累加窗内回报和再除 window，passed 当 success_rate>=success
    且 reward_mean>=reward。首个 end>=min_ep 且连续 patience 窗
    passed 时停训，否则训满 episodes。

    返回键依次为 h、episodes、objectives、report；report 键依次为
    converged、episode、windows：windows 为窗列表，episode 为命中
    的 end 或 None，converged 等价于 episode 非 None。
    """
    if not isinstance(env, GridWorld):
        raise TypeError("env must be a GridWorld")
    if isinstance(episodes, bool) or not isinstance(episodes, int):
        raise TypeError("episodes must be an int")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an int")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int):
        raise TypeError("max_steps must be an int")
    if isinstance(window, bool) or not isinstance(window, int):
        raise TypeError("window must be an int")
    if isinstance(patience, bool) or not isinstance(patience, int):
        raise TypeError("patience must be an int")
    if isinstance(min_ep, bool) or not isinstance(min_ep, int):
        raise TypeError("min_ep must be an int")
    if isinstance(success, bool) or not isinstance(success, (int, float)):
        raise TypeError("success must be an int or float")
    if isinstance(reward, bool) or not isinstance(reward, (int, float)):
        raise TypeError("reward must be an int or float")
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if window <= 0:
        raise ValueError("window must be positive")
    if patience <= 0:
        raise ValueError("patience must be positive")
    if min_ep <= 0:
        raise ValueError("min_ep must be positive")
    try:
        success = float(success)
    except OverflowError:
        raise ValueError("success must be finite") from None
    if not math.isfinite(success):
        raise ValueError("success must be finite")
    if success < 0.0 or success > 1.0:
        raise ValueError("success must be in [0, 1]")
    try:
        reward = float(reward)
    except OverflowError:
        raise ValueError("reward must be finite") from None
    if not math.isfinite(reward):
        raise ValueError("reward must be finite")

    alpha = 0.05
    gamma = 0.9
    lambda_ = 0.95
    c = 0.2
    epochs = 4

    actions, states, state_index, h, values, rng = _ppo_training_setup(
        env, seed
    )
    collector = _PpoConvergeCollector(
        window, success, reward, patience, min_ep
    )
    h = _ppo_run_episodes(
        env,
        episodes,
        actions,
        state_index,
        h,
        values,
        rng,
        max_steps,
        gamma,
        lambda_,
        alpha,
        c,
        epochs,
        collector,
    )

    return {
        "h": _ppo_h_table(h, states),
        "episodes": collector.episodes,
        "objectives": collector.objectives,
        "report": {
            "converged": collector.converged_episode is not None,
            "episode": collector.converged_episode,
            "windows": collector.windows,
        },
    }


def ppo_train_trace_bytes(
    env,
    episodes=100,
    alpha=0.05,
    gamma=0.9,
    lambda_=0.95,
    c=0.2,
    epochs=4,
    seed=0,
    max_steps=1000,
) -> bytes:
    """PPO 训练并以规范单行 JSON 字节返回逐回合追踪。

    参数校验、异常、状态序、H/V 初值、采样与随机消费、GAE、更新、终止
    与截断完全沿用 ppo_train，首次 reset 前完成全部验参；同参同 seed
    与 ppo_train 产生相同的回合、目标与参数。

    输出对象仅含键 episodes，每回合为
    [steps, reward, success, rate, objectives, h_delta, v_delta]：
    steps 为实际步数 int；reward 为从整数 0 按步序顺序累加的未折扣
    回报 int；success 为末步 done；rate 为截至该回合（含）成功比例
    float；objectives 为本回合 ppo_update 返回的各轮目标值列表；
    h_delta 从 0.0 起按坐标升序 × U/R/D/L 列序累加本回合 H 更新前后
    各表项绝对差，v_delta 从 0.0 起按坐标升序累加本回合 V 更新前后
    各状态值绝对差。所有 float 均以 float.hex() 字符串表示（保留
    -0.0），其余字段依次为 int、int、bool 与字符串列表。

    以 (json.dumps(out, ensure_ascii=True, allow_nan=False,
    separators=(",", ":")) + "\\n").encode("utf-8") 返回：无额外空白，
    末尾恰一个 LF；同参同 seed 逐字节一致。仅用标准库，不引入命令行
    入口。
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
    trace_episodes = []
    successes = 0

    for ep_index in range(episodes):
        state = env.reset()
        trajectory = []
        total_reward = 0
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
            total_reward += reward
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
        values_before = list(values)
        result = ppo_update(h, batch, float(alpha), c, epochs)
        new_h = result["logits"]
        h_delta = 0.0
        for i in range(len(states)):
            old_row = h[i]
            new_row = new_h[i]
            for j in range(len(actions)):
                h_delta += abs(new_row[j] - old_row[j])
        h = new_h

        for (s, _a, _r, _old_logp, value, _v2, _done), adv in zip(
            trajectory, advantages
        ):
            new_value = values[s] + alpha * (
                adv + value - values[s]
            )
            if not math.isfinite(new_value):
                raise ValueError("updated value must be finite")
            values[s] = new_value

        v_delta = 0.0
        for i in range(len(states)):
            v_delta += abs(values[i] - values_before[i])

        success = trajectory[-1][6]
        if success:
            successes += 1
        rate = successes / (ep_index + 1)
        trace_episodes.append(
            [
                len(trajectory),
                total_reward,
                success,
                rate.hex(),
                [objective.hex() for objective in result["objectives"]],
                h_delta.hex(),
                v_delta.hex(),
            ]
        )

    out = {"episodes": trace_episodes}
    return (
        json.dumps(
            out,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _ppo_train_trace_canonical_float(text, name, nonnegative=False):
    """解析单个规范 float.hex() 字符串并返回其 float 值。"""
    if not isinstance(text, str):
        raise ValueError(f"{name} must be a canonical float.hex() str")
    try:
        value = float.fromhex(text)
    except (ValueError, OverflowError) as exc:
        raise ValueError(
            f"{name} must be a canonical float.hex() str"
        ) from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value.hex() != text:
        raise ValueError(f"{name} must be a canonical float.hex() str")
    if nonnegative and value < 0.0:
        raise ValueError(f"{name} must be >= 0.0")
    return value


def _ppo_train_trace_validate(data):
    """结构与取值校验，返回 episodes 列表。"""
    if not isinstance(data, dict) or list(data) != ["episodes"]:
        raise ValueError("data must have exactly the key episodes")
    episodes = data["episodes"]
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("episodes must be a non-empty list")
    for ep_index, row in enumerate(episodes):
        if not isinstance(row, list) or len(row) != 7:
            raise ValueError(
                f"episodes[{ep_index}] must be a list of exactly 7 items"
            )
        steps, reward, success, rate, objectives, h_delta, v_delta = row
        if (
            isinstance(steps, bool)
            or not isinstance(steps, int)
            or steps <= 0
        ):
            raise ValueError(
                f"episodes[{ep_index}] steps must be a non-bool positive"
                " int"
            )
        if isinstance(reward, bool) or not isinstance(reward, int):
            raise ValueError(
                f"episodes[{ep_index}] reward must be a non-bool int"
            )
        if not isinstance(success, bool):
            raise ValueError(
                f"episodes[{ep_index}] success must be a bool"
            )
        rate_value = _ppo_train_trace_canonical_float(
            rate, f"episodes[{ep_index}] rate"
        )
        if rate_value < 0.0 or rate_value > 1.0:
            raise ValueError(
                f"episodes[{ep_index}] rate must be in [0, 1]"
            )
        if not isinstance(objectives, list) or not objectives:
            raise ValueError(
                f"episodes[{ep_index}] objectives must be a non-empty"
                " list"
            )
        for obj_index, objective in enumerate(objectives):
            _ppo_train_trace_canonical_float(
                objective,
                f"episodes[{ep_index}] objectives[{obj_index}]",
            )
        _ppo_train_trace_canonical_float(
            h_delta, f"episodes[{ep_index}] h_delta", nonnegative=True
        )
        _ppo_train_trace_canonical_float(
            v_delta, f"episodes[{ep_index}] v_delta", nonnegative=True
        )
    return episodes


def ppo_train_trace_from_bytes(payload) -> dict:
    """将 ppo_train_trace_bytes 的产物严格解析回逐回合训练追踪。

    payload 须恰为 bytes（bytearray、memoryview 等均拒绝），否则抛
    TypeError；空字节、严格 UTF-8 解码失败、BOM、JSON 语法错误或尾随
    内容、重复对象键、NaN/Infinity/-Infinity、根值非 dict、结构或编码
    违约均抛 ValueError。

    仅接受规范单行紧凑 ASCII JSON：末尾恰一个 LF，无 BOM 或额外空白/
    换行。顶层须仅含键 episodes，值为非空 list；每行须恰为七项 list
    [steps, reward, success, rate, objectives, h_delta, v_delta]：
    steps 为非 bool 正 int，reward 为非 bool int，success 为 bool，
    objectives 为非空 list；rate、objectives 各项及两个 delta 均须为
    有限 float 的规范 float.hex() 字符串，rate 值须在 [0, 1] 内，两个
    delta 须 >= 0.0。按 ppo_train_trace_bytes 生成器规则重编码后的字节
    须与输入逐字节相同。

    返回解析得到的独立新容器，同一 payload 多次解析逐值一致。仅用标准
    库，不引入命令行入口。
    """
    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    if not payload:
        raise ValueError("payload must not be empty")
    if payload.startswith(_UTF8_BOM):
        raise ValueError("payload must not start with a BOM")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("payload must be valid UTF-8") from exc
    if not text.endswith("\n"):
        raise ValueError("payload must end with exactly one LF")
    body = text[:-1]

    def reject_constant(value):
        raise ValueError(f"invalid JSON constant: {value}")

    def reject_duplicate_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate object key: {key!r}")
            result[key] = value
        return result

    decoder = json.JSONDecoder(
        object_pairs_hook=reject_duplicate_keys,
        parse_constant=reject_constant,
    )
    try:
        data, end = decoder.raw_decode(body)
    except RecursionError as exc:
        raise ValueError("payload nesting too deep") from exc
    except json.JSONDecodeError as exc:
        raise ValueError("payload must be valid JSON") from exc
    if end != len(body):
        raise ValueError("payload must not contain trailing content")
    if not isinstance(data, dict):
        raise ValueError("payload root value must be a dict")
    _ppo_train_trace_validate(data)

    canonical = (
        json.dumps(
            data,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    if canonical != payload:
        raise ValueError(
            "payload must be canonical ppo_train_trace_bytes output"
        )
    return data


def ppo_train_trace_verify(payload) -> dict:
    """严格解析 PPO 训练追踪字节并校验逐回合汇总一致性。

    先执行 ppo_train_trace_from_bytes 的全部字节、JSON、结构与编码
    校验（TypeError/ValueError 原样透传），再校验跨回合约束：第 i 行
    （0 基）的 rate 须等于截至该行累计成功数 / (i + 1) 的 float.hex()
    字符串，各行 objectives 长度须相等，否则抛 ValueError。

    返回键序恰为 episodes、steps、successes、success_rate、
    fingerprint 的 dict，依次为回合数、各行 steps 之和、成功回合数
    （int）、最终成功率（float）、输入字节的 hashlib.sha256 小写 64 位
    十六进制摘要。重复调用逐值一致。仅用标准库，不引入命令行入口。
    """
    data = ppo_train_trace_from_bytes(payload)
    episodes = data["episodes"]

    total_steps = 0
    successes = 0
    objective_count = None
    for ep_index, row in enumerate(episodes):
        steps, _reward, success, rate, objectives, _h, _v = row
        total_steps += steps
        if success:
            successes += 1
        expected_rate = (successes / (ep_index + 1)).hex()
        if rate != expected_rate:
            raise ValueError(
                f"episodes[{ep_index}] rate must be {expected_rate},"
                f" got {rate}"
            )
        if objective_count is None:
            objective_count = len(objectives)
        elif len(objectives) != objective_count:
            raise ValueError(
                f"episodes[{ep_index}] objectives length must equal the"
                f" first row's ({objective_count})"
            )

    digest = hashlib.sha256(payload).hexdigest()
    return {
        "episodes": len(episodes),
        "steps": total_steps,
        "successes": successes,
        "success_rate": successes / len(episodes),
        "fingerprint": digest,
    }


def ppo_train_adaptive_kl(
    env,
    episodes=100,
    alpha=0.05,
    gamma=0.9,
    lambda_=0.95,
    beta=1.0,
    target=0.01,
    factor=2.0,
    lo=1e-4,
    hi=1e4,
    epochs=4,
    seed=0,
    max_steps=1000,
) -> dict:
    """自适应 KL 惩罚系数的 PPO 训练（每回合 GAE 采样后多轮 KL 惩罚更新），
    返回 h 表、逐回合步表、目标值、KL 与惩罚系数。

    env、episodes、alpha、gamma、lambda_、epochs、seed、max_steps 的
    校验与异常逐项沿用 ppo_train；beta、target、factor、lo、hi 的校验与
    异常同 ppo_adaptive_kl_update：须为非 bool 的 int/float，类型不符抛
    TypeError，转 float 溢出或非有限，beta、target、lo、hi <= 0，
    factor <= 1，或 beta 不在 [lo, hi] 内均抛 ValueError。全部校验在
    采样前完成。

    状态编号、H/V 初值、每回合 reset、softmax(URDL) 采样且每步仅一次
    random()、GAE 逆推、终止/截断自举、V 的步序更新均同 ppo_train。记
    b=float(beta)，每回合将步序批次 [s, a, old_logp, A] 传给
    ppo_adaptive_kl_update(H, batch, float(alpha), b, target, factor,
    lo, hi, epochs)，以其返回 logits 续训，记录该次返回的 objectives、
    kls、betas 三组历史，并以 betas 末项续作下一回合的 b；更新不额外
    消费随机数。异常直接透传，失败时无部分结果。

    返回键依次为 h、episodes、objectives、kls、betas；h 与 episodes 同
    ppo_train，后三项均与回合数等长，第 e 项为该回合更新返回的完整
    float 列表。同参同 seed 逐值一致。
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
    if isinstance(epochs, bool) or not isinstance(epochs, int):
        raise TypeError("epochs must be a non-bool int")
    if epochs <= 0:
        raise ValueError("epochs must be positive")

    numeric_params = (
        ("beta", beta),
        ("target", target),
        ("factor", factor),
        ("lo", lo),
        ("hi", hi),
    )
    for name, value in numeric_params:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a non-bool int or float")

    numeric_values = {}
    for name, value in numeric_params:
        try:
            converted = float(value)
        except OverflowError:
            raise ValueError(f"{name} must convert to a finite float")
        if not math.isfinite(converted):
            raise ValueError(f"{name} must be finite")
        numeric_values[name] = converted
    b = float(numeric_values["beta"])
    t = numeric_values["target"]
    factor_value = numeric_values["factor"]
    lo_value = numeric_values["lo"]
    hi_value = numeric_values["hi"]

    if b <= 0.0:
        raise ValueError("beta must be positive")
    if t <= 0.0:
        raise ValueError("target must be positive")
    if factor_value <= 1.0:
        raise ValueError("factor must be greater than 1")
    if lo_value <= 0.0:
        raise ValueError("lo must be positive")
    if hi_value <= 0.0:
        raise ValueError("hi must be positive")
    if not lo_value <= b <= hi_value:
        raise ValueError("beta must be within [lo, hi]")

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
    all_kls = []
    all_betas = []

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
        for t_ in range(len(trajectory) - 1, -1, -1):
            (
                _s,
                _a,
                reward,
                _old_logp,
                value,
                value_next,
                _done,
            ) = trajectory[t_]
            delta = reward + gamma * value_next - value
            if not math.isfinite(delta):
                raise ValueError("GAE delta must be finite")
            a_t = delta + gamma * lambda_ * a_t
            if not math.isfinite(a_t):
                raise ValueError("GAE advantage must be finite")
            advantages[t_] = a_t

        batch = [
            [s, a, old_logp, adv]
            for (s, a, _r, old_logp, _v, _v2, _done), adv in zip(
                trajectory, advantages
            )
        ]
        result = ppo_adaptive_kl_update(
            h,
            batch,
            float(alpha),
            b,
            t,
            factor_value,
            lo_value,
            hi_value,
            epochs,
        )
        h = result["logits"]
        all_objectives.append(result["objectives"])
        all_kls.append(result["kls"])
        all_betas.append(result["betas"])
        b = result["betas"][-1]

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
        "kls": all_kls,
        "betas": all_betas,
    }


def ppo_train_minibatch_adaptive_kl(
    env,
    episodes=100,
    alpha=0.05,
    gamma=0.9,
    lambda_=0.95,
    batch_size=32,
    beta=1.0,
    target=0.01,
    factor=2.0,
    lo=1e-4,
    hi=1e4,
    epochs=4,
    seed=0,
    shuffle_seed=0,
    max_steps=1000,
) -> dict:
    """自适应 KL 惩罚系数的小批量 PPO 训练（每回合 GAE 采样后多轮小批量
    KL 惩罚更新），返回 h/v 表、逐回合步表、目标值、KL 与惩罚系数。

    env、episodes、alpha、gamma、lambda_、epochs、seed、max_steps 的
    校验与异常逐项沿用 ppo_train_adaptive_kl；beta、target、factor、lo、
    hi 的校验与异常同 ppo_adaptive_kl_update：须为非 bool 的 int/float，
    类型不符抛 TypeError，转 float 溢出或非有限，beta、target、lo、hi
    <= 0，factor <= 1，或 beta 不在 [lo, hi] 内均抛 ValueError。
    batch_size 须为非 bool 的正 int，shuffle_seed 须为非 bool int，类型
    不符抛 TypeError，batch_size 非正抛 ValueError。全部校验在采样前
    完成。

    状态编号、H/V 初值、每回合 reset、softmax(URDL) 采样且每步仅一次
    random.Random(seed).random()、GAE 逆推、终止/截断自举、V 的步序更新
    均同 ppo_train_adaptive_kl。记 b=float(beta)，每回合将步序批次
    [s, a, old_logp, A] 传给
    ppo_minibatch_adaptive_kl(H, batch, batch_size, lr=float(alpha),
    beta=b, target, factor, lo, hi, epochs, seed=shuffle_seed+e)，同名
    参数照传，以其返回 logits 续训，记录该次返回的 objectives、kls、
    betas 三组历史，并以 betas 末项续作下一回合的 b；洗牌仅由
    shuffle_seed+e 控制，与采样随机流互不消费、彼此隔离。异常直接透传，
    失败时无部分结果。

    返回键依次为 h、v、episodes、objectives、kls、betas；h、v、episodes
    同 ppo_train_value_clip，后三项均与回合数等长，第 e 项为该回合
    ppo_minibatch_adaptive_kl 返回的完整 float 列表。同参同 seed 与
    shuffle_seed 逐值一致。
    """
    if not isinstance(env, GridWorld):
        raise TypeError("env must be a GridWorld")
    if isinstance(episodes, bool) or not isinstance(episodes, int):
        raise TypeError("episodes must be an int")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an int")
    if isinstance(shuffle_seed, bool) or not isinstance(shuffle_seed, int):
        raise TypeError("shuffle_seed must be an int")
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
    if isinstance(epochs, bool) or not isinstance(epochs, int):
        raise TypeError("epochs must be a non-bool int")
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError("batch_size must be a non-bool int")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    numeric_params = (
        ("beta", beta),
        ("target", target),
        ("factor", factor),
        ("lo", lo),
        ("hi", hi),
    )
    for name, value in numeric_params:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a non-bool int or float")

    numeric_values = {}
    for name, value in numeric_params:
        try:
            converted = float(value)
        except OverflowError:
            raise ValueError(f"{name} must convert to a finite float")
        if not math.isfinite(converted):
            raise ValueError(f"{name} must be finite")
        numeric_values[name] = converted
    b = float(numeric_values["beta"])
    t = numeric_values["target"]
    factor_value = numeric_values["factor"]
    lo_value = numeric_values["lo"]
    hi_value = numeric_values["hi"]

    if b <= 0.0:
        raise ValueError("beta must be positive")
    if t <= 0.0:
        raise ValueError("target must be positive")
    if factor_value <= 1.0:
        raise ValueError("factor must be greater than 1")
    if lo_value <= 0.0:
        raise ValueError("lo must be positive")
    if hi_value <= 0.0:
        raise ValueError("hi must be positive")
    if not lo_value <= b <= hi_value:
        raise ValueError("beta must be within [lo, hi]")

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
    all_kls = []
    all_betas = []

    for e in range(episodes):
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
        for t_ in range(len(trajectory) - 1, -1, -1):
            (
                _s,
                _a,
                reward,
                _old_logp,
                value,
                value_next,
                _done,
            ) = trajectory[t_]
            delta = reward + gamma * value_next - value
            if not math.isfinite(delta):
                raise ValueError("GAE delta must be finite")
            a_t = delta + gamma * lambda_ * a_t
            if not math.isfinite(a_t):
                raise ValueError("GAE advantage must be finite")
            advantages[t_] = a_t

        batch = [
            [s, a, old_logp, adv]
            for (s, a, _r, old_logp, _v, _v2, _done), adv in zip(
                trajectory, advantages
            )
        ]
        result = ppo_minibatch_adaptive_kl(
            h,
            batch,
            batch_size,
            lr=float(alpha),
            beta=b,
            target=t,
            factor=factor_value,
            lo=lo_value,
            hi=hi_value,
            epochs=epochs,
            seed=shuffle_seed + e,
        )
        h = result["logits"]
        all_objectives.append(result["objectives"])
        all_kls.append(result["kls"])
        all_betas.append(result["betas"])
        b = result["betas"][-1]

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
    v_table = [
        [float(r), float(c), values[i]]
        for i, (r, c) in enumerate(states)
    ]
    return {
        "h": h_table,
        "v": v_table,
        "episodes": all_episodes,
        "objectives": all_objectives,
        "kls": all_kls,
        "betas": all_betas,
    }


def ppo_train_entropy(
    env,
    episodes=100,
    alpha=0.05,
    gamma=0.9,
    lambda_=0.95,
    c=0.2,
    entropy_coef=0.01,
    epochs=4,
    seed=0,
    max_steps=1000,
) -> dict:
    """PPO 训练（每回合 GAE 采样后多轮熵正则截断更新），返回 h 表、步表、
    目标值与熵值。

    参数校验与异常逐项沿用 ppo_train；entropy_coef 沿 ppo_update_entropy，
    须为非 bool 的 int/float，类型不符抛 TypeError，转 float 溢出、非有限
    或越界 [0, 1] 抛 ValueError，校验后记为浮点系数，全部校验在采样前完成。

    状态编号、H/V 初始化、每回合 reset、softmax(URDL) 采样且每步仅一次
    random.Random(seed).random()、(s, a, r, logp, V[s], v2, done) 记录、
    done 时 v2=0 否则取采样时 V[s2]、回合末自 A=0 逆序递推
    delta=r+gamma*v2-v、A=delta+gamma*lambda_*A，以及 done 即停、步限末步
    未 done 仍自举，均逐项沿用 ppo_train。回合批次 [s, a, old_logp, A]
    传给 ppo_update_entropy(H, batch, float(alpha), c, entropy_coef,
    epochs)，以其返回 logits 续训，随后按步序作
    V[s]+=alpha*(A+v-V[s])。新运算结果非有限均抛 ValueError。

    返回键依次为 h、episodes、objectives、entropies；h 行按坐标升序为
    [r, c, lU, lR, lD, lL]，均为 float；episodes 为各回合的步列表，步项
    为 [r, c, action, reward, done]；objectives、entropies 分别为各回合
    ppo_update_entropy 返回的 objectives、entropies。同 seed 逐值一致。
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
    if isinstance(entropy_coef, bool) or not isinstance(
        entropy_coef, (int, float)
    ):
        raise TypeError("entropy_coef must be an int or float")
    try:
        entropy_coef = float(entropy_coef)
    except OverflowError:
        raise ValueError("entropy_coef must convert to a finite float")
    if not math.isfinite(entropy_coef):
        raise ValueError("entropy_coef must be finite")
    if entropy_coef < 0.0 or entropy_coef > 1.0:
        raise ValueError("entropy_coef must be in [0, 1]")

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
    all_entropies = []

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
        result = ppo_update_entropy(
            h, batch, float(alpha), c, entropy_coef, epochs
        )
        h = result["logits"]
        all_objectives.append(result["objectives"])
        all_entropies.append(result["entropies"])

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
        "entropies": all_entropies,
    }


def ppo_train_value_clip(
    env,
    episodes=100,
    alpha=0.05,
    value_lr=0.1,
    gamma=0.9,
    lambda_=0.95,
    c=0.2,
    value_clip=0.2,
    epochs=4,
    seed=0,
    max_steps=1000,
) -> dict:
    """PPO 训练（策略更新后作截断值函数更新），返回 h/v 表、步表与目标值。

    参数校验沿 ppo_train；value_lr、value_clip 分别沿
    ppo_value_update 的 lr、clip，均须为非 bool 有限 float，依次属于
    (0, 1]、[0, 1)，类型不符抛 TypeError，非有限或越界抛 ValueError。
    状态编号、H/V 初始化、softmax 采样、GAE 轨迹、回合边界与每步仅
    一次 random.Random(seed).random() 均沿 ppo_train。每步记录的 v2
    在 done 时为 0.0，否则为当时 V[s2]（步限截断仍自举），回合末逆
    序递推 GAE 优势 A。先以 [s, a, old_logp, A] 批次调用
    ppo_update(H, batch, float(alpha), c, epochs)，以其返回 logits
    续训 H；再以按步序的 [s, 旧 v, R] 批次（R=A+采样时记录的旧 v）
    调用
    ppo_value_update(V, batch, float(value_lr), value_clip, epochs)，
    以其返回 values 续训 V。两次更新调用均透传异常，无部分结果。

    返回键依次为 h、v、episodes、objectives；h 行、episodes 步结构
    沿 ppo_train，v 行按坐标升序为 [r, c, value]，均为更新后的
    float；objectives 为各回合 ppo_update 返回的 objectives。同参
    同 seed 逐值一致。
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
    if not isinstance(value_lr, float):
        raise TypeError("value_lr must be a float")
    if not math.isfinite(value_lr):
        raise ValueError("value_lr must be finite")
    if value_lr <= 0.0 or value_lr > 1.0:
        raise ValueError("value_lr must be in (0, 1]")
    if not isinstance(value_clip, float):
        raise TypeError("value_clip must be a float")
    if not math.isfinite(value_clip):
        raise ValueError("value_clip must be finite")
    if value_clip < 0.0 or value_clip >= 1.0:
        raise ValueError("value_clip must be in [0, 1)")
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

        policy_batch = [
            [s, a, old_logp, adv]
            for (s, a, _r, old_logp, _v, _v2, _done), adv in zip(
                trajectory, advantages
            )
        ]
        result = ppo_update(h, policy_batch, float(alpha), c, epochs)
        h = result["logits"]
        all_objectives.append(result["objectives"])

        value_batch = [
            [s, value, adv + value]
            for (s, _a, _r, _old_logp, value, _v2, _done), adv in zip(
                trajectory, advantages
            )
        ]
        value_result = ppo_value_update(
            values,
            value_batch,
            float(value_lr),
            value_clip,
            epochs,
        )
        values = value_result["values"]
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
    v_table = [
        [float(r), float(c), values[i]]
        for i, (r, c) in enumerate(states)
    ]
    return {
        "h": h_table,
        "v": v_table,
        "episodes": all_episodes,
        "objectives": all_objectives,
    }


def natural_actor_critic(
    env,
    episodes=500,
    alpha=.05,
    gamma=.9,
    lambda_=.95,
    damping=.001,
    seed=0,
    max_steps=1000,
) -> dict:
    """自然梯度 Actor-Critic 训练，返回 h/v 表与逐回合步表。

    参数校验沿 ppo_train（env、episodes、alpha、gamma、lambda_、seed、
    max_steps）；damping 沿 natural_gradient，须为非 bool 的
    int/float，转 float 后有限且大于 0，错型抛 TypeError，转换溢出
    或取值非法抛 ValueError。全部校验在首次 reset 前完成。

    状态编号、H/V 零初值、URDL 稳定 softmax 采样、每步仅一次
    random.Random(seed).random()、done 终止与步限截断均沿 ppo_train。
    每回合记录 (s, a, r, 旧 v, v2)：旧 v 为采样时 V[s]，done 时
    v2=0.0，否则为当时 V[s2]（步限截断仍自举）。回合末自 A=0.0 逆序
    递推 d=r+gamma*v2-v、A=d+gamma*lambda_*A；再将按步序的
    [s, a, float(A)] 批次一次传给
    natural_gradient(H, batch, float(alpha), float(damping), 1e-10,
    100)，以其返回 logits 续训 H；随后按步序作
    V[s]+=alpha*(A+旧v-V[s])。natural_gradient 抛出的异常原样透传，
    无部分结果。

    返回键依次为 h、v、episodes，结构沿 ppo_train_value_clip 前三
    项：h 行按坐标升序为 [r, c, lU, lR, lD, lL]，v 行为
    [r, c, value]，均为 float；episodes 步项为
    [r, c, action, reward, done]，action 为 U/R/D/L 字符串。新运算
    结果非有限均抛 ValueError。同参同 seed 逐值一致。
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
    if isinstance(damping, bool) or not isinstance(damping, (int, float)):
        raise TypeError("damping must be a non-bool int or float")
    try:
        damping = float(damping)
    except OverflowError:
        raise ValueError("damping must convert to a finite float")
    if not math.isfinite(damping) or damping <= 0.0:
        raise ValueError("damping must be finite and positive")

    actions, states, state_index, h, values, rng = _ppo_training_setup(
        env, seed
    )
    all_episodes = []

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
            u = rng.random()
            cumulative = 0.0
            a = 3
            for k, p_k in enumerate(probs):
                cumulative += p_k
                if cumulative > u:
                    a = k
                    break
            action = actions[a]
            value = values[s]
            next_state, reward, done = env.step(action)
            value_next = 0.0 if done else values[state_index[next_state]]
            trajectory.append((s, a, reward, value, value_next))
            step_records.append(
                [state[0], state[1], action, reward, done]
            )
            if done:
                break
            state = next_state

        advantages = [0.0] * len(trajectory)
        a_t = 0.0
        for t in range(len(trajectory) - 1, -1, -1):
            _s, _a, reward, value, value_next = trajectory[t]
            delta = reward + gamma * value_next - value
            if not math.isfinite(delta):
                raise ValueError("GAE delta must be finite")
            a_t = delta + gamma * lambda_ * a_t
            if not math.isfinite(a_t):
                raise ValueError("GAE advantage must be finite")
            advantages[t] = a_t

        batch = [
            [s, a, float(adv)]
            for (s, a, _r, _v, _v2), adv in zip(trajectory, advantages)
        ]
        result = natural_gradient(
            h, batch, float(alpha), float(damping), 1e-10, 100
        )
        h = result["logits"]

        for (s, _a, _r, value, _v2), adv in zip(trajectory, advantages):
            new_value = values[s] + alpha * (
                adv + value - values[s]
            )
            if not math.isfinite(new_value):
                raise ValueError("updated value must be finite")
            values[s] = new_value

        all_episodes.append(step_records)

    h_table = _ppo_h_table(h, states)
    v_table = [
        [float(r), float(c), values[i]]
        for i, (r, c) in enumerate(states)
    ]
    return {
        "h": h_table,
        "v": v_table,
        "episodes": all_episodes,
    }


def ppo_train_grad_clip(
    env,
    episodes=100,
    alpha=0.05,
    gamma=0.9,
    lambda_=0.95,
    c=0.2,
    epochs=4,
    max_norm=1.0,
    seed=0,
    max_steps=1000,
) -> dict:
    """带梯度范数裁剪的 PPO 训练，返回 h 表、逐回合步表与各回合更新统计。

    除 max_norm 外，参数校验、状态编号、采样、GAE、V 更新、终止/截断
    及返回中的 h、episodes 逐项沿用 ppo_train，且全部校验在采样前完成；
    max_norm 须为非 bool 的 int/float，类型不符抛 TypeError，转 float
    溢出、非有限或 <=0 抛 ValueError。每回合按步序构造
    [s, a, old_logp, A] 批次，调用
    ppo_grad_clip(H, batch, float(alpha), c, epochs, max_norm)，以其
    返回 logits 续训 H；该调用不消费随机数，全部随机性仍来自一个
    random.Random(seed)。

    返回键依次为 h、episodes、objectives、norms、scales；前两项沿用
    ppo_train，后三项与回合数等长，第 e 项为该回合 ppo_grad_clip 返回
    的同名完整 float 列表。新运算结果非有限均抛 ValueError 且无部分
    结果；同参同 seed 逐值一致。
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
    if isinstance(max_norm, bool) or not isinstance(max_norm, (int, float)):
        raise TypeError("max_norm must be an int or float")
    try:
        checked_max_norm = float(max_norm)
    except OverflowError:
        raise ValueError("max_norm must convert to a finite float")
    if not math.isfinite(checked_max_norm) or checked_max_norm <= 0.0:
        raise ValueError("max_norm must be finite and > 0")

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
    all_norms = []
    all_scales = []

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
        result = ppo_grad_clip(h, batch, float(alpha), c, epochs, max_norm)
        h = result["logits"]
        all_objectives.append(result["objectives"])
        all_norms.append(result["norms"])
        all_scales.append(result["scales"])

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
        "norms": all_norms,
        "scales": all_scales,
    }


def ppo_train_normalized(
    env,
    episodes=100,
    alpha=0.05,
    gamma=0.9,
    lambda_=0.95,
    c=0.2,
    epochs=4,
    epsilon=1e-8,
    seed=0,
    max_steps=1000,
) -> dict:
    """优势规范化的 PPO 训练（每回合 GAE 采样后规范化多轮截断更新），
    返回 h 表、逐回合步表与目标值。

    状态编号、H/V 初始化、采样、GAE、终止/截断、V 更新、随机消费及
    返回结构均逐项沿用 ppo_train；除 epsilon 外参数校验及异常逐项
    沿用 ppo_train，全部校验在采样前完成。epsilon 须为非 bool 的
    int/float，类型不符抛 TypeError，转 float 溢出、非有限或 <=0
    抛 ValueError。每回合以优势 A 按步序构造 [s, a, old_logp, A]
    批次，调用
    ppo_normalized_update(H, batch, float(alpha), c, epochs, epsilon)，
    以其返回 logits 续训；V 仍按原 A 与记录的旧 v 作
    V[s]+=alpha*(A+v-V[s])，规范化优势不用于 V 或额外采样。

    返回键依次为 h、episodes、objectives；objectives[e] 为第 e 回合
    ppo_normalized_update 返回的 objectives float 列表。新运算结果
    非有限均抛 ValueError，失败无部分结果；同参同 seed 逐值一致。
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
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)):
        raise TypeError("epsilon must be an int or float")
    try:
        epsilon = float(epsilon)
    except OverflowError:
        raise ValueError("epsilon must convert to a finite float")
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and > 0")

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
        result = ppo_normalized_update(
            h, batch, float(alpha), c, epochs, epsilon
        )
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


def ppo_train_minibatch(
    env,
    episodes=100,
    alpha=0.05,
    gamma=0.9,
    lambda_=0.95,
    c=0.2,
    epochs=4,
    batch_size=32,
    seed=0,
    max_steps=1000,
) -> dict:
    """小批量 PPO 训练（每回合 GAE 采样后多轮小批量截断更新），返回
    h 表、逐回合步表与目标值。

    状态编号、采样、GAE、终止/截断、V 更新及返回的 h、episodes 均
    逐项沿用 ppo_train；batch_size 须为非 bool 的正 int，类型不符
    抛 TypeError，非正抛 ValueError，其余参数校验逐项沿用
    ppo_train，全部校验在采样前完成。第 e 回合（0 起）以步序
    [s, a, old_logp, A] 为 batch，调用
    ppo_minibatch(H, batch, batch_size, float(alpha), c, epochs,
    seed+e)，以其返回 logits 续训；seed+e 只控制该回合洗牌，采样
    随机流不额外消费。返回键依次为 h、episodes、objectives；
    objectives 为各回合 ppo_minibatch 返回的完整 objectives。
    新运算结果非有限均抛 ValueError，失败无部分结果；同参同
    seed 逐值一致。
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
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError("batch_size must be a non-bool int")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

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

    for e in range(episodes):
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
        result = ppo_minibatch(
            h, batch, batch_size, float(alpha), c, epochs, seed + e
        )
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


def ppo_train_minibatch_value_clip(
    env,
    episodes=100,
    alpha=0.05,
    value_lr=0.1,
    gamma=0.9,
    lambda_=0.95,
    c=0.2,
    value_clip=0.2,
    epochs=4,
    batch_size=32,
    seed=0,
    shuffle_seed=0,
    max_steps=1000,
) -> dict:
    """小批量 PPO 训练（策略更新后作截断值函数更新），返回 h/v 表、
    步表与目标值。

    参数校验沿 ppo_train_value_clip；batch_size 须为非 bool 的正
    int，类型不符抛 TypeError，非正抛 ValueError；shuffle_seed 须
    为非 bool 的 int（允许负值），类型不符抛 TypeError；全部校验在
    采样前完成。状态编号、H/V 初始化、softmax 采样、GAE 轨迹、回合
    边界与每步仅一次 random.Random(seed).random() 均沿
    ppo_train_value_clip。第 e 回合（0 起）以步序
    [s, a, old_logp, A] 为 batch，调用
    ppo_minibatch(H, batch, batch_size, float(alpha), c, epochs,
    shuffle_seed+e)，以其返回 logits 续训 H；shuffle_seed+e 只控
    制该回合洗牌，采样随机流不额外消费。再以按步序的
    [s, 旧 v, A+旧v] 批次调用
    ppo_value_update(V, batch, float(value_lr), value_clip,
    epochs)，以其返回 values 续训 V。两次更新调用均透传异常，无
    部分结果。

    返回键依次为 h、v、episodes、objectives；h、v、episodes 结构
    沿 ppo_train_value_clip，objectives 为各回合 ppo_minibatch 返
    回的完整小批 objectives 列表。同参两 seed（seed 与
    shuffle_seed）逐值一致。
    """
    if not isinstance(env, GridWorld):
        raise TypeError("env must be a GridWorld")
    if isinstance(episodes, bool) or not isinstance(episodes, int):
        raise TypeError("episodes must be an int")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an int")
    if isinstance(shuffle_seed, bool) or not isinstance(shuffle_seed, int):
        raise TypeError("shuffle_seed must be an int")
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
    if not isinstance(value_lr, float):
        raise TypeError("value_lr must be a float")
    if not math.isfinite(value_lr):
        raise ValueError("value_lr must be finite")
    if value_lr <= 0.0 or value_lr > 1.0:
        raise ValueError("value_lr must be in (0, 1]")
    if not isinstance(value_clip, float):
        raise TypeError("value_clip must be a float")
    if not math.isfinite(value_clip):
        raise ValueError("value_clip must be finite")
    if value_clip < 0.0 or value_clip >= 1.0:
        raise ValueError("value_clip must be in [0, 1)")
    if isinstance(epochs, bool) or not isinstance(epochs, int):
        raise TypeError("epochs must be a non-bool int")
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError("batch_size must be a non-bool int")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

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

    for e in range(episodes):
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

        policy_batch = [
            [s, a, old_logp, adv]
            for (s, a, _r, old_logp, _v, _v2, _done), adv in zip(
                trajectory, advantages
            )
        ]
        result = ppo_minibatch(
            h,
            policy_batch,
            batch_size,
            float(alpha),
            c,
            epochs,
            shuffle_seed + e,
        )
        h = result["logits"]
        all_objectives.append(result["objectives"])

        value_batch = [
            [s, value, adv + value]
            for (s, _a, _r, _old_logp, value, _v2, _done), adv in zip(
                trajectory, advantages
            )
        ]
        value_result = ppo_value_update(
            values,
            value_batch,
            float(value_lr),
            value_clip,
            epochs,
        )
        values = value_result["values"]
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
    v_table = [
        [float(r), float(c), values[i]]
        for i, (r, c) in enumerate(states)
    ]
    return {
        "h": h_table,
        "v": v_table,
        "episodes": all_episodes,
        "objectives": all_objectives,
    }


def ppo_train_minibatch_kl(
    env,
    episodes=100,
    alpha=0.05,
    gamma=0.9,
    lambda_=0.95,
    c=0.2,
    epochs=4,
    batch_size=32,
    target_kl=0.01,
    seed=0,
    max_steps=1000,
) -> dict:
    """带 KL 早停的小批量 PPO 训练（每回合 GAE 采样后多轮小批量截断更新），
    返回 h 表、逐回合步表、目标值、KL 与早停标记。

    状态编号、采样、GAE、终止/截断、V 更新及返回的 h、episodes 均逐项
    沿用 ppo_train_minibatch；除 target_kl 外参数校验逐项沿用该接口。
    target_kl 须为非 bool 的 int/float，类型不符抛 TypeError，转 float
    溢出、非有限或 <=0 抛 ValueError，全部校验在采样前完成。第 e 回合
    （0 起）以步序 [s, a, old_logp, A] 为 batch，调用
    ppo_minibatch_kl(H, batch, batch_size, float(alpha), c, epochs,
    target_kl, seed+e)，以其返回 logits 续训；seed+e 只控制该回合洗牌，
    采样随机流不额外消费。

    返回键依次为 h、episodes、objectives、kls、stopped；h 与 episodes
    同 ppo_train_minibatch，objectives、kls、stopped 均与回合数等长，
    第 e 项依次为该回合 ppo_minibatch_kl 返回的 objectives、kls 与
    stopped。新运算结果非有限均抛 ValueError，且不返回部分结果；
    同参同 seed 逐值一致。
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
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError("batch_size must be a non-bool int")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if isinstance(target_kl, bool) or not isinstance(
        target_kl, (int, float)
    ):
        raise TypeError("target_kl must be an int or float")
    try:
        t = float(target_kl)
    except OverflowError:
        raise ValueError("target_kl must convert to a finite float")
    if not math.isfinite(t) or t <= 0.0:
        raise ValueError("target_kl must be finite and > 0")

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
    all_kls = []
    all_stopped = []

    for e in range(episodes):
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
        for t_ in range(len(trajectory) - 1, -1, -1):
            (
                _s,
                _a,
                reward,
                _old_logp,
                value,
                value_next,
                _done,
            ) = trajectory[t_]
            delta = reward + gamma * value_next - value
            if not math.isfinite(delta):
                raise ValueError("GAE delta must be finite")
            a_t = delta + gamma * lambda_ * a_t
            if not math.isfinite(a_t):
                raise ValueError("GAE advantage must be finite")
            advantages[t_] = a_t

        batch = [
            [s, a, old_logp, adv]
            for (s, a, _r, old_logp, _v, _v2, _done), adv in zip(
                trajectory, advantages
            )
        ]
        result = ppo_minibatch_kl(
            h, batch, batch_size, float(alpha), c, epochs, t, seed + e
        )
        h = result["logits"]
        all_objectives.append(result["objectives"])
        all_kls.append(result["kls"])
        all_stopped.append(result["stopped"])

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
        "kls": all_kls,
        "stopped": all_stopped,
    }


def ppo_train_kl(
    env,
    episodes=100,
    alpha=0.05,
    gamma=0.9,
    lambda_=0.95,
    c=0.2,
    epochs=4,
    target_kl=0.01,
    seed=0,
    max_steps=1000,
) -> dict:
    """带 KL 早停的 PPO 训练，返回 h 表、逐回合步表、目标值、KL 与早停标记。

    参数校验、状态编号、H/V 初值、采样、GAE、终止/截断与 V 更新均沿用
    ppo_train。target_kl 为 bool 或非 int/float 抛 TypeError；转 float
    溢出、非有限或 <=0 抛 ValueError，并在采样前完成校验。每回合将
    [s, a, old_logp, A] 批次传给
    ppo_update_kl(H, batch, float(alpha), c, epochs, target_kl)，以其
    返回 logits 续训，不额外消费随机数。

    返回键依次为 h、episodes、objectives、kls、stopped；h 与 episodes
    同 ppo_train，objectives、kls、stopped 均与回合数等长，第 e 项依次
    为该回合 ppo_update_kl 返回的 objectives 浮点列表、kls 浮点列表与
    stopped 布尔值。新运算结果非有限均抛 ValueError，且不返回部分结果；
    同参同 seed 逐值一致。
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
    if isinstance(target_kl, bool) or not isinstance(
        target_kl, (int, float)
    ):
        raise TypeError("target_kl must be an int or float")
    try:
        t = float(target_kl)
    except OverflowError:
        raise ValueError("target_kl must convert to a finite float")
    if not math.isfinite(t) or t <= 0.0:
        raise ValueError("target_kl must be finite and > 0")

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
    all_kls = []
    all_stopped = []

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
        for t_ in range(len(trajectory) - 1, -1, -1):
            (
                _s,
                _a,
                reward,
                _old_logp,
                value,
                value_next,
                _done,
            ) = trajectory[t_]
            delta = reward + gamma * value_next - value
            if not math.isfinite(delta):
                raise ValueError("GAE delta must be finite")
            a_t = delta + gamma * lambda_ * a_t
            if not math.isfinite(a_t):
                raise ValueError("GAE advantage must be finite")
            advantages[t_] = a_t

        batch = [
            [s, a, old_logp, adv]
            for (s, a, _r, old_logp, _v, _v2, _done), adv in zip(
                trajectory, advantages
            )
        ]
        result = ppo_update_kl(h, batch, float(alpha), c, epochs, t)
        h = result["logits"]
        all_objectives.append(result["objectives"])
        all_kls.append(result["kls"])
        all_stopped.append(result["stopped"])

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
        "kls": all_kls,
        "stopped": all_stopped,
    }


def ppo_train_line_search(
    env,
    episodes=100,
    alpha=0.05,
    gamma=0.9,
    lambda_=0.95,
    c=0.2,
    target=0.01,
    steps=8,
    seed=0,
    max_steps=1000,
) -> dict:
    """带几何衰减学习率线搜索的 PPO 训练，返回 h 表、逐回合步表、各回合
    KL、接受标记与采用的学习率。

    env、episodes、alpha、gamma、lambda_、c、seed、max_steps 的校验与
    异常逐项沿用 ppo_train（不设 epochs）；target、steps 的校验与异常
    逐项沿用 ppo_line_search（target 须为非 bool 的 int/float，转 float
    溢出、非有限或 <=0 抛 ValueError；steps 须为非 bool 的 int 且属于
    [1, 61]，越界抛 ValueError）。全部参数校验均在采样前完成。

    训练的状态编号、H/V 初值、采样、轨迹记录、GAE 递推、终止/截断、
    随机消费及 V 更新均逐项沿用 ppo_train；不额外消费随机数。每回合
    以步序 [s, a, old_logp, A] 批次调用
    ppo_line_search(H, batch, float(alpha), c, target, steps)：接受则
    续用其返回 logits，拒绝则 H 不变；无论接受与否，V 仍按原 A 以
    ppo_train 的 V 更新式更新。被调接口异常原样透传，失败时无部分
    结果。

    返回键依次为 h、episodes、kls、accepted、lrs；h 与 episodes 同
    ppo_train；后三项均与回合数等长，依次为该回合 KL 浮点列表、
    accepted 布尔值与采用的浮点学习率（全部拒绝时为 None）。同参同
    seed 逐值一致。
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
    if isinstance(target, bool) or not isinstance(target, (int, float)):
        raise TypeError("target must be an int or float")
    try:
        t = float(target)
    except OverflowError:
        raise ValueError("target must convert to a finite float")
    if not math.isfinite(t) or t <= 0.0:
        raise ValueError("target must be finite and > 0")
    if isinstance(steps, bool) or not isinstance(steps, int):
        raise TypeError("steps must be a non-bool int")
    if steps < 1 or steps > 61:
        raise ValueError("steps must be in [1, 61]")

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
    all_kls = []
    all_accepted = []
    all_lrs = []

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
        for t_ in range(len(trajectory) - 1, -1, -1):
            (
                _s,
                _a,
                reward,
                _old_logp,
                value,
                value_next,
                _done,
            ) = trajectory[t_]
            delta = reward + gamma * value_next - value
            if not math.isfinite(delta):
                raise ValueError("GAE delta must be finite")
            a_t = delta + gamma * lambda_ * a_t
            if not math.isfinite(a_t):
                raise ValueError("GAE advantage must be finite")
            advantages[t_] = a_t

        batch = [
            [s, a, old_logp, adv]
            for (s, a, _r, old_logp, _v, _v2, _done), adv in zip(
                trajectory, advantages
            )
        ]
        result = ppo_line_search(
            h, batch, float(alpha), c, target, steps
        )
        all_kls.append(result["kls"])
        all_accepted.append(result["accepted"])
        all_lrs.append(result["lr"])
        if result["accepted"]:
            h = result["logits"]

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
        "kls": all_kls,
        "accepted": all_accepted,
        "lrs": all_lrs,
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


def ppo_eval_stats(
    env,
    h,
    episodes=100,
    max_steps=1000,
    seed=0,
    window=20,
    success=0.9,
    reward=-20.0,
    patience=3,
) -> dict:
    """按 softmax 策略评估 PPO 风格 logits 表，返回双阈值滑窗收敛统计。

    env、h、episodes、max_steps、seed 的校验与异常、稳定 softmax
    采样、random.Random(seed) 消费顺序、终止/截断边界及 h 不修改均
    沿用 ppo_evaluate（window 同样沿用其校验）。window、patience
    须为非 bool 的正 int，类型不符抛 TypeError，非正抛 ValueError；
    success、reward 须为非 bool 的 int/float，类型不符抛 TypeError，
    转 float 溢出或非有限抛 ValueError，此外 success 须落在 [0, 1]，
    越界抛 ValueError。

    回合项为 [steps, total_reward, success]，与 ppo_evaluate 逐值
    一致。每个完整窗生成 [start, end, success_rate, reward_mean,
    passed]：start、end 为从 1 起的回合序号；success_rate 为窗内
    success 按序从 0.0 累加（成功记 1.0）后除以 window 的 float，
    reward_mean 为窗内 total_reward 按序从 0.0 累加后除以 window 的
    float；passed 仅当 success_rate >= success 且
    reward_mean >= reward 时为 True。episode 为首个由连续 patience
    个 passed 窗构成的序列末窗的 end，不存在则为 None。

    返回键依次为 episodes、windows、success_rate、reward_mean、
    converged、episode；success_rate、reward_mean 为全部回合按同法
    从 0.0 累加后除以回合数的 float，converged 等价于
    episode is not None。相同输入与 seed 逐值一致。
    """
    # 前五参与 window 的校验及回合仿真完全沿用 ppo_evaluate（threshold
    # 不影响 episodes 项）；新参数的校验紧随其后，保持签名参数顺序。
    result = ppo_evaluate(
        env, h, episodes, max_steps, seed, window, 0.0
    )
    episode_records = result["episodes"]

    if isinstance(success, bool) or not isinstance(success, (int, float)):
        raise TypeError("success must be an int or float")
    try:
        success_threshold = float(success)
    except OverflowError:
        raise ValueError("success threshold must be finite")
    if not math.isfinite(success_threshold):
        raise ValueError("success threshold must be finite and in [0, 1]")
    if success_threshold < 0.0 or success_threshold > 1.0:
        raise ValueError("success threshold must be finite and in [0, 1]")
    if isinstance(reward, bool) or not isinstance(reward, (int, float)):
        raise TypeError("reward must be an int or float")
    try:
        reward_threshold = float(reward)
    except OverflowError:
        raise ValueError("reward threshold must be finite")
    if not math.isfinite(reward_threshold):
        raise ValueError("reward threshold must be finite")
    if isinstance(patience, bool) or not isinstance(patience, int):
        raise TypeError("patience must be an int")
    if patience <= 0:
        raise ValueError("patience must be positive")

    windows = []
    for start in range(len(episode_records) - window + 1):
        block = episode_records[start:start + window]
        success_total = 0.0
        reward_total = 0.0
        for item in block:
            success_total += 1.0 if item[2] else 0.0
            reward_total += item[1]
        window_success_rate = success_total / window
        window_reward_mean = reward_total / window
        passed = (
            window_success_rate >= success_threshold
            and window_reward_mean >= reward_threshold
        )
        windows.append(
            [
                start + 1,
                start + window,
                window_success_rate,
                window_reward_mean,
                passed,
            ]
        )

    consecutive = 0
    converged_episode = None
    for entry in windows:
        if entry[4]:
            consecutive += 1
            if consecutive >= patience:
                converged_episode = entry[1]
                break
        else:
            consecutive = 0

    all_success = 0.0
    all_reward = 0.0
    for item in episode_records:
        all_success += 1.0 if item[2] else 0.0
        all_reward += item[1]
    return {
        "episodes": episode_records,
        "windows": windows,
        "success_rate": all_success / episodes,
        "reward_mean": all_reward / episodes,
        "converged": converged_episode is not None,
        "episode": converged_episode,
    }


def ppo_eval_stats_many(
    env,
    h,
    seeds,
    episodes=100,
    max_steps=1000,
    window=20,
    success=0.9,
    reward=-20.0,
    patience=3,
    minimum=0.8,
) -> dict:
    """按多种子运行 ppo_eval_stats，返回跨种子汇总统计。

    seeds 须为非空 list/tuple，元素为非 bool 的 int，允许重复且不修改
    seeds 及其成员；容器或元素类型不符抛 TypeError，空序列抛
    ValueError。minimum 须为非 bool 的 int/float，类型不符抛
    TypeError，转 float 溢出或非有限、或越出 [0, 1] 抛 ValueError。
    其余参数的校验与异常沿用 ppo_eval_stats。

    所有参数校验均在首次 reset/step 前完成（env、h、episodes、
    max_steps、window 由首次调用的 ppo_evaluate 在仿真前校验，h 坐标
    域仅只读访问 env）；随后按 seeds 顺序逐项独立调用 ppo_eval_stats，
    每项仅替换 seed（各自独立的随机流），h 与 seeds 全程不变，env
    无需恢复。

    返回键依次为 results、means、passed、pass_rate、converged：
    results 与 seeds 同序，每项为键序 seed、result 的 dict，result 即
    对应 ppo_eval_stats 的完整返回值。means 键序 success_rate、
    reward_mean，分别为各 result 同名顶层 float 按 results 序从 0.0
    累加后除以项数所得 float。passed 为 result.converged 为真的项数；
    pass_rate 为 passed / len(seeds) 的 float；converged 等价于
    pass_rate >= float(minimum)。相同输入逐值一致。
    """
    if not isinstance(seeds, (list, tuple)):
        raise TypeError("seeds must be a list or tuple")
    if not seeds:
        raise ValueError("seeds must be non-empty")
    for seed in seeds:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("every seed must be a non-bool int")

    if isinstance(minimum, bool) or not isinstance(minimum, (int, float)):
        raise TypeError("minimum must be an int or float")
    try:
        minimum_threshold = float(minimum)
    except OverflowError:
        raise ValueError("minimum must be finite")
    if not math.isfinite(minimum_threshold):
        raise ValueError("minimum must be finite and in [0, 1]")
    if minimum_threshold < 0.0 or minimum_threshold > 1.0:
        raise ValueError("minimum must be finite and in [0, 1]")

    # success/reward/patience 在 ppo_eval_stats 中于仿真之后才校验，
    # 这里提前到首次 reset/step 之前，规则与异常分类完全沿用原接口。
    if isinstance(success, bool) or not isinstance(success, (int, float)):
        raise TypeError("success must be an int or float")
    try:
        success_threshold = float(success)
    except OverflowError:
        raise ValueError("success threshold must be finite")
    if not math.isfinite(success_threshold):
        raise ValueError("success threshold must be finite and in [0, 1]")
    if success_threshold < 0.0 or success_threshold > 1.0:
        raise ValueError("success threshold must be finite and in [0, 1]")
    if isinstance(reward, bool) or not isinstance(reward, (int, float)):
        raise TypeError("reward must be an int or float")
    try:
        reward_threshold = float(reward)
    except OverflowError:
        raise ValueError("reward threshold must be finite")
    if not math.isfinite(reward_threshold):
        raise ValueError("reward threshold must be finite")
    if isinstance(patience, bool) or not isinstance(patience, int):
        raise TypeError("patience must be an int")
    if patience <= 0:
        raise ValueError("patience must be positive")

    results = []
    for seed in seeds:
        result = ppo_eval_stats(
            env,
            h,
            episodes=episodes,
            max_steps=max_steps,
            seed=seed,
            window=window,
            success=success,
            reward=reward,
            patience=patience,
        )
        results.append({"seed": seed, "result": result})

    success_total = 0.0
    reward_total = 0.0
    for item in results:
        success_total += item["result"]["success_rate"]
        reward_total += item["result"]["reward_mean"]
    means = {
        "success_rate": success_total / len(results),
        "reward_mean": reward_total / len(results),
    }

    passed = sum(1 for item in results if item["result"]["converged"])
    pass_rate = passed / len(seeds)
    return {
        "results": results,
        "means": means,
        "passed": passed,
        "pass_rate": pass_rate,
        "converged": pass_rate >= minimum_threshold,
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


def ppo_evaluate_trace_many(
    env, h, seeds, episodes=100, max_steps=1000
) -> dict:
    """按多种子评估 PPO 风格 logits 表，返回轨迹与可复现性分组。

    seeds 须为非空 list/tuple，元素为非 bool 的 int，允许重复且不修改
    seeds；容器或元素类型不符抛 TypeError，空序列抛 ValueError。其余
    参数（env、h、episodes、max_steps）的校验与异常沿用
    ppo_evaluate_trace；全部参数校验在首次 reset/step 之前完成，且不
    修改 h 或 seeds。按 seeds 顺序逐项调用 ppo_evaluate_trace，每项仅
    替换 seed（各自独立的 random.Random(seed) 随机流）。

    返回键依次为 results、groups、reproducible。results 与 seeds 同序，
    每项为键序 seed、result、summary 的 dict：result 即对应
    ppo_evaluate_trace 的完整返回值；summary 为
    [steps, rewards, successes]，三个列表逐回合取 trace 长度、
    total_reward、success。groups 按 seed 首次出现序排列，每项为键序
    seed、indices、identical 的 dict：indices 为该 seed 在 seeds 中的
    零基索引列表；identical 仅当组内各 result 经 ppo_trace_bytes 所得
    规范字节均与首项相同，单项组为 True。reproducible 等价于所有组
    identical。相同输入逐值一致。
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
        result = ppo_evaluate_trace(
            env,
            h,
            episodes=episodes,
            max_steps=max_steps,
            seed=seed,
        )
        steps = []
        rewards = []
        successes = []
        for episode in result["episodes"]:
            steps.append(len(episode["trace"]))
            rewards.append(episode["total_reward"])
            successes.append(episode["success"])
        results.append(
            {
                "seed": seed,
                "result": result,
                "summary": [steps, rewards, successes],
            }
        )

    order = []
    indices_by_seed = {}
    for index, seed in enumerate(seeds):
        if seed not in indices_by_seed:
            indices_by_seed[seed] = []
            order.append(seed)
        indices_by_seed[seed].append(index)

    groups = []
    for seed in order:
        indices = indices_by_seed[seed]
        canonical = ppo_trace_bytes(results[indices[0]]["result"])
        identical = all(
            ppo_trace_bytes(results[index]["result"]) == canonical
            for index in indices[1:]
        )
        groups.append(
            {
                "seed": seed,
                "indices": indices,
                "identical": identical,
            }
        )

    return {
        "results": results,
        "groups": groups,
        "reproducible": all(group["identical"] for group in groups),
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


def _int_to_decimal(value):
    """将任意大小的 int 转为十进制 str，不受 int 串转换位数限制。"""
    if value == 0:
        return "0"
    negative = value < 0
    remaining = -value if negative else value
    chunks = []
    while remaining:
        remaining, chunk = divmod(remaining, 10**9)
        chunks.append(chunk)
    text = str(chunks[-1])
    for chunk in reversed(chunks[:-1]):
        text += str(chunk).rjust(9, "0")
    return "-" + text if negative else text


def _fingerprint_encode(value):
    """将 None/bool/int/float/str/list/dict 递归编码为确定性文本。"""
    if value is None:
        return "N;"
    if isinstance(value, bool):
        return "B1;" if value else "B0;"
    if isinstance(value, int):
        return "I" + _int_to_decimal(value) + ";"
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


def return_plateau(returns, window=20, tol=0.01, min_episodes=40) -> dict:
    """根据逐回合 return 的相邻双窗均值差判定平台期，返回诊断字典。

    returns 须为非空 list/tuple，成员为非 bool 的 int/float 且可转为
    有限 float；window、min_episodes 为非 bool 正 int，tol 为非 bool
    的 int/float 且转为有限 float 后 >= 0。类型错抛 TypeError，空
    returns、整数参数 <= 0、成员或 tol 转 float 溢出/非有限、tol<0
    抛 ValueError。

    取 float 副本 R 与 T=float(tol)，不修改输入。令 w=window、稳定数
    初值 0；对 e=2w..len(R)，前窗 R[e-2w:e-w]、近窗 R[e-w:e] 均从
    0.0 起按序求和后除 w，change 为两均值差的绝对值；change<=T 则
    稳定数加 1，否则归零。episode 取首个 e>=min_episodes 且稳定数
    >=w 的 e，但始终计算至末尾。返回键序为 converged、episode、
    previous、recent、change、stable：previous、recent、change 取末次
    e 的 float，无双窗（len(R)<2w）时均为 None；stable 为末尾稳定数，
    converged 当且仅当 episode 非 None。运算或输出出现非有限值抛
    ValueError。同输入逐值一致，仅用标准库。
    """
    if not isinstance(returns, (list, tuple)):
        raise TypeError("returns must be a list or tuple")
    if isinstance(window, bool) or not isinstance(window, int):
        raise TypeError("window must be an int")
    if isinstance(min_episodes, bool) or not isinstance(min_episodes, int):
        raise TypeError("min_episodes must be an int")
    if isinstance(tol, bool) or not isinstance(tol, (int, float)):
        raise TypeError("tol must be an int or float")
    if not returns:
        raise ValueError("returns must be non-empty")
    if window <= 0:
        raise ValueError("window must be positive")
    if min_episodes <= 0:
        raise ValueError("min_episodes must be positive")
    try:
        tolerance = float(tol)
    except OverflowError:
        raise ValueError("tol must be convertible to a finite float")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tol must be finite and >= 0")

    values = []
    for value in returns:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("every return must be a non-bool int or float")
        try:
            converted = float(value)
        except OverflowError:
            raise ValueError(
                "every return must be convertible to a finite float"
            )
        if not math.isfinite(converted):
            raise ValueError("every return must be a finite number")
        values.append(converted)

    w = window
    stable = 0
    episode = None
    previous = None
    recent = None
    change = None
    for e in range(2 * w, len(values) + 1):
        prev_mean = sum(values[e - 2 * w:e - w], 0.0) / w
        recent_mean = sum(values[e - w:e], 0.0) / w
        difference = abs(prev_mean - recent_mean)
        if not (
            math.isfinite(prev_mean)
            and math.isfinite(recent_mean)
            and math.isfinite(difference)
        ):
            raise ValueError("plateau statistics must be finite")
        previous = prev_mean
        recent = recent_mean
        change = difference
        if difference <= tolerance:
            stable += 1
        else:
            stable = 0
        if episode is None and e >= min_episodes and stable >= w:
            episode = e

    return {
        "converged": episode is not None,
        "episode": episode,
        "previous": previous,
        "recent": recent,
        "change": change,
        "stable": stable,
    }


def _check_plateau_params(window, tol, min_episodes):
    """校验 window、tol、min_episodes，异常与 return_plateau 完全一致。"""
    if isinstance(window, bool) or not isinstance(window, int):
        raise TypeError("window must be an int")
    if isinstance(min_episodes, bool) or not isinstance(min_episodes, int):
        raise TypeError("min_episodes must be an int")
    if isinstance(tol, bool) or not isinstance(tol, (int, float)):
        raise TypeError("tol must be an int or float")
    if window <= 0:
        raise ValueError("window must be positive")
    if min_episodes <= 0:
        raise ValueError("min_episodes must be positive")
    try:
        tolerance = float(tol)
    except OverflowError:
        raise ValueError("tol must be convertible to a finite float")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tol must be finite and >= 0")


def _check_plateau_items(runs):
    """校验 runs 各项结构与 returns 成员，返回 (seed, returns) 列表。

    returns 的校验、异常与 return_plateau 完全一致；不修改输入。
    """
    validated = []
    seen_seeds = set()
    for index, item in enumerate(runs):
        if not isinstance(item, list):
            raise TypeError(f"runs[{index}] must be a list")
        if len(item) != 2:
            raise ValueError(
                f"runs[{index}] must have exactly two elements"
            )
        seed, returns = item
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError(f"runs[{index}] seed must be a non-bool int")
        if seed in seen_seeds:
            raise ValueError(f"runs[{index}] seed must be distinct")
        seen_seeds.add(seed)
        # 完整沿用 return_plateau 的 returns 契约，先全量校验。
        if not isinstance(returns, (list, tuple)):
            raise TypeError("returns must be a list or tuple")
        if not returns:
            raise ValueError("returns must be non-empty")
        for value in returns:
            if isinstance(value, bool) or not isinstance(
                value, (int, float)
            ):
                raise TypeError(
                    "every return must be a non-bool int or float"
                )
            try:
                converted = float(value)
            except OverflowError:
                raise ValueError(
                    "every return must be convertible to a finite float"
                )
            if not math.isfinite(converted):
                raise ValueError("every return must be a finite number")
        validated.append((seed, returns))
    return validated


def _validate_plateau_runs(runs, window, tol, min_episodes):
    """完整校验 runs 契约与三个参数，返回 (seed, returns) 列表。"""
    if not isinstance(runs, list):
        raise TypeError("runs must be a list")
    if not runs:
        raise ValueError("runs must not be empty")
    _check_plateau_params(window, tol, min_episodes)
    return _check_plateau_items(runs)


def return_plateau_many(
    runs, window=20, tol=0.01, min_episodes=40, minimum=1.0
) -> dict:
    """批量判定多种子逐回合 return 的平台期并汇总，返回汇总报告。

    runs 须为非空 list，每项须为恰含两元素的 list [seed, returns]：
    runs 非 list 抛 TypeError，为空抛 ValueError；项非 list 抛
    TypeError，项长度不为二抛 ValueError。seed 须为互异的非 bool
    int，错型抛 TypeError、重复抛 ValueError。各 returns 以及
    window、tol、min_episodes 的校验、异常完全沿用 return_plateau。
    minimum 须为非 bool 的 int/float，错型抛 TypeError，转 float
    溢出、非有限或越出 [0, 1] 抛 ValueError。

    先依 runs 序完整校验全部输入（含四个参数），任一失败都不返回
    部分结果，且不修改输入；全量校验通过后再按 runs 序逐项调用
    return_plateau(returns, window, tol, min_episodes)，每项恰
    调用一次。

    返回键序为 converged、count、rate、groups、failed：groups 与
    runs 同序，每项为键序 seed、report 的 dict，report 为
    return_plateau 的原样结果；count 为 report.converged 为真的项
    数（int）；rate 为 count 除以项数的 float；failed 按 runs 序列
    出 converged 为假的 seed；converged 仅当 rate >= float(minimum)
    时为 True。相同输入逐值一致，仅用标准库，不引入命令行入口。
    """
    if not isinstance(runs, list):
        raise TypeError("runs must be a list")
    if not runs:
        raise ValueError("runs must not be empty")

    _check_plateau_params(window, tol, min_episodes)

    if isinstance(minimum, bool) or not isinstance(minimum, (int, float)):
        raise TypeError("minimum must be an int or float")
    try:
        minimum_rate = float(minimum)
    except OverflowError:
        raise ValueError("minimum must be convertible to a finite float")
    if not math.isfinite(minimum_rate):
        raise ValueError("minimum must be finite")
    if minimum_rate < 0.0 or minimum_rate > 1.0:
        raise ValueError("minimum must be in [0, 1]")

    validated = _check_plateau_items(runs)

    groups = []
    failed = []
    count = 0
    for seed, returns in validated:
        report = return_plateau(
            returns,
            window=window,
            tol=tol,
            min_episodes=min_episodes,
        )
        groups.append({"seed": seed, "report": report})
        if report["converged"]:
            count += 1
        else:
            failed.append(seed)

    rate = count / len(validated)
    return {
        "converged": rate >= minimum_rate,
        "count": count,
        "rate": rate,
        "groups": groups,
        "failed": failed,
    }


def return_plateau_regression(
    base, new, window=20, tol=0.01, min_episodes=40, max_delay=0
) -> dict:
    """比较两组多种子运行的平台期回合，判定是否出现回归，返回报告。

    base 与 new 均须满足 return_plateau_many 的 runs 契约（非空
    list，每项为恰含 [seed, returns] 的 list，seed 为互异的非
    bool int，returns 沿用 return_plateau 契约），两侧 seed 集合
    必须相同、顺序可异，不同抛 ValueError。window、tol、
    min_episodes 的校验、异常完全沿用 return_plateau；max_delay
    须为非 bool 的非负 int，错型抛 TypeError、负值抛 ValueError。

    先完整校验全部输入（含五个参数与 seed 集合一致性），任一失败
    都不返回部分结果，且不修改输入；验毕以
    return_plateau_many(runs, window, tol, min_episodes,
    minimum=1.0) 分别求得 base、new 两报告。

    按 base 的 runs 序生成 rows，每行为 [seed, b, n, d, status]：
    b、n 分别为两报告中该 seed 的 report.episode；仅当 b、n 均为
    int 时 d = n - b，否则 d 为 None。status 判定：n 为 int 且
    （b 为 None 或 n < b）记 "improved"；b、n 均为 int 且
    0 <= d <= max_delay 记 "stable"；d > max_delay 记 "late"；
    仅 b 为 int 记 "lost"；其余记 "pending"。

    返回键序为 passed、base、new、rows、regressed：base、new 为
    两报告的原始结果；regressed 按 base 序列出 status 为 "late"
    或 "lost" 的 seed；passed 当且仅当 regressed 为空时为 True。
    相同输入逐值一致，仅用标准库，不引入命令行入口。
    """
    base_validated = _validate_plateau_runs(base, window, tol, min_episodes)
    new_validated = _validate_plateau_runs(new, window, tol, min_episodes)
    base_seeds = [seed for seed, _ in base_validated]
    if set(base_seeds) != {seed for seed, _ in new_validated}:
        raise ValueError("base and new must have the same seeds")
    if isinstance(max_delay, bool) or not isinstance(max_delay, int):
        raise TypeError("max_delay must be an int")
    if max_delay < 0:
        raise ValueError("max_delay must be non-negative")

    base_report = return_plateau_many(
        base, window=window, tol=tol, min_episodes=min_episodes,
        minimum=1.0,
    )
    new_report = return_plateau_many(
        new, window=window, tol=tol, min_episodes=min_episodes,
        minimum=1.0,
    )

    new_episodes = {
        group["seed"]: group["report"]["episode"]
        for group in new_report["groups"]
    }
    rows = []
    regressed = []
    for group in base_report["groups"]:
        seed = group["seed"]
        b = group["report"]["episode"]
        n = new_episodes[seed]
        both_int = isinstance(b, int) and isinstance(n, int)
        d = n - b if both_int else None
        if isinstance(n, int) and (b is None or n < b):
            status = "improved"
        elif both_int and 0 <= d <= max_delay:
            status = "stable"
        elif both_int and d > max_delay:
            status = "late"
        elif isinstance(b, int):
            status = "lost"
        else:
            status = "pending"
        rows.append([seed, b, n, d, status])
        if status in ("late", "lost"):
            regressed.append(seed)

    return {
        "passed": not regressed,
        "base": base_report,
        "new": new_report,
        "rows": rows,
        "regressed": regressed,
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


def ppo_train_until(
    env,
    max_episodes=100,
    seed=0,
    max_steps=1000,
    window=20,
    threshold=0.9,
    patience=3,
) -> dict:
    """PPO 训练至成功序列收敛或达回合上限，返回 h 表、逐回合步表、目标值与收敛报告。

    env、seed、max_steps 的校验与异常沿用 ppo_train；max_episodes 按
    ppo_train 的 episodes 契约（非 bool 正 int）；window、threshold、
    patience 的校验与异常沿用 ppo_success_streak 同名参数，且在采样前
    全量校验。以 ppo_train 默认训练超参（alpha=0.05、gamma=0.9、
    lambda_=0.95、c=0.2、epochs=4）及给定 seed、max_steps 连续训练，
    全部随机性来自一个 random.Random(seed)。

    每次完成整回合采样和更新后，将该回合步列表末项的 done 加入成功
    序列并调用 ppo_success_streak(successes, window, threshold,
    patience)；首次 converged 为真即停止，否则训练 max_episodes 回合。

    返回键依次为 h、episodes、objectives、convergence；实际完成 k
    回合时，前三项逐值等于 ppo_train(env, episodes=k, seed=seed,
    max_steps=max_steps) 的对应返回，convergence 为成功序列的完整
    ppo_success_streak 结果。不额外消费随机数，同参同 seed 逐值一致。
    """
    if not isinstance(env, GridWorld):
        raise TypeError("env must be a GridWorld")
    if isinstance(max_episodes, bool) or not isinstance(max_episodes, int):
        raise TypeError("max_episodes must be an int")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an int")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int):
        raise TypeError("max_steps must be an int")
    if isinstance(window, bool) or not isinstance(window, int):
        raise TypeError("window must be a non-bool int")
    if isinstance(patience, bool) or not isinstance(patience, int):
        raise TypeError("patience must be a non-bool int")
    if isinstance(threshold, bool) or not isinstance(
        threshold, (int, float)
    ):
        raise TypeError("threshold must be an int or float")
    if max_episodes <= 0:
        raise ValueError("max_episodes must be positive")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if window <= 0:
        raise ValueError("window must be positive")
    if patience <= 0:
        raise ValueError("patience must be positive")
    try:
        float(threshold)
    except OverflowError:
        raise ValueError("threshold must convert to a finite float")
    if not math.isfinite(float(threshold)):
        raise ValueError("threshold must be finite")
    if float(threshold) < 0.0 or float(threshold) > 1.0:
        raise ValueError("threshold must be in [0, 1]")

    alpha = 0.05
    gamma = 0.9
    lambda_ = 0.95
    c = 0.2
    epochs = 4

    actions, states, state_index, h, values, rng = _ppo_training_setup(
        env, seed
    )
    collector = _PpoSuccessCollector(window, threshold, patience)
    h = _ppo_run_episodes(
        env,
        max_episodes,
        actions,
        state_index,
        h,
        values,
        rng,
        max_steps,
        gamma,
        lambda_,
        alpha,
        c,
        epochs,
        collector,
    )

    return {
        "h": _ppo_h_table(h, states),
        "episodes": collector.episodes,
        "objectives": collector.objectives,
        "convergence": collector.convergence,
    }


def ppo_trace_bytes(data) -> bytes:
    """将一份 PPO 轨迹结构严格校验后序列化为单行 JSON 字节。

    data 须为 dict，否则抛 TypeError；其余任何违约均抛 ValueError。
    顶层键序须恰为 episodes、success_rate；episodes 为非空 list，
    每项为键序恰为 trace、total_reward、success 的 dict；trace 为
    非空 list，每步为恰七项的 list
    [r, c, action, next_r, next_c, reward, done]。r、c、next_r、
    next_c、reward 及 total_reward 均须为非 bool 的 int；action 须为
    U、R、D、L 之一；done 与 success 须为 bool；success_rate 须为
    [0, 1] 内的有限 float。任何结构、键序、类型、长度、取值或多余
    字段违约均抛 ValueError；不重算也不改写任何汇总字段，不修改
    输入。

    返回 (json.dumps(data, ensure_ascii=True, allow_nan=False,
    separators=(",", ":")) + "\\n").encode("utf-8")：无额外空白，
    末尾恰一个 LF；相同输入逐字节一致。仅用标准库，不引入命令行
    入口。
    """
    if not isinstance(data, dict):
        raise TypeError("data must be a dict")
    if list(data) != ["episodes", "success_rate"]:
        raise ValueError(
            "data must have exactly the keys episodes, success_rate"
        )

    episodes = data["episodes"]
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("episodes must be a non-empty list")
    for ep_index, episode in enumerate(episodes):
        if not isinstance(episode, dict) or list(episode) != [
            "trace",
            "total_reward",
            "success",
        ]:
            raise ValueError(
                f"episodes[{ep_index}] must have exactly the keys trace,"
                " total_reward, success"
            )
        trace = episode["trace"]
        if not isinstance(trace, list) or not trace:
            raise ValueError(
                f"episodes[{ep_index}] trace must be a non-empty list"
            )
        for step_index, step in enumerate(trace):
            if not isinstance(step, list) or len(step) != 7:
                raise ValueError(
                    f"episodes[{ep_index}] trace[{step_index}] must be a"
                    " list of exactly 7 items"
                )
            r, c, action, next_r, next_c, reward, done = step
            for field in (r, c, next_r, next_c, reward):
                if isinstance(field, bool) or not isinstance(field, int):
                    raise ValueError(
                        f"episodes[{ep_index}] trace[{step_index}]"
                        " numeric fields must be non-bool ints"
                    )
            if not isinstance(action, str) or action not in _ACTIONS:
                raise ValueError(
                    f"episodes[{ep_index}] trace[{step_index}] action"
                    " must be one of U, R, D, L"
                )
            if not isinstance(done, bool):
                raise ValueError(
                    f"episodes[{ep_index}] trace[{step_index}] done"
                    " must be a bool"
                )
        total_reward = episode["total_reward"]
        if isinstance(total_reward, bool) or not isinstance(
            total_reward, int
        ):
            raise ValueError(
                f"episodes[{ep_index}] total_reward must be a non-bool"
                " int"
            )
        if not isinstance(episode["success"], bool):
            raise ValueError(
                f"episodes[{ep_index}] success must be a bool"
            )

    success_rate = data["success_rate"]
    if (
        not isinstance(success_rate, float)
        or not math.isfinite(success_rate)
        or success_rate < 0
        or success_rate > 1
    ):
        raise ValueError(
            "success_rate must be a finite float in [0, 1]"
        )

    return (
        json.dumps(
            data,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


_UTF8_BOM = b"\xef\xbb\xbf"


def ppo_trace_from_bytes(payload) -> dict:
    """将 ppo_trace_bytes 的产物严格解析回 PPO 轨迹结构。

    payload 须恰为 bytes（bytearray、memoryview 等均拒绝），否则
    抛 TypeError；空字节、严格 UTF-8 解码失败、BOM、JSON 语法错误
    或尾随内容、重复对象键、根值非 dict，或任一层违反
    ppo_trace_bytes 输入契约，均抛 ValueError。解析保留对象键序，
    拒绝 NaN、Infinity、-Infinity；不重算也不改写任何汇总字段。

    仅接受规范单行紧凑 ASCII JSON：末尾恰一个 LF，无 BOM 或额外
    空白/换行；数值拼写、整数/浮点类型及 -0.0 均以
    ppo_trace_bytes 重编码后的字节逐字节核对。返回解析得到的独立
    新容器，同一 payload 多次解析逐值一致。仅用标准库，不引入命令
    行入口。
    """
    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    if not payload:
        raise ValueError("payload must not be empty")
    if payload.startswith(_UTF8_BOM):
        raise ValueError("payload must not start with a BOM")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("payload must be valid UTF-8") from exc
    if not text.endswith("\n"):
        raise ValueError("payload must end with exactly one LF")
    body = text[:-1]

    def reject_constant(value):
        raise ValueError(f"invalid JSON constant: {value}")

    def reject_duplicate_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate object key: {key!r}")
            result[key] = value
        return result

    decoder = json.JSONDecoder(
        object_pairs_hook=reject_duplicate_keys,
        parse_constant=reject_constant,
    )
    try:
        data, end = decoder.raw_decode(body)
    except RecursionError as exc:
        raise ValueError("payload nesting too deep") from exc
    except json.JSONDecodeError as exc:
        raise ValueError("payload must be valid JSON") from exc
    if end != len(body):
        raise ValueError("payload must not contain trailing content")
    if not isinstance(data, dict):
        raise ValueError("payload root value must be a dict")

    canonical = ppo_trace_bytes(data)
    if canonical != payload:
        raise ValueError(
            "payload must be canonical ppo_trace_bytes output"
        )
    return data


def ppo_trace_verify(env, data) -> dict:
    """对照 GridWorld 环境逐步验证一份 PPO 轨迹并给出指纹摘要。

    env 须为 GridWorld，否则抛 TypeError。先调用 ppo_trace_bytes(data)
    校验结构并取得规范字节，其异常原样透传；不修改输入。

    随后逐回合验证：每回合首步的 (r, c) 须为起点 S；其余各步的
    (r, c) 须等于上一步的 (next_r, next_c)；每步须满足
    env.transition((r, c), action) == ((next_r, next_c), reward, done)；
    done 为 True 仅允许出现在末步；回合 success 须等于末步 done；
    total_reward 须等于从整数 0 起按步序累加各步 reward 的结果。
    任何违约均抛 ValueError。

    successes 为成功回合数，rate = successes / len(episodes)；rate 与
    data["success_rate"] 的 float.hex() 须相同，否则抛 ValueError。
    digest 为 hashlib.sha256(规范字节).hexdigest()。

    返回键序恰为 episodes、steps、successes、success_rate、
    fingerprint 的 dict，依次为回合数、总步数、成功数（int）、
    rate（float）、digest（小写 64 位十六进制 str）。无随机消费，
    重复调用逐值一致。仅用标准库，不引入命令行入口。
    """
    if not isinstance(env, GridWorld):
        raise TypeError("env must be a GridWorld")
    canonical = ppo_trace_bytes(data)

    start = None
    for r, row in enumerate(env.rows):
        for c, cell in enumerate(row):
            if cell == "S":
                start = (r, c)
                break
        if start is not None:
            break

    episodes = data["episodes"]
    steps = 0
    successes = 0
    for ep_index, episode in enumerate(episodes):
        trace = episode["trace"]
        total = 0
        for step_index, step in enumerate(trace):
            r, c, action, next_r, next_c, reward, done = step
            where = f"episodes[{ep_index}] trace[{step_index}]"
            expected = start if step_index == 0 else prev_next
            if (r, c) != expected:
                raise ValueError(
                    f"{where} (r, c) must be {expected}, got {(r, c)}"
                )
            outcome = env.transition((r, c), action)
            if outcome != ((next_r, next_c), reward, done):
                raise ValueError(
                    f"{where} does not match env.transition: "
                    f"expected {outcome}, got "
                    f"{((next_r, next_c), reward, done)}"
                )
            if done and step_index != len(trace) - 1:
                raise ValueError(
                    f"{where} done may be True only at the last step"
                )
            total += reward
            prev_next = (next_r, next_c)
        if episode["success"] != trace[-1][6]:
            raise ValueError(
                f"episodes[{ep_index}] success must equal the last"
                " step's done"
            )
        if episode["total_reward"] != total:
            raise ValueError(
                f"episodes[{ep_index}] total_reward must equal the sum"
                " of step rewards"
            )
        steps += len(trace)
        if episode["success"]:
            successes += 1

    rate = successes / len(episodes)
    if rate.hex() != data["success_rate"].hex():
        raise ValueError(
            "success_rate must equal successes / len(episodes)"
        )
    digest = hashlib.sha256(canonical).hexdigest()
    return {
        "episodes": len(episodes),
        "steps": steps,
        "successes": successes,
        "success_rate": rate,
        "fingerprint": digest,
    }


def ppo_trace_converge(
    items, window=20, threshold=0.9, tolerance=0.01
) -> dict:
    """对多份 PPO 训练追踪按完整滑窗判定成功收敛，返回汇总字典。

    items 须为至少两项的 list，每项须恰为 bytes（bytearray、
    memoryview 等均拒绝）：非 list 抛 TypeError，成员非 bytes 抛
    TypeError，长度小于 2 抛 ValueError。window 须为非 bool 正 int，
    否则错型抛 TypeError、非正抛 ValueError；threshold、tolerance 须
    恰为 float（bool 与 int 均拒绝）且有限，threshold 须在 [0, 1] 内，
    tolerance 须 >= 0.0，错型抛 TypeError、非有限或越界抛 ValueError。

    各项先通过 ppo_train_trace_verify 的全部校验，异常原样透传；各项
    追踪的回合数须相同，否则抛 ValueError。不修改输入。

    对每个长度为 window 的完整连续滑窗（window 大于回合数时为空），
    按窗终点（1 基回合号）升序生成 [end, rate, change]：end 为窗内末
    回合的 1 基序号 int；rate 从 0.0 起按 items 序、各份内回合序累加
    窗内 success（bool 作为 0/1），再除以 len(items) * window；每份每
    回合目标取其 objectives 末项 float.fromhex(objectives[-1])，change
    为窗内、跨各份相邻回合目标绝对差的最大值（window 为 1 时无相邻
    对，取 0.0）；rate 或 change 非有限均抛 ValueError。

    返回键序恰为 reproducible、episode、windows：reproducible 为各项
    输入字节是否全等；episode 为可复现且窗内 rate >= threshold、
    change <= tolerance 的首个窗 end，否则 None。重复调用逐值一致，
    仅用标准库，不引入命令行入口。
    """
    if not isinstance(items, list):
        raise TypeError("items must be a list")
    for index, item in enumerate(items):
        if not isinstance(item, bytes):
            raise TypeError(f"items[{index}] must be bytes")
    if len(items) < 2:
        raise ValueError("items must contain at least two entries")
    if isinstance(window, bool) or not isinstance(window, int):
        raise TypeError("window must be a non-bool int")
    if window <= 0:
        raise ValueError("window must be positive")
    if not isinstance(threshold, float):
        raise TypeError("threshold must be a float")
    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite")
    if threshold < 0.0 or threshold > 1.0:
        raise ValueError("threshold must be in [0, 1]")
    if not isinstance(tolerance, float):
        raise TypeError("tolerance must be a float")
    if not math.isfinite(tolerance):
        raise ValueError("tolerance must be finite")
    if tolerance < 0.0:
        raise ValueError("tolerance must be >= 0.0")

    traces = []
    episode_count = None
    for index, item in enumerate(items):
        ppo_train_trace_verify(item)
        data = ppo_train_trace_from_bytes(item)
        episodes = data["episodes"]
        if episode_count is None:
            episode_count = len(episodes)
        elif len(episodes) != episode_count:
            raise ValueError(
                f"items[{index}] episodes length must equal the first"
                f" item's ({episode_count})"
            )
        traces.append(episodes)

    reproducible = all(item == items[0] for item in items[1:])

    windows = []
    denominator = len(items) * window
    for end_zero in range(window - 1, episode_count):
        success_sum = 0.0
        change = 0.0
        for episodes in traces:
            for ep_index in range(end_zero - window + 1, end_zero + 1):
                row = episodes[ep_index]
                success_sum += 1.0 if row[2] else 0.0
            for ep_index in range(
                end_zero - window + 2, end_zero + 1
            ):
                current = float.fromhex(
                    episodes[ep_index][4][-1]
                )
                previous = float.fromhex(
                    episodes[ep_index - 1][4][-1]
                )
                difference = abs(current - previous)
                if difference > change:
                    change = difference
        rate = success_sum / denominator
        if not math.isfinite(rate):
            raise ValueError("window rate must be finite")
        if not math.isfinite(change):
            raise ValueError("window change must be finite")
        windows.append([end_zero + 1, rate, change])

    episode = None
    if reproducible:
        for end, rate, change in windows:
            if rate >= threshold and change <= tolerance:
                episode = end
                break

    return {
        "reproducible": reproducible,
        "episode": episode,
        "windows": windows,
    }


def ppo_trace_converge_many(
    runs, window=20, threshold=0.9, tolerance=0.01, minimum=0.8
) -> dict:
    """对多组带 seed 的 PPO 训练追踪批量判定收敛并汇总。

    runs 须为非空 list，每项须为键序恰为 seed、items 的 dict：非
    list 抛 TypeError，为空抛 ValueError；成员非 dict 抛 TypeError，
    键序不符抛 ValueError。seed 须为互异的非 bool int，错型抛
    TypeError、重复抛 ValueError。items 须为至少两项的 bytes 列表：
    非 list 或成员非 bytes 抛 TypeError，长度小于 2 抛 ValueError。
    window、threshold、tolerance 的校验与异常同 ppo_trace_converge。
    minimum 须为非 bool 的 int 或 float，错型抛 TypeError；转 float
    溢出、非有限或越出 [0, 1] 抛 ValueError。

    全量校验通过后按 runs 序逐项调用 ppo_trace_converge，轨迹与回合
    数异常原样透传，不产生部分结果。不修改输入。

    返回键序恰为 converged、rate、earliest、latest、groups、failed
    的 dict：groups 为与 runs 同序的 [seed, report] 列表，report 为
    ppo_trace_converge 的原样结果；failed 为 report 中 episode 为
    None 的 seed 列表（同序）；rate 为 episode 非 None 的组数除以
    总组数的 float；earliest、latest 分别为非 None episode 的最小、
    最大 int，全为 None 时二者均为 None；converged 仅当
    rate >= float(minimum) 时为 True。重复调用逐值一致，仅用标准库，
    不引入命令行入口。
    """
    if not isinstance(runs, list):
        raise TypeError("runs must be a list")
    if not runs:
        raise ValueError("runs must not be empty")
    seen_seeds = set()
    for index, run in enumerate(runs):
        if not isinstance(run, dict):
            raise TypeError(f"runs[{index}] must be a dict")
        if list(run) != ["seed", "items"]:
            raise ValueError(
                f"runs[{index}] keys must be exactly ['seed', 'items']"
                " in order"
            )
        seed = run["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError(
                f"runs[{index}] seed must be a non-bool int"
            )
        if seed in seen_seeds:
            raise ValueError(f"runs[{index}] seed must be unique")
        seen_seeds.add(seed)
        items = run["items"]
        if not isinstance(items, list):
            raise TypeError(f"runs[{index}] items must be a list")
        for item_index, item in enumerate(items):
            if not isinstance(item, bytes):
                raise TypeError(
                    f"runs[{index}] items[{item_index}] must be bytes"
                )
        if len(items) < 2:
            raise ValueError(
                f"runs[{index}] items must contain at least two entries"
            )
    if isinstance(window, bool) or not isinstance(window, int):
        raise TypeError("window must be a non-bool int")
    if window <= 0:
        raise ValueError("window must be positive")
    if not isinstance(threshold, float):
        raise TypeError("threshold must be a float")
    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite")
    if threshold < 0.0 or threshold > 1.0:
        raise ValueError("threshold must be in [0, 1]")
    if not isinstance(tolerance, float):
        raise TypeError("tolerance must be a float")
    if not math.isfinite(tolerance):
        raise ValueError("tolerance must be finite")
    if tolerance < 0.0:
        raise ValueError("tolerance must be >= 0.0")
    if isinstance(minimum, bool) or not isinstance(minimum, (int, float)):
        raise TypeError("minimum must be a non-bool int or float")
    try:
        minimum_rate = float(minimum)
    except OverflowError:
        raise ValueError(
            "minimum must be convertible to float"
        ) from None
    if not math.isfinite(minimum_rate):
        raise ValueError("minimum must be finite")
    if minimum_rate < 0.0 or minimum_rate > 1.0:
        raise ValueError("minimum must be in [0, 1]")

    groups = []
    for run in runs:
        report = ppo_trace_converge(
            run["items"],
            window=window,
            threshold=threshold,
            tolerance=tolerance,
        )
        groups.append([run["seed"], report])

    episodes = [
        report["episode"]
        for _, report in groups
        if report["episode"] is not None
    ]
    failed = [
        seed for seed, report in groups if report["episode"] is None
    ]
    rate = len(episodes) / len(groups)
    earliest = min(episodes) if episodes else None
    latest = max(episodes) if episodes else None

    return {
        "converged": rate >= minimum_rate,
        "rate": rate,
        "earliest": earliest,
        "latest": latest,
        "groups": groups,
        "failed": failed,
    }


def _format_value(value):
    if abs(value) < 0.5e-12:
        value = 0.0
    return format(value, ".12f")


def _reject_constant(value):
    raise ValueError(f"invalid JSON constant: {value}")


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate object key: {key!r}")
        result[key] = value
    return result


_MANIFEST_DECODER = json.JSONDecoder(
    object_pairs_hook=_reject_duplicate_keys,
    parse_constant=_reject_constant,
)


def _load_trace_converge_manifest(path):
    """严格解析 trace-converge 清单，返回可直接传给
    ppo_trace_converge_many 的实参与轨迹文件字节。

    清单须为 UTF-8 JSON 字节：无 BOM、无重复键、无
    NaN/Infinity/-Infinity、无尾随内容；结构与取值违约均抛
    ValueError。相对路径基于清单所在目录按序读取，文件读取失败
    抛 ValueError。
    """
    try:
        with open(path, "rb") as handle:
            payload = handle.read()
    except OSError as exc:
        raise ValueError("cannot read manifest") from exc
    if not payload:
        raise ValueError("manifest must not be empty")
    if payload.startswith(_UTF8_BOM):
        raise ValueError("manifest must not start with a BOM")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("manifest must be valid UTF-8") from exc
    try:
        manifest, end = _MANIFEST_DECODER.raw_decode(text)
    except RecursionError as exc:
        raise ValueError("manifest nesting too deep") from exc
    except json.JSONDecodeError as exc:
        raise ValueError("manifest must be valid JSON") from exc
    if end != len(text):
        raise ValueError("manifest must not contain trailing content")
    if not isinstance(manifest, dict) or list(manifest) != [
        "runs",
        "window",
        "threshold",
        "tolerance",
        "minimum",
    ]:
        raise ValueError(
            "manifest root keys must be exactly"
            " ['runs', 'window', 'threshold', 'tolerance',"
            " 'minimum'] in order"
        )

    runs = manifest["runs"]
    if not isinstance(runs, list) or not runs:
        raise ValueError("runs must be a non-empty list")
    base_dir = os.path.dirname(os.path.abspath(path))
    seen_seeds = set()
    converted = []
    for index, run in enumerate(runs):
        if not isinstance(run, dict) or list(run) != ["seed", "files"]:
            raise ValueError(
                f"runs[{index}] keys must be exactly ['seed', 'files']"
                " in order"
            )
        seed = run["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError(f"runs[{index}] seed must be a non-bool int")
        if seed in seen_seeds:
            raise ValueError(f"runs[{index}] seed must be unique")
        seen_seeds.add(seed)
        files = run["files"]
        if not isinstance(files, list) or len(files) < 2:
            raise ValueError(
                f"runs[{index}] files must contain at least two entries"
            )
        items = []
        for file_index, name in enumerate(files):
            if not isinstance(name, str) or not name:
                raise ValueError(
                    f"runs[{index}] files[{file_index}] must be a"
                    " non-empty str"
                )
            trace_path = os.path.join(base_dir, name)
            try:
                with open(trace_path, "rb") as handle:
                    items.append(handle.read())
            except OSError as exc:
                raise ValueError(
                    f"runs[{index}] files[{file_index}] cannot be read"
                ) from exc
        converted.append({"seed": seed, "items": items})

    window = manifest["window"]
    if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
        raise ValueError("window must be a non-bool positive int")
    threshold = manifest["threshold"]
    if not isinstance(threshold, float) or not math.isfinite(threshold):
        raise ValueError("threshold must be a finite float")
    if threshold < 0.0 or threshold > 1.0:
        raise ValueError("threshold must be in [0, 1]")
    tolerance = manifest["tolerance"]
    if not isinstance(tolerance, float) or not math.isfinite(tolerance):
        raise ValueError("tolerance must be a finite float")
    if tolerance < 0.0:
        raise ValueError("tolerance must be >= 0.0")
    minimum = manifest["minimum"]
    if isinstance(minimum, bool) or not isinstance(minimum, (int, float)):
        raise ValueError("minimum must be a non-bool number")
    try:
        minimum = float(minimum)
    except OverflowError as exc:
        raise ValueError("minimum must be convertible to float") from exc
    if not math.isfinite(minimum) or minimum < 0.0 or minimum > 1.0:
        raise ValueError("minimum must be finite and in [0, 1]")

    return converted, window, threshold, tolerance, minimum


def _trace_converge_compare(base_path, new_path):
    """加载两份 trace-converge 清单并按 BASE 顺序比较汇总结果。

    两清单均完整加载、轨迹全部读取、汇总完成后才生成输出字符串；
    window 须相同，threshold/tolerance/minimum 转 float 后
    float.hex() 须相同，seed 集合须相同（各清单内重复 seed 已由
    清单契约拒绝）。任一违约抛 ValueError。
    """
    (
        base_runs,
        base_window,
        base_threshold,
        base_tolerance,
        base_minimum,
    ) = _load_trace_converge_manifest(base_path)
    (
        new_runs,
        new_window,
        new_threshold,
        new_tolerance,
        new_minimum,
    ) = _load_trace_converge_manifest(new_path)

    if base_window != new_window:
        raise ValueError("window must match between manifests")
    for name, base_value, new_value in (
        ("threshold", base_threshold, new_threshold),
        ("tolerance", base_tolerance, new_tolerance),
        ("minimum", base_minimum, new_minimum),
    ):
        if float.hex(float(base_value)) != float.hex(float(new_value)):
            raise ValueError(f"{name} must match between manifests")

    base_report = ppo_trace_converge_many(
        base_runs,
        window=base_window,
        threshold=base_threshold,
        tolerance=base_tolerance,
        minimum=base_minimum,
    )
    new_report = ppo_trace_converge_many(
        new_runs,
        window=new_window,
        threshold=new_threshold,
        tolerance=new_tolerance,
        minimum=new_minimum,
    )

    new_episodes = {
        seed: group_report["episode"]
        for seed, group_report in new_report["groups"]
    }
    base_seeds = [seed for seed, _ in base_report["groups"]]
    if set(base_seeds) != set(new_episodes):
        raise ValueError("seed sets must match between manifests")

    episodes = []
    for seed, base_group in base_report["groups"]:
        base_episode = base_group["episode"]
        new_episode = new_episodes[seed]
        if base_episode is not None and new_episode is not None:
            delta = new_episode - base_episode
        else:
            delta = None
        episodes.append([seed, base_episode, new_episode, delta])

    result = {
        "base": {
            "converged": base_report["converged"],
            "rate": base_report["rate"],
            "failed": base_report["failed"],
        },
        "new": {
            "converged": new_report["converged"],
            "rate": new_report["rate"],
            "failed": new_report["failed"],
        },
        "rate_delta": new_report["rate"] - base_report["rate"],
        "episodes": episodes,
    }
    return json.dumps(
        result,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
    )


def _trace_converge_history_reports(paths):
    """加载多份 trace-converge 清单并返回各自的汇总报告。

    全部清单完整加载、轨迹全部读取、汇总完成后才返回；window 须相同，
    threshold/tolerance/minimum 转 float 后 float.hex() 须相同，seed
    集合须相同（各清单内重复 seed 已由清单契约拒绝）。任一违约抛
    ValueError。
    """
    loaded = [_load_trace_converge_manifest(path) for path in paths]
    (
        base_runs,
        base_window,
        base_threshold,
        base_tolerance,
        base_minimum,
    ) = loaded[0]
    base_seed_set = {run["seed"] for run in base_runs}
    for runs, window, threshold, tolerance, minimum in loaded[1:]:
        if window != base_window:
            raise ValueError("window must match between manifests")
        for name, base_value, value in (
            ("threshold", base_threshold, threshold),
            ("tolerance", base_tolerance, tolerance),
            ("minimum", base_minimum, minimum),
        ):
            if float.hex(float(base_value)) != float.hex(float(value)):
                raise ValueError(f"{name} must match between manifests")
        if {run["seed"] for run in runs} != base_seed_set:
            raise ValueError("seed sets must match between manifests")

    return [
        ppo_trace_converge_many(
            runs,
            window=window,
            threshold=threshold,
            tolerance=tolerance,
            minimum=minimum,
        )
        for runs, window, threshold, tolerance, minimum in loaded
    ]


def _trace_converge_history(paths):
    """加载多份 trace-converge 清单并按 M1 的 seed 顺序汇总历史。

    全部清单完整加载、轨迹全部读取、汇总完成后才生成输出字符串；
    跨清单的 window、三项浮点参数与 seed 集合规则同
    _trace_converge_history_reports。
    """
    reports = _trace_converge_history_reports(paths)

    experiments = [
        [index, report["converged"], report["rate"], report["failed"]]
        for index, report in enumerate(reports)
    ]
    deltas = [
        [index - 1, index, reports[index]["rate"] - reports[index - 1]["rate"]]
        for index in range(1, len(reports))
    ]

    episodes_by_seed = [
        {seed: group["episode"] for seed, group in report["groups"]}
        for report in reports
    ]
    trends = []
    for seed, _ in reports[0]["groups"]:
        episodes = [per_seed[seed] for per_seed in episodes_by_seed]
        changes = []
        for index in range(1, len(episodes)):
            old_episode = episodes[index - 1]
            new_episode = episodes[index]
            if old_episode is not None and new_episode is not None:
                changes.append(new_episode - old_episode)
            else:
                changes.append(None)
        trends.append([seed, episodes, changes])

    result = {
        "experiments": experiments,
        "deltas": deltas,
        "trends": trends,
    }
    return json.dumps(
        result,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
    )


def _parse_gate_int(text):
    """JSON 整数解析钩子：允许任意位数字面量。

    标准 int() 受 sys.get_int_max_str_digits() 位数上限约束；超过
    上限时逐位累加转换，保证任意位整数字面量（含 -0）均可解析。
    """
    try:
        return int(text)
    except ValueError:
        negative = text.startswith("-")
        digits = text[1:] if negative else text
        value = 0
        for char in digits:
            value = value * 10 + (ord(char) - ord("0"))
        return -value if negative else value


_GATE_PARAM_DECODER = json.JSONDecoder(
    parse_constant=_reject_constant,
    parse_int=_parse_gate_int,
)


def _trace_converge_history_gate(drop_text, delay_text, paths):
    """加载多份 trace-converge 清单并按 DROP/DELAY 门限汇总历史。

    DROP 须为 JSON 非 bool 有限数且在 [0, 1]，DELAY 须为 JSON 非
    bool 非负整数字面量（任意位，-0 合法），否则抛 ValueError；
    不允许尾随内容。全部清单完整
    加载、轨迹全部读取、汇总完成后才生成输出字符串，跨清单的
    window、三项浮点参数与 seed 集合规则同
    _trace_converge_history_reports。

    对每对相邻实验令 d = 新 rate - 旧 rate；逐 seed 比较收敛回合：
    旧为 int 而新为 None 记 lost，均为 int 且新 - 旧 > DELAY 记
    late。该对仅在 d >= -DROP 且无任何 lost/late 时通过。输出键序
    恰为 passed、comparisons、seeds，seed 顺序取首份清单。
    """
    try:
        drop_raw, drop_end = _GATE_PARAM_DECODER.raw_decode(drop_text)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("DROP must be valid JSON") from exc
    if drop_end != len(drop_text):
        raise ValueError("DROP must not contain trailing content")
    if isinstance(drop_raw, bool) or not isinstance(drop_raw, (int, float)):
        raise ValueError("DROP must be a non-bool JSON number")
    try:
        drop = float(drop_raw)
    except OverflowError as exc:
        raise ValueError("DROP must be convertible to float") from exc
    if not math.isfinite(drop) or drop < 0.0 or drop > 1.0:
        raise ValueError("DROP must be finite and in [0, 1]")

    try:
        delay, delay_end = _GATE_PARAM_DECODER.raw_decode(delay_text)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("DELAY must be valid JSON") from exc
    if delay_end != len(delay_text):
        raise ValueError("DELAY must not contain trailing content")
    if isinstance(delay, bool) or not isinstance(delay, int):
        raise ValueError("DELAY must be a non-bool JSON integer")
    if delay < 0:
        raise ValueError("DELAY must be >= 0")

    reports = _trace_converge_history_reports(paths)

    episodes_by_seed = [
        {seed: group["episode"] for seed, group in report["groups"]}
        for report in reports
    ]

    comparisons = []
    pair_passes = []
    pair_failures_by_seed = {
        seed: [] for seed, _ in reports[0]["groups"]
    }
    for index in range(1, len(reports)):
        rate_delta = reports[index]["rate"] - reports[index - 1]["rate"]
        failures = []
        for seed, _ in reports[0]["groups"]:
            old_episode = episodes_by_seed[index - 1][seed]
            new_episode = episodes_by_seed[index][seed]
            if isinstance(old_episode, int) and new_episode is None:
                kind = "lost"
            elif (
                isinstance(old_episode, int)
                and isinstance(new_episode, int)
                and new_episode - old_episode > delay
            ):
                kind = "late"
            else:
                continue
            failures.append([seed, kind])
            pair_failures_by_seed[seed].append([index - 1, index, kind])
        pair_pass = rate_delta >= -drop and not failures
        comparisons.append([index - 1, index, rate_delta, pair_pass])
        pair_passes.append(pair_pass)

    seeds = []
    for seed, _ in reports[0]["groups"]:
        episodes = [per_seed[seed] for per_seed in episodes_by_seed]
        seeds.append([seed, episodes, pair_failures_by_seed[seed]])

    result = {
        "passed": all(pair_passes),
        "comparisons": comparisons,
        "seeds": seeds,
    }
    return json.dumps(
        result,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
    )


def _validate_gate_result(result, name, drop, delay):
    """校验一份 trace-converge-history-gate 成功输出。

    键序须恰为 passed、comparisons、seeds；错型抛 TypeError，其余
    违约抛 ValueError。failures 须与 episodes 及 delay 隐含的
    lost/late 事件一致，通过标记与 passed 须满足门限等式。返回
    (comparisons, seeds)，不修改输入。
    """
    if not isinstance(result, dict):
        raise TypeError(f"{name} must be a dict")
    if list(result) != ["passed", "comparisons", "seeds"]:
        raise ValueError(
            f"{name} must have exactly the keys passed, comparisons,"
            " seeds in order"
        )
    passed = result["passed"]
    if not isinstance(passed, bool):
        raise TypeError(f"{name} passed must be a bool")

    comparisons = result["comparisons"]
    if not isinstance(comparisons, list):
        raise TypeError(f"{name} comparisons must be a list")
    if not comparisons:
        raise ValueError(f"{name} comparisons must be non-empty")
    for index, entry in enumerate(comparisons):
        entry_name = f"{name} comparisons[{index}]"
        if not isinstance(entry, list):
            raise TypeError(f"{entry_name} must be a list")
        if len(entry) != 4:
            raise ValueError(
                f"{entry_name} must contain exactly four fields"
            )
        left, right, delta, pair_pass = entry
        if isinstance(left, bool) or not isinstance(left, int):
            raise TypeError(
                f"{entry_name} left index must be a non-bool int"
            )
        if isinstance(right, bool) or not isinstance(right, int):
            raise TypeError(
                f"{entry_name} right index must be a non-bool int"
            )
        if left != index or right != index + 1:
            raise ValueError(
                f"{entry_name} indices must be [{index}, {index + 1}]"
            )
        if not isinstance(delta, float):
            raise TypeError(f"{entry_name} rate delta must be a float")
        if not math.isfinite(delta) or delta < -1.0 or delta > 1.0:
            raise ValueError(
                f"{entry_name} rate delta must be finite and in [-1, 1]"
            )
        if not isinstance(pair_pass, bool):
            raise TypeError(f"{entry_name} pair pass flag must be a bool")

    seeds = result["seeds"]
    if not isinstance(seeds, list):
        raise TypeError(f"{name} seeds must be a list")
    if not seeds:
        raise ValueError(f"{name} seeds must be non-empty")
    episode_count = len(comparisons) + 1
    seen_seeds = set()
    failed_pairs = set()
    for seed_index, entry in enumerate(seeds):
        entry_name = f"{name} seeds[{seed_index}]"
        if not isinstance(entry, list):
            raise TypeError(f"{entry_name} must be a list")
        if len(entry) != 3:
            raise ValueError(
                f"{entry_name} must contain exactly three fields"
            )
        seed, episodes, failures = entry
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError(f"{entry_name} seed must be a non-bool int")
        if seed in seen_seeds:
            raise ValueError(f"{entry_name} seed must be unique")
        seen_seeds.add(seed)
        if not isinstance(episodes, list):
            raise TypeError(f"{entry_name} episodes must be a list")
        if len(episodes) != episode_count:
            raise ValueError(
                f"{entry_name} episodes must contain {episode_count}"
                " entries"
            )
        for episode_index, episode in enumerate(episodes):
            if episode is None:
                continue
            if isinstance(episode, bool) or not isinstance(episode, int):
                raise TypeError(
                    f"{entry_name} episodes[{episode_index}] must be"
                    " None or a non-bool int"
                )
            if episode <= 0:
                raise ValueError(
                    f"{entry_name} episodes[{episode_index}] must be"
                    " positive"
                )
        if not isinstance(failures, list):
            raise TypeError(f"{entry_name} failures must be a list")
        for failure_index, failure in enumerate(failures):
            failure_name = f"{entry_name} failures[{failure_index}]"
            if not isinstance(failure, list):
                raise TypeError(f"{failure_name} must be a list")
            if len(failure) != 3:
                raise ValueError(
                    f"{failure_name} must contain exactly three fields"
                )
            left, right, kind = failure
            if isinstance(left, bool) or not isinstance(left, int):
                raise TypeError(
                    f"{failure_name} left index must be a non-bool int"
                )
            if isinstance(right, bool) or not isinstance(right, int):
                raise TypeError(
                    f"{failure_name} right index must be a non-bool int"
                )
            if not isinstance(kind, str):
                raise TypeError(f"{failure_name} kind must be a str")
        expected_failures = []
        for index in range(len(comparisons)):
            old_episode = episodes[index]
            new_episode = episodes[index + 1]
            if old_episode is not None and new_episode is None:
                expected_failures.append([index, index + 1, "lost"])
            elif (
                old_episode is not None
                and new_episode is not None
                and new_episode - old_episode > delay
            ):
                expected_failures.append([index, index + 1, "late"])
        if failures != expected_failures:
            raise ValueError(
                f"{entry_name} failures must equal the lost/late events"
                " implied by episodes and delay"
            )
        for left, _, _ in expected_failures:
            failed_pairs.add(left)

    pair_passes = []
    for index, entry in enumerate(comparisons):
        expected_pass = entry[2] >= -drop and index not in failed_pairs
        if entry[3] != expected_pass:
            raise ValueError(
                f"{name} comparisons[{index}] pair pass flag must equal"
                " (rate delta >= -drop and no lost/late)"
            )
        pair_passes.append(entry[3])
    if passed != all(pair_passes):
        raise ValueError(
            f"{name} passed must equal the conjunction of pair pass"
            " flags"
        )
    return comparisons, seeds


def _gate_deep_identical(left, right):
    """按类型、float.hex 与顺序判定两个 JSON-like 值全同。"""
    if type(left) is not type(right):
        return False
    if isinstance(left, float):
        return float.hex(left) == float.hex(right)
    if isinstance(left, dict):
        if list(left) != list(right):
            return False
        return all(
            _gate_deep_identical(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        if len(left) != len(right):
            return False
        return all(
            _gate_deep_identical(item_left, item_right)
            for item_left, item_right in zip(left, right)
        )
    return left == right


def ppo_trace_gate_consensus(groups) -> dict:
    """汇总多组 trace-converge-history-gate 成功结果的一致性。

    groups 须为至少两项的 list；每项须为键序恰为 drop、delay、
    result 的 dict：drop 须为 [0, 1] 有限 float，delay 须为非
    bool 非负 int，result 须符合 trace-converge-history-gate 成功
    输出契约（键序 passed、comparisons、seeds；comparisons 非空，
    逐项为 [左索引, 右索引, rate 差, 通过标记]，索引须从 0 起连续，
    rate 差须为 [-1, 1] 有限 float；seeds 非空且 seed 唯一，逐项为
    [seed, episodes, failures]，episodes 长度须为比较数加一且每项
    为 None 或非 bool 正 int，failures 须与 episodes 及 delay 隐含
    的 lost/late 事件完全一致；通过标记须等于 rate 差 >= -drop 且
    该对无 lost/late，passed 须等于各对通过标记的合取）。错型抛
    TypeError，其余违约抛 ValueError；不修改输入。

    全部校验通过后，各组 drop 的 float.hex()、delay、比较索引与
    seed 顺序还须相同，否则抛 ValueError；失败不产生部分结果。

    返回键序 fingerprint、consensus、passed、seeds：fingerprint 为
    完整 groups（保持列表及公开 dict 键序）经 _fingerprint_encode
    编码所得 UTF-8 字节的 SHA-256 小写 64 位 hex；consensus 仅当各
    result 按类型、float.hex 与顺序全同；passed 仅当各 result 的
    passed 全真；seeds 按首组 seed 顺序列 [seed, lost_count,
    late_count]，计数为各组该 seed 的 lost/late 失败数之和。结果
    确定，仅用标准库，不新增命令行入口。
    """
    if not isinstance(groups, list):
        raise TypeError("groups must be a list")
    if len(groups) < 2:
        raise ValueError("groups must contain at least two entries")

    validated = []
    for index, group in enumerate(groups):
        name = f"groups[{index}]"
        if not isinstance(group, dict):
            raise TypeError(f"{name} must be a dict")
        if list(group) != ["drop", "delay", "result"]:
            raise ValueError(
                f"{name} keys must be exactly ['drop', 'delay',"
                " 'result'] in order"
            )
        drop = group["drop"]
        if not isinstance(drop, float):
            raise TypeError(f"{name} drop must be a float")
        if not math.isfinite(drop) or drop < 0.0 or drop > 1.0:
            raise ValueError(f"{name} drop must be finite and in [0, 1]")
        delay = group["delay"]
        if isinstance(delay, bool) or not isinstance(delay, int):
            raise TypeError(f"{name} delay must be a non-bool int")
        if delay < 0:
            raise ValueError(f"{name} delay must be >= 0")
        comparisons, seeds = _validate_gate_result(
            group["result"], f"{name} result", drop, delay
        )
        validated.append((drop, delay, comparisons, seeds, group["result"]))

    first_drop_hex = float.hex(validated[0][0])
    first_delay = validated[0][1]
    first_indices = [[entry[0], entry[1]] for entry in validated[0][2]]
    first_seed_order = [entry[0] for entry in validated[0][3]]
    for drop, delay, comparisons, seeds, _ in validated[1:]:
        if float.hex(drop) != first_drop_hex:
            raise ValueError("drop must match between groups")
        if delay != first_delay:
            raise ValueError("delay must match between groups")
        if [[entry[0], entry[1]] for entry in comparisons] != first_indices:
            raise ValueError(
                "comparison indices must match between groups"
            )
        if [entry[0] for entry in seeds] != first_seed_order:
            raise ValueError("seed order must match between groups")

    results = [entry[4] for entry in validated]
    consensus = all(
        _gate_deep_identical(results[0], result) for result in results[1:]
    )
    passed = all(result["passed"] for result in results)
    seeds_out = []
    for position, seed in enumerate(first_seed_order):
        lost_count = 0
        late_count = 0
        for _, _, _, seeds, _ in validated:
            for _, _, kind in seeds[position][2]:
                if kind == "lost":
                    lost_count += 1
                else:
                    late_count += 1
        seeds_out.append([seed, lost_count, late_count])

    return {
        "fingerprint": _fingerprint_hex(groups),
        "consensus": consensus,
        "passed": passed,
        "seeds": seeds_out,
    }


def _validate_consensus_summary(data, name):
    """校验一份 ppo_trace_gate_consensus 摘要。

    键序须恰为 fingerprint、consensus、passed、seeds；fingerprint
    须为 64 位小写十六进制 str，consensus、passed 须为 bool，
    seeds 须为非空 list，逐项恰为 [seed, lost, late] 且三项均为
    非 bool int，seed 唯一、计数非负。错型抛 TypeError，其余违约
    抛 ValueError；不修改输入。返回
    (fingerprint, consensus, passed, [(seed, lost, late), ...])。
    """
    if not isinstance(data, dict):
        raise TypeError(f"{name} must be a dict")
    if list(data) != ["fingerprint", "consensus", "passed", "seeds"]:
        raise ValueError(
            f"{name} keys must be exactly ['fingerprint', 'consensus',"
            " 'passed', 'seeds'] in order"
        )
    fingerprint = data["fingerprint"]
    if not isinstance(fingerprint, str):
        raise TypeError(f"{name} fingerprint must be a str")
    if len(fingerprint) != 64 or any(
        char not in _HEX_DIGITS for char in fingerprint
    ):
        raise ValueError(
            f"{name} fingerprint must be 64 lowercase hex characters"
        )
    consensus = data["consensus"]
    if not isinstance(consensus, bool):
        raise TypeError(f"{name} consensus must be a bool")
    passed = data["passed"]
    if not isinstance(passed, bool):
        raise TypeError(f"{name} passed must be a bool")
    seeds = data["seeds"]
    if not isinstance(seeds, list):
        raise TypeError(f"{name} seeds must be a list")
    if not seeds:
        raise ValueError(f"{name} seeds must be non-empty")
    seen_seeds = set()
    rows = []
    for index, row in enumerate(seeds):
        row_name = f"{name} seeds[{index}]"
        if not isinstance(row, list):
            raise TypeError(f"{row_name} must be a list")
        if len(row) != 3:
            raise ValueError(
                f"{row_name} must contain exactly three fields"
            )
        seed, lost, late = row
        for field, field_name in (
            (seed, "seed"),
            (lost, "lost"),
            (late, "late"),
        ):
            if isinstance(field, bool) or not isinstance(field, int):
                raise TypeError(
                    f"{row_name} {field_name} must be a non-bool int"
                )
        if seed in seen_seeds:
            raise ValueError(f"{row_name} seed must be unique")
        seen_seeds.add(seed)
        if lost < 0 or late < 0:
            raise ValueError(f"{row_name} counts must be >= 0")
        rows.append((seed, lost, late))
    return fingerprint, consensus, passed, rows


def ppo_trace_gate_consensus_compare(left, right) -> dict:
    """比较两份 ppo_trace_gate_consensus 摘要的一致性。

    left、right 均须为键序恰为 fingerprint、consensus、passed、
    seeds 的 dict：fingerprint 须为 64 位小写十六进制 str，
    consensus、passed 须为 bool，seeds 须为非空 list，逐项恰为
    [seed, lost, late] 且三项均为非 bool int，seed 唯一、计数
    非负。参数非 dict 或字段错型抛 TypeError；键序、摘要格式、
    空表、行长、重复 seed、负计数或两侧 seed 顺序不同抛
    ValueError。先全量校验，失败不产生部分结果；不修改输入。

    返回键序 identical、fingerprint_equal、flags、seeds、
    differences：fingerprint_equal 为两侧摘要是否相同；flags 为
    [consensus 是否相同, passed 是否相同]；seeds 按原 seed 顺序列
    [seed, 左 lost, 左 late, 右 lost, 右 late, 计数是否相同]；
    differences 按原序列出计数不同的 seed；identical 仅当摘要、
    标志与计数全同。结果确定，仅用标准库，不新增命令行入口。
    """
    left_fp, left_consensus, left_passed, left_rows = (
        _validate_consensus_summary(left, "left")
    )
    right_fp, right_consensus, right_passed, right_rows = (
        _validate_consensus_summary(right, "right")
    )
    if [row[0] for row in left_rows] != [row[0] for row in right_rows]:
        raise ValueError("seed order must match between left and right")

    fingerprint_equal = left_fp == right_fp
    flags = [
        left_consensus == right_consensus,
        left_passed == right_passed,
    ]
    seeds_out = []
    differences = []
    for left_row, right_row in zip(left_rows, right_rows):
        seed, left_lost, left_late = left_row
        _, right_lost, right_late = right_row
        equal = left_lost == right_lost and left_late == right_late
        seeds_out.append(
            [seed, left_lost, left_late, right_lost, right_late, equal]
        )
        if not equal:
            differences.append(seed)
    identical = fingerprint_equal and all(flags) and not differences
    return {
        "identical": identical,
        "fingerprint_equal": fingerprint_equal,
        "flags": flags,
        "seeds": seeds_out,
        "differences": differences,
    }


def ppo_trace_gate_consensus_history(items) -> dict:
    """比较多份 ppo_trace_gate_consensus 摘要的一致性历史。

    items 须为至少两项的 list；每项须符合
    ppo_trace_gate_consensus_compare 单侧完整契约（键序恰为
    fingerprint、consensus、passed、seeds 的 dict：fingerprint
    须为 64 位小写十六进制 str，consensus、passed 须为 bool，
    seeds 须为非空 list，逐项恰为 [seed, lost, late] 且三项均为
    非 bool int，seed 唯一、计数非负）。items 非 list 或项、字段
    错型抛 TypeError；少于两项、键序、摘要格式、空 seed 表、行
    长、重复 seed、负计数或各项 seed 顺序不同抛 ValueError。先全
    量校验，失败不产生部分结果；不修改输入。

    以第 0 项为基准，按 items 顺序（含自身）调用
    ppo_trace_gate_consensus_compare。返回键序 identical、
    comparisons、mismatches、seeds：comparisons 为各次完整报
    告；mismatches 为报告 identical 为 False 的非零索引升序
    list；seeds 按首项 seed 顺序，每行为 [seed, losts, lates,
    changes]，losts、lates 为各项对应非负计数列表，changes 为
    相邻项的 [lost 差, late 差] 列表，按后项减前项且均为 int；
    顶层 identical 当且仅当 mismatches 为空。结果确定，重复调用
    及深拷贝逐值一致，仅用标准库，不新增命令行入口。
    """
    if not isinstance(items, list):
        raise TypeError("items must be a list")
    if len(items) < 2:
        raise ValueError("items must contain at least two entries")

    validated = []
    for index, item in enumerate(items):
        validated.append(
            _validate_consensus_summary(item, f"items[{index}]")[3]
        )
    first_seed_order = [row[0] for row in validated[0]]
    for rows in validated[1:]:
        if [row[0] for row in rows] != first_seed_order:
            raise ValueError("seed order must match between items")

    baseline = items[0]
    comparisons = [
        ppo_trace_gate_consensus_compare(baseline, item) for item in items
    ]
    mismatches = [
        index
        for index, report in enumerate(comparisons)
        if index != 0 and not report["identical"]
    ]
    seeds_out = []
    for position, seed in enumerate(first_seed_order):
        losts = [rows[position][1] for rows in validated]
        lates = [rows[position][2] for rows in validated]
        changes = [
            [losts[next_index] - losts[next_index - 1],
             lates[next_index] - lates[next_index - 1]]
            for next_index in range(1, len(items))
        ]
        seeds_out.append([seed, losts, lates, changes])
    return {
        "identical": not mismatches,
        "comparisons": comparisons,
        "mismatches": mismatches,
        "seeds": seeds_out,
    }


def _write_line(text):
    sys.stdout.buffer.write(text.encode("utf-8") + b"\n")


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
        _write_line(json.dumps(out, separators=(",", ":")))
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
        _write_line(json.dumps(out, separators=(",", ":")))
        return 0
    if command == "trace-converge" and len(argv) == 3:
        try:
            (
                runs,
                window,
                threshold,
                tolerance,
                minimum,
            ) = _load_trace_converge_manifest(argv[2])
            report = ppo_trace_converge_many(
                runs,
                window=window,
                threshold=threshold,
                tolerance=tolerance,
                minimum=minimum,
            )
            out = json.dumps(
                report,
                ensure_ascii=True,
                allow_nan=False,
                separators=(",", ":"),
            )
        except (TypeError, ValueError):
            return 2
        _write_line(out)
        return 0
    if command == "trace-converge-compare" and len(argv) == 4:
        try:
            out = _trace_converge_compare(argv[2], argv[3])
        except (TypeError, ValueError):
            return 2
        _write_line(out)
        return 0
    if command == "trace-converge-history" and len(argv) >= 4:
        try:
            out = _trace_converge_history(argv[2:])
        except (TypeError, ValueError):
            return 2
        _write_line(out)
        return 0
    if command == "trace-converge-history-gate" and len(argv) >= 6:
        try:
            out = _trace_converge_history_gate(
                argv[2], argv[3], argv[4:]
            )
        except (TypeError, ValueError):
            return 2
        _write_line(out)
        return 0
    return 2


if __name__ == "__main__":
    try:
        code = _run(sys.argv)
    except Exception:
        code = 2
    if code != 0:
        sys.stderr.buffer.write(b"error\n")
    sys.exit(code)
