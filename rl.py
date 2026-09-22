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
    dones = []
    returns = []
    windows = []
    streak = 0
    converged_episode = None

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
            next_state, reward_, done = env.step(action)
            value_next = 0.0 if done else values[state_index[next_state]]
            trajectory.append(
                (s, a, reward_, old_logp, value, value_next, done)
            )
            step_records.append(
                [state[0], state[1], action, reward_, done]
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
                reward_,
                _old_logp,
                value,
                value_next,
                _done,
            ) = trajectory[t]
            delta = reward_ + gamma * value_next - value
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

        dones.append(trajectory[-1][6])
        episode_return = 0.0
        for (_s, _a, reward_, _ol, _v, _v2, _d) in trajectory:
            episode_return += reward_
        returns.append(episode_return)

        end = len(returns)
        if end >= window:
            start = end - window + 1
            hits = 0
            for flag in dones[start - 1:end]:
                if flag:
                    hits += 1
            success_rate = hits / window
            window_return = 0.0
            for past_return in returns[start - 1:end]:
                window_return += past_return
            reward_mean = window_return / window
            passed = success_rate >= success and reward_mean >= reward
            windows.append(
                [start, end, success_rate, reward_mean, passed]
            )
            if passed:
                streak += 1
            else:
                streak = 0
            if end >= min_ep and streak >= patience:
                converged_episode = end
                break

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
        "report": {
            "converged": converged_episode is not None,
            "episode": converged_episode,
            "windows": windows,
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
    successes = []
    convergence = None

    for _ in range(max_episodes):
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

        successes.append(step_records[-1][4])
        convergence = ppo_success_streak(
            successes, window, threshold, patience
        )
        if convergence["converged"]:
            break

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
        "convergence": convergence,
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
