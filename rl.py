"""仅依赖标准库的 GridWorld 强化学习环境。

用法：
    python rl.py run MAP ACTIONS

MAP 以 "/" 分行，ACTIONS 为非空的 URDL 字符串（U=上 R=右 D=下 L=左）。
"""

import json
import sys

# 动作 -> (行位移, 列位移)，顺序即 U R D L
_MOVES = {
    "U": (-1, 0),
    "R": (0, 1),
    "D": (1, 0),
    "L": (0, -1),
}
_ACTIONS = "URDL"
_ALLOWED_CHARS = frozenset("S.G#")


class GridWorld:
    """确定性网格世界。

    rows 为非空等宽字符串元组，字符仅限 S/. /G/#：恰有一个 S（起点），
    至少一个 G（终点）。非法类型抛 TypeError，其余构造错误抛 ValueError。
    """

    def __init__(self, rows):
        if not isinstance(rows, tuple):
            raise TypeError("rows must be a tuple")
        for row in rows:
            if not isinstance(row, str):
                raise TypeError("every row must be a str")

        if len(rows) == 0:
            raise ValueError("rows must be non-empty")
        width = len(rows[0])
        if width == 0:
            raise ValueError("rows must not contain empty strings")
        if any(len(row) != width for row in rows):
            raise ValueError("rows must be equal width")
        if any(any(ch not in _ALLOWED_CHARS for ch in row) for row in rows):
            raise ValueError("rows may only contain S, ., G, #")

        starts = [
            (r, c)
            for r, row in enumerate(rows)
            for c, ch in enumerate(row)
            if ch == "S"
        ]
        if len(starts) != 1:
            raise ValueError("rows must contain exactly one S")
        if not any("G" in row for row in rows):
            raise ValueError("rows must contain at least one G")

        self.rows = rows
        self._start = starts[0]
        self.state = self._start
        self.done = False

    def reset(self):
        """回到起点并清除终止标记，返回复原后的 state。"""
        self.state = self._start
        self.done = False
        return self.state

    @staticmethod
    def _check_action(action):
        if not isinstance(action, str):
            raise TypeError("action must be a str")
        if action not in _MOVES:
            raise ValueError("action must be one of U, R, D, L")

    def _advance(self, state, action):
        """按规则计算目标位置，越界或撞墙 (#) 则原地不动。"""
        r, c = state
        dr, dc = _MOVES[action]
        nr, nc = r + dr, c + dc
        if not (0 <= nr < len(self.rows) and 0 <= nc < len(self.rows[0])):
            return r, c
        if self.rows[nr][nc] == "#":
            return r, c
        return nr, nc

    def step(self, action):
        """执行动作并写回 state/done，返回 (state, reward, done)。

        到达 G 得 0 且终止；其余情况（含越界、撞墙后原地）得 -1。
        已终止再 step 抛 RuntimeError。
        """
        self._check_action(action)
        if self.done:
            raise RuntimeError("episode is done")

        next_state = self._advance(self.state, action)
        self.state = next_state
        if self.rows[next_state[0]][next_state[1]] == "G":
            self.done = True
            return next_state, 0, True
        return next_state, -1, False

    def transition(self, state, action):
        """纯查询版 step：不读写实例的 state/done。

        先校验 state（类型先于越界/#），再校验 action。
        给定 state 位于 G 时返回 (state, 0, True)。
        """
        if (
            not isinstance(state, tuple)
            or len(state) != 2
            or not all(isinstance(v, int) and not isinstance(v, bool) for v in state)
        ):
            raise TypeError("state must be a tuple of two ints")

        r, c = state
        if not (0 <= r < len(self.rows) and 0 <= c < len(self.rows[0])):
            raise ValueError("state is out of bounds")
        if self.rows[r][c] == "#":
            raise ValueError("state must not be on a wall")

        self._check_action(action)

        if self.rows[r][c] == "G":
            return state, 0, True

        next_state = self._advance((r, c), action)
        if self.rows[next_state[0]][next_state[1]] == "G":
            return next_state, 0, True
        return next_state, -1, False


def _run(map_text, actions):
    world = GridWorld(tuple(map_text.split("/")))
    if not isinstance(actions, str) or len(actions) == 0:
        raise ValueError("actions must be a non-empty str")
    if any(ch not in _ACTIONS for ch in actions):
        raise ValueError("actions may only contain U, R, D, L")

    trace = []
    for action in actions:
        state, reward, done = world.step(action)
        trace.append([state[0], state[1], reward, done])
    return trace


def main(argv):
    if len(argv) != 4 or argv[1] != "run":
        sys.stderr.write("error\n")
        return 2
    try:
        trace = _run(argv[2], argv[3])
    except Exception:
        sys.stderr.write("error\n")
        return 2

    sys.stdout.write(json.dumps(trace, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
