# -*- coding: utf-8 -*-
"""纯函数展示工具：颜色、格式化、进度条、轨迹图。"""
import os
import shutil


class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    GRAY = "\033[90m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    CYAN = "\033[36m"
    MAGENTA = "\033[35m"


def fmt_time(seconds):
    try:
        s = max(0, int(round(float(seconds))))
    except (TypeError, ValueError):
        return "--"
    h, rem = divmod(s, 3600)
    m, s = rem // 60, rem % 60
    if h > 0:
        return "%dh%dm" % (h, m)
    if m > 0:
        return "%dm%ds" % (m, s)
    return "%ds" % s


def fmt_num(v, width=8, prec=4):
    if v is None:
        txt = "--"
    elif isinstance(v, int) or float(v).is_integer():
        txt = str(int(v))
    else:
        txt = ("%." + str(prec) + "f") % float(v)
    return txt.rjust(width)


def term_width():
    try:
        w = shutil.get_terminal_size().columns
    except Exception:
        w = 0
    if not w:
        try:
            w = int(os.environ.get("COLUMNS", 0))
        except (TypeError, ValueError):
            w = 0
    return w if w >= 40 else 80


def bar(ratio, width=20):
    try:
        r = min(1.0, max(0.0, float(ratio)))
    except (TypeError, ValueError):
        r = 0.0
    width = max(1, int(width))
    filled = int(round(r * width))
    color = color_for_ratio(r)
    return "%s[%s%s]%s" % (color, "█" * filled, "░" * (width - filled), C.RESET)


_SPARK = "▁▂▃▄▅▆▇█"


def sparkline(values, width=40):
    vals = [v for v in (values or []) if v is not None and v == v]
    width = max(1, int(width))
    if not vals:
        return " " * width
    if len(vals) > width:
        out = []
        n = len(vals)
        for i in range(width):
            a = i * n // width
            b = max(a + 1, (i + 1) * n // width)
            chunk = vals[a:b]
            out.append(sum(chunk) / len(chunk))
        vals = out
    lo, hi = min(vals), max(vals)
    span = hi - lo
    chars = []
    for v in vals:
        if span <= 1e-12:
            idx = 3
        else:
            idx = int((v - lo) / span * (len(_SPARK) - 1) + 1e-9)
            idx = max(0, min(len(_SPARK) - 1, idx))
        chars.append(_SPARK[idx])
    line = "".join(chars)
    return "%s%s%s [%s→%s]" % (color_for_ratio(0.5), line, C.RESET,
                                _short(lo), _short(hi))


def _short(v):
    if abs(v) >= 1000:
        return "%.0f" % v
    if abs(v) >= 10:
        return "%.1f" % v
    return "%.3f" % v


def color_for_ratio(r):
    try:
        r = min(1.0, max(0.0, float(r)))
    except (TypeError, ValueError):
        r = 0.0
    stops = [(0.00, C.GREEN), (0.20, C.CYAN), (0.45, C.BLUE),
             (0.65, C.YELLOW), (0.85, "\033[38;5;208m"), (1.01, C.RED)]
    chosen = stops[-1][1]
    for lo, code in stops:
        if r >= lo:
            chosen = code
    return chosen
