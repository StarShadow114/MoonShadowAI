# -*- coding: utf-8 -*-
"""MoonShadowAI 训练入口：Textual TUI + 后台训练线程。

用法：
    python scripts/train.py
    python scripts/train.py -E train.epochs=30 train.learning_rate=0.005
    python scripts/train.py --config config/train.yaml -E model.hidden_dim=192
    python scripts/train.py --headless -E train.epochs=2     # 无 TTY 时直跑

输入栏命令：help / pause / resume / status / lr <值> / gen [提示词] / save [路径] / clear
快捷键：ctrl+c 第一次暂停、第二次退出；ctrl+q 直接退出。
"""
import os
import sys
import gc
import math
import time
import queue
import random
import threading
import traceback
import re
from html import escape as _esc

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (PROJECT_ROOT, os.path.join(PROJECT_ROOT, "neural")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import torch.nn as nn

from mobutools import MobuModel  # noqa: E402
from neural.config import load_config  # noqa: E402
from neural.model import LSTMLanguageModel, count_params  # noqa: E402
from neural.dataset import create_dataloader, load_tokens  # noqa: E402
from neural import benchmark as autotune_mod  # noqa: E402
from neural.dashboard import C, fmt_time, fmt_num, bar, term_width  # noqa: E402

try:
    from textual.app import App, ComposeResult
    from textual.containers import Container, Horizontal, Vertical
    from textual.reactive import reactive
    from textual.widgets import Footer, Header, Input, Label, ProgressBar, RichLog, Static
    from textual import __version__ as _TEXTUAL_VER
    TEXTUAL_OK = True
except Exception:                                    # pragma: no cover
    TEXTUAL_OK = False
    _TEXTUAL_VER = "n/a"

try:
    from torch.utils.tensorboard import SummaryWriter
    TB_OK = True
except Exception:                                    # pragma: no cover
    TB_OK = False

BANNER = "[bold cyan]MoonShadowAI · LSTM 语言模型 · 纯 CPU 手写[/bold cyan]"

HELP_TEXT = """可用命令：
  help              显示这份帮助
  pause / resume    暂停 / 继续训练
  status            打印当前进度、最佳成绩与配置摘要
  lr <数值>         实时修改优化器学习率（并写回配置）
  gen [提示词]      在 step 间隙生成一段文本
  save [路径]       立即保存一份 checkpoint
  clear             清空日志面板
命令在输入框回车触发；ctrl+c 第一次暂停、第二次退出；ctrl+q 直接退出。"""


def _strip_ansi(text):
    """Textual/Rich 不认裸 ANSI 转义码，混用会渲染成黑块；统一剥掉，颜色交给 markup 标签。"""
    return re.sub(r"\033\[[0-9;]*m", "", str(text))


# ---------------------------------------------------------------- 终端兜底
def terminal_reset():
    """尽力把终端恢复到干净状态，任何异常都吞掉。"""
    try:
        sys.stdout.write("\033[0m\033[?25h\033[?1000l\033[?1002l\033[?1003l\033[?1049l\033[2q")
        sys.stdout.flush()
    except Exception:
        pass
    try:
        sys.stderr.write("\033[0m")
        sys.stderr.flush()
    except Exception:
        pass
    try:
        import termios
        fd = sys.stdin.fileno()
        termios.tcsetattr(fd, termios.TCSANOW, termios.tcgetattr(fd))
    except Exception:
        pass


# ---------------------------------------------------------------- 组件
if TEXTUAL_OK:

    class MetricCard(Container):
        """一张指标卡：label + 定宽数值，reactive 变化自动局部刷新。"""

        value = reactive(None)

        def __init__(self, label, color, fmt, card_id):
            super().__init__(id=card_id)
            self.label = label
            self.color = color
            self.fmt = fmt
            self._ready = False
            self._pending = None

        def compose(self) -> ComposeResult:
            yield Label(self.label, classes="metric-label")
            yield Static("--", classes="metric-value", id="%s_val" % self.id)

        def on_mount(self) -> None:
            self._ready = True
            if self._pending is not None:
                val, self._pending = self._pending, None
                self._render_value(val)

        def watch_value(self, val) -> None:
            if not self._ready:
                self._pending = val
                return
            self._render_value(val)

        def _render_value(self, val) -> None:
            try:
                text = self.fmt(val) if callable(self.fmt) else str(val)
            except Exception:
                text = "--"
            try:
                w = self.query_one("#%s_val" % self.id, Static)
                w.update("[%s]%s[/%s]" % (self.color, text, self.color))
            except Exception:
                pass

    class SparkLine(Static):
        """轨迹图：按窗口内 min/max 真实归一化，宽度随组件实际宽度自适应。"""

        data = reactive(list)

        def __init__(self, title, color_name="cyan", *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.spark_title = title
            self.spark_color_name = color_name
            self._ready = False

        def on_mount(self) -> None:
            self._ready = True
            self._draw(self.data)

        def watch_data(self, vals) -> None:
            if not self._ready:
                return
            self._draw(vals)

        def _draw(self, vals) -> None:
            vals = list(vals or [])
            w = self.size.width if self.size.width > 0 else max(20, term_width() - 4)
            avail = max(8, w - 2)                 # SparkLine 左右各 1 列内边距
            label = self.spark_title.strip()
            if not vals:
                blank = max(0, min(24, avail - len(label) - 1))
                self.update("%s %s" % (label, "·" * blank))
                return
            lo, hi = min(vals), max(vals)
            span = hi - lo
            # 圆括号：方括号会被 Textual 当成 Rich markup 标签
            tail = ("(%.2f→%.2f)" % (lo, hi)) if abs(hi) < 1000 else ("(%.0f→%.0f)" % (lo, hi))
            bar_w = avail - len(label) - 2 - len(tail)
            if bar_w < 1:                         # 极窄屏丢掉标注，只画轨迹
                tail = ""
                bar_w = avail - len(label) - 1
            bar_w = max(1, min(bar_w, 200))
            chars = []
            n = len(vals)
            for i in range(bar_w):
                a = i * n // bar_w
                b = max(a + 1, (i + 1) * n // bar_w)
                chunk = vals[a:b]
                v = sum(chunk) / len(chunk)
                idx = 3 if span <= 1e-12 else int((v - lo) / span * 7 + 1e-9)
                chars.append("▁▂▃▄▅▆▇█"[max(0, min(7, idx))])
            # 只用 markup 上色，绝不塞裸 ANSI（否则右侧会渲染成黑块）
            line = "%s [%s]%s[/%s]" % (label, self.spark_color_name,
                                       "".join(chars), self.spark_color_name)
            if tail:
                line += " " + tail
            self.update(line)

    CSS = """
    Screen { background: $surface; }
    Header { background: $primary; color: $text; }
    Footer { background: $panel; }
    #titlebar { height: 1; padding: 0 1; background: $boost; color: $text;
                text-style: bold; overflow: hidden hidden; }
    #prog { height: 1; margin: 0 1; }
    #metrics { height: 8; margin: 0 1; padding: 0 0; border: round $primary;
               layout: grid; grid-size: 3; align: center middle; overflow: hidden hidden; }
    MetricCard { width: 1fr; height: 3; border: none; padding: 0 0; margin: 0;
                 align: center middle; }
    MetricCard .metric-label { color: $text-muted; text-align: center; width: 100%; }
    MetricCard .metric-value { text-style: bold; text-align: center; width: 100%; }
    #sparks { height: 4; margin: 0 1; padding: 0 1; border: round $secondary;
              overflow: hidden hidden; }
    SparkLine { height: 1; width: 100%; padding: 0 1; overflow: hidden hidden; }
    #log { height: 1fr; margin: 0 1; padding: 0 1; border: round $success;
           scrollbar-size: 1 1; }
    #cmd { dock: bottom; margin: 0 1; }
    """

    class TrainTUI(App):
        TITLE = "MoonShadowAI · LSTM 语言模型"
        CSS = CSS
        BINDINGS = [("ctrl+c", "toggle_pause", "暂停/退出"),
                    ("ctrl+q", "quit_app", "退出")]

        def __init__(self, cfg, cli_overrides=None):
            super().__init__()
            self.cfg = cfg
            self.cli_overrides = list(cli_overrides or [])
            self.device = torch.device("cpu")
            if hasattr(torch, "set_num_threads"):
                try:
                    torch.set_num_threads(max(1, int(cfg.get("runtime.threads", 4))))
                except Exception:
                    pass
            if hasattr(torch, "set_interop_threads"):
                try:
                    torch.set_interop_threads(max(1, int(cfg.get("runtime.interop_threads", 1))))
                except Exception:
                    pass

            self.st = {
                "epoch": 0, "step": 0, "total_steps": 0, "loss": None, "smooth": None,
                "ppl": None, "lr": float(cfg.get("train.learning_rate", 5e-3)),
                "speed": 0.0, "elapsed": 0.0, "best": float("inf"), "best_epoch": 0,
                "preset": "-", "vocab_size": 0, "params": 0, "num_blocks": 0,
                "gen_queue": queue.Queue(), "save_queue": queue.Queue(),
                "pause_ev": threading.Event(), "stop_ev": threading.Event(),
                "paused": False, "pause_warned": False,
                "running": True, "finished": False, "early_stop": False,
            }
            self.st["pause_ev"].set()
            self.ui_push_every = 5
            self.tb_every = 50
            self.history_loss = []
            self.history_ppl = []
            self._train_thread = None
            self.writer = None
            self._exiting = False
            self._lr_req = 0

        # ---------------- 布局 ----------------
        def compose(self) -> ComposeResult:
            yield Header(show_clock=True)
            yield Static(self._title_text(), id="titlebar")
            yield ProgressBar(total=100, show_eta=False, show_percentage=False, id="prog")
            with Container(id="metrics"):
                yield MetricCard("loss", "cyan", lambda v: fmt_num(v, 8, 4), "m_loss")
                yield MetricCard("ppl", "magenta", lambda v: fmt_num(v, 8, 3), "m_ppl")
                yield MetricCard("lr", "green", lambda v: fmt_num(v, 8, 5), "m_lr")
                yield MetricCard("step/s", "yellow", lambda v: fmt_num(v, 8, 2), "m_spd")
                yield MetricCard("elapsed", "blue", lambda v: fmt_time(v).rjust(8), "m_elp")
                yield MetricCard("ETA", "red", lambda v: (fmt_time(v) if v else "--").rjust(8), "m_eta")
            with Vertical(id="sparks"):
                yield SparkLine("loss", "cyan", id="sp_loss")
                yield SparkLine("ppl ", "magenta", id="sp_ppl")
            yield RichLog(id="log", markup=True, wrap=True, auto_scroll=True, highlight=False)
            yield Input(placeholder="输入命令，help 查看帮助", id="cmd")
            yield Footer()

        def _title_text(self):
            st = self.st
            ep = int(self.cfg.get("train.epochs", 0) or 0)
            pct = min(100, int(st["step"] * 100 / max(1, st["total_steps"])))
            if term_width() < 60:                 # 手机窄屏：短标题，百分比并入本行
                return " MSAI·LSTM ep %s/%s st %s/%s %d%%%s" % (
                    st["epoch"], ep, st["step"], st["total_steps"], pct,
                    " |PAUSE" if not st["pause_ev"].is_set() else "")
            return (" MoonShadowAI · LSTM 语言模型   epoch %s/%s   step %s/%s   %d%%%s" % (
                st["epoch"], ep, st["step"], st["total_steps"], pct,
                " paused" if not st["pause_ev"].is_set() else ""))

        def on_mount(self) -> None:
            self.log_line(BANNER)
            self.log_line("%s[dim]textual %s · torch %s · device %s · threads=%s[/dim]%s" % (
                C.GRAY, _TEXTUAL_VER, torch.__version__, self.device,
                self.cfg.get("runtime.threads", 4), C.RESET))
            if self.cli_overrides:
                self.log_line("%s[dim]命令行覆盖：%s[/dim]%s" % (
                    C.GRAY, " ".join(self.cli_overrides), C.RESET))
            self._train_thread = threading.Thread(target=self._train_entry, name="trainer", daemon=True)
            self._train_thread.start()
            try:
                self.query_one("#cmd", Input).focus()
            except Exception:
                pass

        # ---------------- 日志与刷新 ----------------
        def log_line(self, text):
            text = _strip_ansi(text)
            try:
                w = self.query_one("#log", RichLog)
            except Exception:
                self._pending_log = getattr(self, "_pending_log", [])
                self._pending_log.append(text)
                return
            try:
                w.write(text)
            except Exception:
                pass
            pending = getattr(self, "_pending_log", None)
            if pending:
                for p in pending:
                    try:
                        w.write(p)
                    except Exception:
                        pass
                self._pending_log = []

        def _push_metrics(self, force=False):
            st = self.st
            step, total = st["step"], max(1, st["total_steps"])
            try:
                self.query_one("#prog", ProgressBar).progress = min(100.0, step * 100.0 / total)
            except Exception:
                pass
            mapping = [("m_loss", st["smooth"] if st["smooth"] is not None else st["loss"]),
                       ("m_ppl", st["ppl"]), ("m_lr", st["lr"]), ("m_spd", st["speed"]),
                       ("m_elp", st["elapsed"]),
                       ("m_eta", (total - step) / st["speed"] if st["speed"] > 1e-9 else None)]
            for cid, val in mapping:
                try:
                    card = self.query_one("#" + cid, MetricCard)
                except Exception:
                    continue
                if force or val is not None:
                    card.value = val
            try:
                self.query_one("#titlebar", Static).update(self._title_text())
            except Exception:
                pass
            if len(self.history_loss) >= 2:
                try:
                    self.query_one("#sp_loss", SparkLine).data = list(self.history_loss)
                except Exception:
                    pass
            if len(self.history_ppl) >= 2:
                try:
                    self.query_one("#sp_ppl", SparkLine).data = list(self.history_ppl)
                except Exception:
                    pass

        # ---------------- 快捷键 ----------------
        def action_toggle_pause(self) -> None:
            if self.st.get("finished") or self._exiting:
                self.action_quit_app()
                return
            if self.st["pause_ev"].is_set():
                self.st["pause_ev"].clear()
                self.st["paused"] = True
                self.st["pause_warned"] = False
                self.log_line("%s[yellow]⏸ 已暂停，再次 Ctrl+C 退出[/yellow]%s" % (C.BOLD, C.RESET))
            else:
                self.log_line("%s[yellow]再次 Ctrl+C —— 正在退出，终端会干净收尾…[/yellow]%s" % (C.BOLD, C.RESET))
                self.action_quit_app()

        def action_quit_app(self) -> None:
            if self._exiting:
                return
            self._exiting = True
            self.st["stop_ev"].set()
            self.st["pause_ev"].set()
            self.log_line("%s[dim]正在收尾…[/dim]%s" % (C.GRAY, C.RESET))
            try:
                self.query_one("#cmd", Input).value = ""
            except Exception:
                pass
            try:
                self.exit()
            except Exception:
                pass

        def on_input_submitted(self, event) -> None:
            text = (getattr(event, "value", None) or "").strip()
            try:
                inp = self.query_one("#cmd", Input)
                inp.value = ""
            except Exception:
                pass
            if not text:
                return
            try:
                self._handle_command(text)
            except Exception as exc:
                self.log_line("%s[red]命令出错：%s[/red]%s" % (C.BOLD, _esc(str(exc)), C.RESET))

        # ---------------- 命令 ----------------
        def _handle_command(self, text):
            parts = text.split(None, 1)
            cmd = parts[0].lower()
            arg = (parts[1].strip() if len(parts) > 1 else "")
            st = self.st

            if cmd == "help":
                for line in HELP_TEXT.rstrip().splitlines():
                    self.log_line(line)
            elif cmd in ("pause", "resume"):
                if cmd == "pause" and st["pause_ev"].is_set():
                    st["pause_ev"].clear(); st["paused"] = True
                elif cmd == "resume" and not st["pause_ev"].is_set():
                    st["pause_ev"].set(); st["paused"] = False
                self.log_line("%s[cyan]%s[/cyan]%s" % (C.BOLD, "已继续" if st["pause_ev"].is_set() else "已暂停", C.RESET))
                if not st["pause_ev"].is_set():
                    self.log_line("%s[yellow]已暂停，输入 resume 继续；Ctrl+C 也可切换[/yellow]%s" % (C.BOLD, C.RESET))
            elif cmd == "status":
                best = st["best"]
                eta = (fmt_time((st["total_steps"] - st["step"]) / st["speed"])
                       if st["speed"] > 1e-9 else "--")
                best_s = "∞" if best == float("inf") else "%.4f" % best
                self.log_line(
                    "%s[bold]status[/bold]%s epoch=%s/%s step=%s/%s loss=%s ppl=%s lr=%s speed=%s/s "
                    "elapsed=%s eta=%s\n        best_loss=%s (epoch %s)  vocab=%s params=%s "
                    "blocks=%s preset=%s" % (
                        C.CYAN, C.RESET, st["epoch"], self.cfg.get("train.epochs", 0), st["step"],
                        st["total_steps"], fmt_num(st["loss"], 8, 4), fmt_num(st["ppl"], 8, 3),
                        fmt_num(st["lr"], 8, 5), fmt_num(st["speed"], 6, 2), fmt_time(st["elapsed"]),
                        eta, best_s, st["best_epoch"],
                        st["vocab_size"], st["params"], st["num_blocks"], st["preset"]))
            elif cmd == "lr":
                try:
                    val = float(arg)
                    if val <= 0:
                        raise ValueError
                except ValueError:
                    self.log_line("%s[red]用法：lr <正数>，例如 lr 0.001[/red]%s" % (C.BOLD, C.RESET))
                    return
                st["lr"] = val
                self.cfg.set("train.learning_rate", val)
                self._lr_req = getattr(self, "_lr_req", 0) + 1
                self.log_line("%s[green]学习率已置为 %s（下个 step 生效，并写回配置）[/green]%s" % (C.BOLD, val, C.RESET))
                self._push_metrics(force=True)
            elif cmd == "gen":
                st["gen_queue"].put(arg)
                self.log_line("%s[cyan]生成请求已入队，将在 step 间隙或暂停时执行：%s[/cyan]%s" % (
                    C.BOLD, (arg or "<随机起头>"), C.RESET))
            elif cmd == "save":
                path = arg or os.path.join(self.cfg.get("checkpoint.dir", "neural/checkpoints"),
                                           "manual_step%d.pt" % st["step"])
                st["save_queue"].put(path)
                self.log_line("%s[cyan]存档请求已入队：%s[/cyan]%s" % (C.BOLD, path, C.RESET))
            elif cmd == "clear":
                try:
                    self.query_one("#log", RichLog).clear()
                except Exception:
                    pass
            else:
                self.log_line("%s[red]未知命令：%s（输入 help 查看帮助）[/red]%s" % (C.BOLD, cmd, C.RESET))

        # ---------------- 退出清理 ----------------
        def on_unmount(self) -> None:
            self.st["stop_ev"].set()
            self.st["pause_ev"].set()
            if self.writer is not None:
                try:
                    self.writer.flush(); self.writer.close()
                except Exception:
                    pass
                self.writer = None
            if self._train_thread is not None:
                self._train_thread.join(timeout=8.0)
            gc.collect()

        # ================= 训练线程 =================
        def _train_entry(self):
            model = None
            vocab_mobu = None
            try:
                model, vocab_mobu = self._train_loop()
            except BaseException as exc:                      # noqa: B902
                self.call_from_thread(self.log_line, "%s[red]训练线程异常：%s: %s[/red]%s" % (
                    C.BOLD, type(exc).__name__, exc, C.RESET))
                for line in traceback.format_exc().splitlines()[-12:]:
                    self.call_from_thread(self.log_line, "%s[dim]%s[/dim]%s" % (C.GRAY, line, C.RESET))
            finally:
                self.st["running"] = False
                self.st["finished"] = True
                self.st["pause_ev"].set()
                if model is not None and vocab_mobu is not None:
                    try:
                        self._drain_commands(model, vocab_mobu, self.st.get("epoch", 0))
                    except Exception as exc:
                        self.call_from_thread(self.log_line, "%s[red]收尾命令出错：%s[/red]%s" % (
                            C.BOLD, _esc(str(exc)), C.RESET))
                if self.writer is not None:
                    try:
                        self.writer.flush(); self.writer.close()
                    except Exception:
                        pass
                    self.writer = None
                try:
                    self.call_from_thread(self._on_train_finished)
                except Exception:
                    pass

        def _on_train_finished(self):
            best = self.st["best"]
            msg = "训练结束，最佳 loss %s" % ("∞" if best == float("inf") else "%.4f" % best)
            if self.st.get("early_stop"):
                msg += "（早停触发，第 %d 轮）" % self.st["epoch"]
            self.log_line("%s[bold green]%s[/bold green]%s" % (C.BOLD, msg, C.RESET))
            ckpt = os.path.join(self.cfg.get("checkpoint.dir", "neural/checkpoints"),
                                self.cfg.get("checkpoint.best_name", "best_model.pt"))
            if os.path.exists(_abs(ckpt)):
                self.log_line("%s[dim]最佳权重：%s[/dim]%s" % (C.GRAY, ckpt, C.RESET))
                self.log_line("%s[dim]下一步：python scripts/generate.py -n 2[/dim]%s" % (C.GRAY, C.RESET))
            self._push_metrics(force=True)
            try:
                self.query_one("#cmd", Input).focus()
            except Exception:
                pass

        def _ui(self, fn, *args):
            try:
                self.call_from_thread(fn, *args)
            except Exception:
                pass

        def _maybe_pause(self, model, vocab_mobu):
            """返回 False 表示应当结束训练；暂停时 while paused: sleep(0.1) 空转。"""
            if self.st["stop_ev"].is_set():
                return False
            if not self.st["pause_ev"].is_set():
                self._ui(self.log_line, "%s[bold yellow]已暂停，再次 Ctrl+C 退出[/bold yellow]%s" % (C.BOLD, C.RESET))
                while not self.st["pause_ev"].is_set():
                    if self.st["stop_ev"].is_set():
                        return False
                    try:
                        self._drain_commands(model, vocab_mobu, self.st["epoch"])
                    except Exception as exc:
                        self._ui(self.log_line, "%s[red]暂停期命令出错：%s[/red]%s" % (C.BOLD, _esc(str(exc)), C.RESET))
                    if not self.st["pause_ev"].is_set():
                        time.sleep(0.1)
            return True

        def _drain_commands(self, model, vocab_mobu, epoch):
            """处理 gen / save 队列（训练 step 间隙与暂停期间共用）。"""
            save_q = self.st["save_queue"]
            if not save_q.empty():
                best_path = _abs(self.cfg.get("checkpoint.dir", "neural/checkpoints") + "/" +
                                 self.cfg.get("checkpoint.best_name", "best_model.pt"))
                while not save_q.empty():
                    p = save_q.get()
                    self._save_checkpoint(model, None, epoch if epoch else self.st["epoch"],
                                          self.st["smooth"], self.st["vocab_size"],
                                          best_path if p == best_path else p, manual=True)
            gen_q = self.st["gen_queue"]
            if not gen_q.empty():
                prompt = ""
                while not gen_q.empty():
                    prompt = gen_q.get()
                self._run_generation(model, vocab_mobu, prompt)

        def _train_loop(self):
            cfg = self.cfg
            self._ui(self.log_line, "%s[bold cyan]▸ 初始化[/bold cyan]%s" % (C.BOLD, C.RESET))

            seq_path = _abs(cfg.get("data.seq_path", "neural/data/sequences.mobu"))
            vocab_path = _abs(cfg.get("data.vocab_path", "neural/data/vocab.mobu"))
            block_size = int(cfg.get("model.block_size", 64))
            tokens = load_tokens(seq_path)
            self.st["num_blocks"] = int(tokens.size(0))
            vocab_mobu = MobuModel.load(vocab_path)
            vocab_size = int(vocab_mobu.vocab_size())
            if vocab_size < 4:
                raise RuntimeError("词表异常：只有 %d 词" % vocab_size)
            self.st["vocab_size"] = vocab_size

            batch_size = int(cfg.get("train.batch_size", 256))
            train_dl = create_dataloader(seq_path, block_size, batch_size, shuffle=True,
                                         split=cfg.get("data.train_split", 0.9), split_idx=0)
            val_dl = create_dataloader(seq_path, block_size, batch_size, shuffle=False,
                                       split=cfg.get("data.train_split", 0.9), split_idx=1)

            torch.manual_seed(1337)

            def make_model():
                return LSTMLanguageModel(
                    vocab_size=vocab_size,
                    embed_dim=int(cfg.get("model.embed_dim", 80)),
                    hidden_dim=int(cfg.get("model.hidden_dim", 160)),
                    num_layers=int(cfg.get("model.num_layers", 1)),
                    dropout=float(cfg.get("model.dropout", 0.2)),
                    block_size=block_size).to(self.device)

            model = make_model()
            n_params = count_params(model)
            self.st["params"] = n_params
            self._ui(self.log_line, "  词表      : %d 词（%s）" % (vocab_size, vocab_path))
            self._ui(self.log_line, "  语料块    : %d 块 × %d token（train %d / val %d）" % (
                self.st["num_blocks"], block_size + 1, len(train_dl.dataset), len(val_dl.dataset)))
            self._ui(self.log_line, "  模型      : embed=%d hidden=%d layers=%d dropout=%s block=%d" % (
                model.embed_dim, model.hidden_dim, model.num_layers, model.dropout, block_size))
            self._ui(self.log_line, "  参数量    : %s（约 %.2f M，纯 CPU / %s）" % (
                format(n_params, ","), n_params / 1e6, self.device))
            self._ui(self.log_line, "  初始 batch: %d" % batch_size)

            if bool(cfg.get("autotune.enabled", True)):
                self._ui(self.log_line, "%s[cyan]▸ autotune 实测中（30 步 forward+backward）[/cyan]%s" % (C.BOLD, C.RESET))
                name, sps = autotune_mod.autotune(cfg, vocab_size, self.device)
                self.st["preset"] = name
                self._ui(self.log_line, "  实测      : %.2f step/s → 档位 %s，batch_size=%s" % (
                    sps, name, cfg.get("train.batch_size", batch_size)))
                new_bs = int(cfg.get("train.batch_size", batch_size))
                if new_bs != batch_size:
                    batch_size = new_bs
                    train_dl = create_dataloader(seq_path, block_size, batch_size, shuffle=True,
                                                 split=cfg.get("data.train_split", 0.9), split_idx=0)
                    val_dl = create_dataloader(seq_path, block_size, batch_size, shuffle=False,
                                               split=cfg.get("data.train_split", 0.9), split_idx=1)
                    self._ui(self.log_line, "  已按档位重建 DataLoader，batch_size=%d，训练步数 %d" % (
                        batch_size, len(train_dl)))
                del model
                gc.collect()
                torch.manual_seed(1337)
                model = make_model()
            else:
                self.st["preset"] = "off"

            epochs = int(cfg.get("train.epochs", 50))
            lr = float(cfg.get("train.learning_rate", 5e-3))
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                          weight_decay=float(cfg.get("train.weight_decay", 1e-5)))
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
            criterion = nn.CrossEntropyLoss()
            grad_clip = float(cfg.get("train.grad_clip", 1.0))
            patience = int(cfg.get("train.early_stop_patience", 5))
            min_delta = float(cfg.get("train.min_delta", 1e-4))
            save_every = int(cfg.get("checkpoint.save_every", 5))
            ckpt_dir = _abs(cfg.get("checkpoint.dir", "neural/checkpoints"))
            os.makedirs(ckpt_dir, exist_ok=True)
            best_path = os.path.join(ckpt_dir, str(cfg.get("checkpoint.best_name", "best_model.pt")))

            if TB_OK:
                try:
                    logdir = _abs("neural/logs")
                    os.makedirs(logdir, exist_ok=True)
                    self.writer = SummaryWriter(log_dir=logdir)
                except Exception:
                    self.writer = None
            if self.writer is None:
                self._ui(self.log_line, "%s[dim]tensorboard 不可用，跳过 scalar 写入[/dim]%s" % (C.GRAY, C.RESET))

            steps_per_epoch = len(train_dl)
            total_steps = max(1, steps_per_epoch * epochs)
            self.st["total_steps"] = total_steps
            self.st["lr"] = lr
            self.st["best"] = float("inf")
            self._ui(self.log_line, "%s[bold cyan]▸ 开训[/bold cyan]%s epochs=%d steps/epoch=%d 总步数=%d 早停耐心=%d" % (
                C.BOLD, C.RESET, epochs, steps_per_epoch, total_steps, patience))
            if steps_per_epoch == 0:
                self._ui(self.log_line, "%s[red]训练集中没有样本，检查语料与 train_split[/red]%s" % (C.BOLD, C.RESET))
                return model, vocab_mobu

            smooth = None
            best = float("inf")
            best_epoch = 0
            no_improve = 0
            gstep = 0
            t_start = time.perf_counter()
            win = max(4, min(64, total_steps // 4))
            win_t0, win_step0 = t_start, 0
            last_lr_req = 0

            model.train()
            for epoch in range(1, epochs + 1):
                if not self._maybe_pause(model, vocab_mobu):
                    break
                self.st["epoch"] = epoch
                epoch_loss_sum = 0.0
                epoch_batches = 0
                epoch_t0 = time.perf_counter()

                for bx, (x, y) in enumerate(train_dl, 1):
                    if not self._maybe_pause(model, vocab_mobu):
                        break
                    lr_req = getattr(self, "_lr_req", 0)
                    if lr_req != last_lr_req:
                        last_lr_req = lr_req
                        new_lr = float(self.cfg.get("train.learning_rate", lr))
                        for grp in optimizer.param_groups:
                            grp["lr"] = new_lr
                        self.st["lr"] = new_lr

                    x = x.to(self.device, non_blocking=False)
                    y = y.to(self.device, non_blocking=False)
                    optimizer.zero_grad(set_to_none=True)
                    logits, _hidden = model(x)
                    loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))
                    loss.backward()
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()

                    loss_val = float(loss.item())
                    smooth = loss_val if smooth is None else 0.9 * smooth + 0.1 * loss_val
                    epoch_loss_sum += loss_val
                    epoch_batches += 1
                    gstep += 1
                    self.st["step"] = gstep
                    self.st["loss"] = loss_val
                    self.st["smooth"] = smooth
                    ppl = math.exp(min(smooth, 20.0))
                    self.st["ppl"] = ppl
                    self.st["elapsed"] = time.perf_counter() - t_start

                    if gstep - win_step0 >= win:
                        dt = time.perf_counter() - win_t0
                        if dt > 1e-6:
                            self.st["speed"] = (gstep - win_step0) / dt
                        win_t0 = time.perf_counter()
                        win_step0 = gstep

                    self.history_loss = (self.history_loss + [smooth])[-400:]
                    self.history_ppl = (self.history_ppl + [min(ppl, 1e6)])[-400:]

                    if gstep % self.ui_push_every == 0 or bx == steps_per_epoch:
                        self._ui(self._push_metrics)
                    if self.writer is not None and gstep % self.tb_every == 0:
                        try:
                            self.writer.add_scalar("train/loss", loss_val, gstep)
                            self.writer.add_scalar("train/smooth_loss", smooth, gstep)
                            self.writer.add_scalar("train/ppl", ppl, gstep)
                            self.writer.add_scalar("train/lr", float(optimizer.param_groups[0]["lr"]), gstep)
                            self.writer.add_scalar("train/step_per_sec", self.st["speed"], gstep)
                        except Exception:
                            pass
                    self._drain_commands(model, vocab_mobu, epoch)

                if not self.st["pause_ev"].is_set() or self.st["stop_ev"].is_set():
                    self._drain_commands(model, vocab_mobu, epoch)
                    while not self.st["pause_ev"].is_set() and not self.st["stop_ev"].is_set():
                        self._drain_commands(model, vocab_mobu, epoch)
                        time.sleep(0.1)
                    if self.st["stop_ev"].is_set():
                        break
                    continue

                avg_loss = epoch_loss_sum / max(1, epoch_batches)
                avg_ppl = math.exp(min(avg_loss, 20.0))
                dt_epoch = time.perf_counter() - epoch_t0
                cur_lr = float(optimizer.param_groups[0]["lr"])
                scheduler.step()
                self._ui(self.log_line,
                         "%s[bold]epoch %d/%d[/bold]%s avg_loss=%s avg_ppl=%s lr=%s 用时=%s 步数=%d%s" % (
                             C.CYAN, epoch, epochs, C.RESET, fmt_num(avg_loss, 9, 4),
                             fmt_num(avg_ppl, 9, 3), fmt_num(cur_lr, 9, 5), fmt_time(dt_epoch),
                             epoch_batches,
                             ("  %s" % bar(min(1.0, epoch / max(1, epochs)), 12)) if term_width() > 90 else ""))
                if self.writer is not None:
                    try:
                        self.writer.add_scalar("epoch/avg_loss", avg_loss, epoch)
                        self.writer.add_scalar("epoch/avg_ppl", avg_ppl, epoch)
                        self.writer.add_scalar("epoch/lr", cur_lr, epoch)
                    except Exception:
                        pass

                improved = avg_loss < best - min_delta
                if improved:
                    best = avg_loss
                    best_epoch = epoch
                    no_improve = 0
                    self.st["best"] = best
                    self.st["best_epoch"] = epoch
                    self._save_checkpoint(model, optimizer, epoch, avg_loss, vocab_size, best_path)
                    self._ui(self.log_line, "%s[green]  ↳ 刷新最佳 loss=%.4f，已存 %s[/green]%s" % (
                        C.BOLD, best, os.path.relpath(best_path, PROJECT_ROOT), C.RESET))
                else:
                    no_improve += 1
                    self._ui(self.log_line, "%s[dim]  ↳ 未改善（best=%.4f，连续 %d/%d 轮）[/dim]%s" % (
                        C.GRAY, best, no_improve, patience, C.RESET))

                if save_every > 0 and epoch % save_every == 0 and not improved:
                    self._save_checkpoint(model, optimizer, epoch, avg_loss, vocab_size,
                                          os.path.join(ckpt_dir, "epoch%03d.pt" % epoch))
                if no_improve >= patience:
                    self.st["early_stop"] = True
                    self._ui(self.log_line, "%s[yellow]连续 %d 轮改善不足 %s，触发早停，收工。[/yellow]%s" % (
                        C.BOLD, patience, min_delta, C.RESET))
                    break

                if len(val_dl.dataset) > 0:
                    val_loss = self._evaluate(model, val_dl, criterion, vocab_size)
                    self._ui(self.log_line, "%s[dim]  ↳ val_loss=%s val_ppl=%s[/dim]%s" % (
                        C.GRAY, fmt_num(val_loss, 8, 4), fmt_num(math.exp(min(val_loss, 20.0)), 8, 3), C.RESET))
                    if self.writer is not None:
                        try:
                            self.writer.add_scalar("val/loss", val_loss, epoch)
                            self.writer.add_scalar("val/ppl", math.exp(min(val_loss, 20.0)), epoch)
                        except Exception:
                            pass
                    model.train()

            total_elapsed = time.perf_counter() - t_start
            self.st["elapsed"] = total_elapsed
            if gstep > 0 and total_elapsed > 1e-6:
                self.st["speed"] = gstep / total_elapsed
            self._ui(self._push_metrics, True)
            return model, vocab_mobu

        # ---------------- 训练线程里的辅助 ----------------
        def _evaluate(self, model, dl, criterion, vocab_size):
            model.eval()
            total = 0.0
            n = 0
            with torch.no_grad():
                for x, y in dl:
                    x = x.to(self.device)
                    y = y.to(self.device)
                    logits, _h = model(x)
                    loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))
                    total += float(loss.item()) * x.size(0)
                    n += x.size(0)
            return total / max(1, n)

        def _save_checkpoint(self, model, optimizer, epoch, loss, vocab_size, path, manual=False):
            try:
                os.makedirs(os.path.dirname(_abs(path)), exist_ok=True)
                payload = {
                    "epoch": int(epoch),
                    "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
                    "loss": float(loss if loss is not None else 0.0),
                    "vocab_size": int(vocab_size),
                    "embed_dim": int(model.embed_dim),
                    "hidden_dim": int(model.hidden_dim),
                    "num_layers": int(model.num_layers),
                    "block_size": int(model.block_size),
                    "config": self.cfg.as_dict(),
                    "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                torch.save(payload, _abs(path))
                tag = "手动存档" if manual else "checkpoint"
                self._ui(self.log_line, "%s[green]  ✓ %s：%s（epoch %d, loss %.4f）[/green]%s" % (
                    C.BOLD, tag, os.path.relpath(path, PROJECT_ROOT), epoch, loss or 0.0, C.RESET))
            except Exception as exc:
                self._ui(self.log_line, "%s[red]  ✗ 存档失败 %s: %s[/red]%s" % (C.BOLD, path, exc, C.RESET))

        def _run_generation(self, model, vocab_mobu, prompt):
            was_training = model.training
            model.eval()
            try:
                text = generate_from(model, vocab_mobu, prompt=prompt, n=1, max_len=60,
                                     temperature=0.8, top_k=20, top_p=0.9, device=self.device)
                self._ui(self.log_line, "%s[bold magenta]gen ›[/bold magenta] %s%s%s" % (
                    C.BOLD, C.MAGENTA, text[0] if text else "(空)", C.RESET))
            except Exception as exc:
                self._ui(self.log_line, "%s[red]生成失败：%s: %s[/red]%s" % (C.BOLD, type(exc).__name__, exc, C.RESET))
            finally:
                if was_training:
                    model.train()

    # ------------------------------------------------------------ 生成（训练线程内复用）
    @torch.no_grad()
    def generate_from(model, vocab_mobu, prompt="", n=1, max_len=100, temperature=0.8,
                      top_k=20, top_p=0.9, device=None, seed=None):
        """温度 + top-k + top-p 核采样，返回字符串列表。"""
        device = torch.device(device or getattr(model, "device"))
        temps = _TempAdapter(vocab_mobu)
        if seed is not None:
            torch.manual_seed(seed)
        ids = temps.encode(prompt or "")
        ids = [i for i in ids if i >= 4] or [4 + random.randrange(max(1, temps.vocab_size() - 4))]
        block_size = int(getattr(model, "block_size", 64))
        temp = max(1e-5, float(temperature))
        outs = []
        for _ in range(max(1, int(n))):
            ctx = ids[:]
            hidden = None
            for _step in range(max(1, int(max_len))):
                window = ctx[-block_size:] if len(ctx) > block_size else ctx
                x = torch.tensor([window], dtype=torch.long, device=device)
                logits, hidden = model(x, hidden)
                log = logits[0, -1] / temp
                log = _top_k_mask(log, top_k)
                log = _top_p_mask(log, top_p)
                if bool((log == float("-inf")).all().item()):
                    nxt = int(torch.argmax(logits[0, -1]).item())
                else:
                    nxt = int(torch.multinomial(torch.softmax(log, dim=-1), 1).item())
                if nxt <= 3 or nxt >= temps.vocab_size():
                    break
                ctx.append(nxt)
            outs.append(temps.decode([i for i in ctx if i >= 4]))
        return outs

    class _TempAdapter:
        """词表接口兼容探测：encode/decode、word_to_id/id_to_word、纯 list。"""

        def __init__(self, mobu):
            self.mobu = mobu
            self.vocab = list(mobu.get_vocab()) if hasattr(mobu, "get_vocab") else list(mobu.vocab)
            if not self.vocab:
                raise RuntimeError("词表为空，无法编码/解码，请检查 vocab.mobu")
            self._table = {t: i for i, t in enumerate(self.vocab)}
            self._enc = None
            for name in ("encode", "token_to_id", "word_to_id", "str_to_id"):
                if callable(getattr(mobu, name, None)):
                    self._enc = name
                    break
            self._dec = None
            for name in ("decode", "id_to_word", "token_from_id", "id_to_token"):
                if callable(getattr(mobu, name, None)):
                    self._dec = name
                    break

        def vocab_size(self):
            return len(self.vocab)

        def encode(self, text):
            if self._enc:
                try:
                    out = getattr(self.mobu, self._enc)(text)
                    if isinstance(out, (list, tuple)):
                        return [int(i) for i in out]
                    if isinstance(out, int):
                        return [out]
                except Exception:
                    pass
            return [self._table.get(ch, 1) for ch in str(text)]

        def decode(self, ids):
            if self._dec:
                try:
                    out = getattr(self.mobu, self._dec)(ids)
                    if isinstance(out, str):
                        return out
                    if isinstance(out, (list, tuple)):
                        return "".join(str(x) for x in out)
                except Exception:
                    pass
            return "".join(self.vocab[i] if 0 <= i < len(self.vocab) else "" for i in ids)

    def _top_k_mask(logits, k):
        k = int(k or 0)
        if k <= 0 or k >= logits.numel():
            return logits
        floor = torch.topk(logits, k).values[-1]
        return torch.where(logits >= floor, logits, torch.full_like(logits, float("-inf")))

    def _top_p_mask(logits, p):
        p = float(p or 1.0)
        if p <= 0 or p >= 1.0:
            return logits
        order = torch.argsort(logits, descending=True)
        sl = logits[order]
        cum = torch.cumsum(torch.softmax(sl, dim=-1), dim=-1)
        mask = cum > p
        if bool(mask.all().item()):
            mask[0] = False
        sl = torch.where(mask, torch.full_like(sl, float("-inf")), sl)
        out = torch.full_like(logits, float("-inf"))
        out[order] = sl
        return out

else:                                                  # pragma: no cover
    class TrainTUI:                                    # type: ignore
        def __init__(self, *a, **kw):
            raise RuntimeError("textual 未安装，无法启动 TUI")

    class MetricCard:                                  # type: ignore
        pass

    class SparkLine:                                   # type: ignore
        pass


def _abs(p):
    p = os.path.expanduser(str(p))
    return p if os.path.isabs(p) else os.path.join(PROJECT_ROOT, p)


# ---------------------------------------------------------------- 命令行
def parse_args(argv):
    cfg_path = None
    overrides = []
    headless = False
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("--config", "-c"):
            i += 1
            if i < len(argv):
                cfg_path = argv[i]
        elif a.startswith("--config="):
            cfg_path = a.split("=", 1)[1]
        elif a in ("-E", "--edit"):
            i += 1
            while i < len(argv) and not argv[i].startswith("-"):
                overrides.append(argv[i])
                i += 1
            continue
        elif a.startswith("-E=") or a.startswith("--edit="):
            overrides.append(a.split("=", 1)[1])
        elif a in ("--headless", "-H"):
            headless = True
        elif a in ("-h", "--help"):
            print(__doc__)
            raise SystemExit(0)
        else:
            print("[warn] 忽略未知参数：%s" % a)
        i += 1
    return cfg_path, overrides, headless


def apply_overrides(cfg, overrides):
    for item in overrides:
        if "=" not in item:
            print("[warn] 忽略非法覆盖（缺 =）：%s" % item)
            continue
        key, val = item.split("=", 1)
        key, val = key.strip(), val.strip()
        low = val.lower()
        if low in ("true", "false"):
            value = (low == "true")
        else:
            try:
                value = int(val)
            except ValueError:
                try:
                    value = float(val)
                except ValueError:
                    value = val.strip('"\'')
        cfg.set(key, value)
    return cfg


def headless_train(cfg):
    """无 TTY / textual 缺失时的降级路径：同样的训练循环，纯文本心跳。"""
    import csv
    seq_path = _abs(cfg.get("data.seq_path", "neural/data/sequences.mobu"))
    vocab_path = _abs(cfg.get("data.vocab_path", "neural/data/vocab.mobu"))
    block_size = int(cfg.get("model.block_size", 64))
    tokens = load_tokens(seq_path)
    vocab_mobu = MobuModel.load(vocab_path)
    vocab_size = int(vocab_mobu.vocab_size())
    batch_size = int(cfg.get("train.batch_size", 256))
    train_dl = create_dataloader(seq_path, block_size, batch_size, shuffle=True,
                                 split=cfg.get("data.train_split", 0.9), split_idx=0)
    device = torch.device("cpu")
    if hasattr(torch, "set_num_threads"):
        try:
            torch.set_num_threads(max(1, int(cfg.get("runtime.threads", 4))))
        except Exception:
            pass
    torch.manual_seed(1337)

    def make_model():
        return LSTMLanguageModel(vocab_size=vocab_size,
                                 embed_dim=int(cfg.get("model.embed_dim", 80)),
                                 hidden_dim=int(cfg.get("model.hidden_dim", 160)),
                                 num_layers=int(cfg.get("model.num_layers", 1)),
                                 dropout=float(cfg.get("model.dropout", 0.2)),
                                 block_size=block_size).to(device)

    model = make_model()
    print("参数量 %s | vocab %d | blocks %d | batch %d | steps/epoch %d" % (
        format(count_params(model), ","), vocab_size, tokens.size(0), batch_size, len(train_dl)))
    if bool(cfg.get("autotune.enabled", True)):
        name, sps = autotune_mod.autotune(cfg, vocab_size, device)
        print("autotune: %s %.2f step/s -> batch %s" % (name, sps, cfg.get("train.batch_size", batch_size)))
        if int(cfg.get("train.batch_size", batch_size)) != batch_size:
            batch_size = int(cfg.get("train.batch_size"))
            train_dl = create_dataloader(seq_path, block_size, batch_size, shuffle=True,
                                         split=cfg.get("data.train_split", 0.9), split_idx=0)
        del model
        gc.collect()
        torch.manual_seed(1337)
        model = make_model()
    epochs = int(cfg.get("train.epochs", 50))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("train.learning_rate", 5e-3)),
                                  weight_decay=float(cfg.get("train.weight_decay", 1e-5)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    criterion = nn.CrossEntropyLoss()
    grad_clip = float(cfg.get("train.grad_clip", 1.0))
    patience = int(cfg.get("train.early_stop_patience", 5))
    min_delta = float(cfg.get("train.min_delta", 1e-4))
    ckpt_dir = _abs(cfg.get("checkpoint.dir", "neural/checkpoints"))
    os.makedirs(ckpt_dir, exist_ok=True)
    best_path = os.path.join(ckpt_dir, str(cfg.get("checkpoint.best_name", "best_model.pt")))
    best, no_improve, gstep = float("inf"), 0, 0
    smooth = None
    t0 = time.perf_counter()
    history_path = os.path.join(ckpt_dir, "history.csv")
    csv_fh = open(history_path, "w", encoding="utf-8", newline="")
    writer = csv.writer(csv_fh)
    writer.writerow(["epoch", "step", "avg_loss", "ppl", "lr", "elapsed"])
    model.train()
    for epoch in range(1, epochs + 1):
        ep_sum, ep_n = 0.0, 0
        ep_t0 = time.perf_counter()
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, _h = model(x)
            loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            lv = float(loss.item())
            smooth = lv if smooth is None else 0.9 * smooth + 0.1 * lv
            ep_sum += lv
            ep_n += 1
            gstep += 1
            if gstep % 20 == 0:
                el = time.perf_counter() - t0
                print("epoch %d step %d/%d loss=%.4f smooth=%.4f ppl=%.3f lr=%.5f %s  %.1f step/s" % (
                    epoch, gstep, len(train_dl) * epochs, lv, smooth, math.exp(min(smooth, 20)),
                    float(optimizer.param_groups[0]["lr"]), fmt_time(el), gstep / max(1e-9, el)))
        avg = ep_sum / max(1, ep_n)
        cur_lr = float(optimizer.param_groups[0]["lr"])
        scheduler.step()
        writer.writerow([epoch, gstep, avg, math.exp(min(avg, 20)), cur_lr, round(time.perf_counter() - t0, 2)])
        csv_fh.flush()
        print("%sepoch %d/%d%s avg_loss=%.4f avg_ppl=%.3f 用时=%s" % (
            C.CYAN, epoch, epochs, C.RESET, avg, math.exp(min(avg, 20)), fmt_time(time.perf_counter() - ep_t0)))
        if avg < best - min_delta:
            best, no_improve = avg, 0
            torch.save({"epoch": epoch, "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                        "optimizer_state_dict": optimizer.state_dict(), "loss": avg,
                        "vocab_size": vocab_size, "embed_dim": model.embed_dim,
                        "hidden_dim": model.hidden_dim, "num_layers": model.num_layers,
                        "block_size": block_size, "config": cfg.as_dict()}, best_path)
            print("%s  ↳ best %.4f 已存 %s%s" % (C.GREEN, best, best_path, C.RESET))
        else:
            no_improve += 1
            print("%s  ↳ 未改善 %d/%d%s" % (C.GRAY, no_improve, patience, C.RESET))
            if no_improve >= patience:
                print("%s早停触发%s" % (C.YELLOW, C.RESET))
                break
    csv_fh.close()
    print("%s训练结束，最佳 loss %.4f%s" % (C.BOLD + C.GREEN, best, C.RESET))
    return best


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    cfg_path, overrides, headless = parse_args(argv)
    cfg = load_config(cfg_path)
    apply_overrides(cfg, overrides)
    if hasattr(torch, "set_num_threads"):
        try:
            torch.set_num_threads(max(1, int(cfg.get("runtime.threads", 4))))
        except Exception:
            pass

    want_tui = (not headless) and TEXTUAL_OK and bool(sys.stdin.isatty() and sys.stdout.isatty())
    if not want_tui:
        if not headless:
            print("%s[dim]非 TTY 环境或 textual 不可用，降级为 headless 训练模式%s" % (C.GRAY, C.RESET))
        try:
            headless_train(cfg)
        except KeyboardInterrupt:
            print("\n%s[dim]已中断（KeyboardInterrupt），终端已复位%s" % (C.GRAY, C.RESET))
        finally:
            terminal_reset()
        return 0

    app = TrainTUI(cfg, overrides)
    rc = 0
    try:
        rc = app.run()
    except KeyboardInterrupt:
        print("\n%s[dim]KeyboardInterrupt —— 正在收尾%s" % (C.GRAY, C.RESET))
        try:
            app.st["stop_ev"].set()
            app.st["pause_ev"].set()
        except Exception:
            pass
    except Exception as exc:
        print("%s[red]TUI 异常：%s: %s%s" % (C.BOLD, type(exc).__name__, exc, C.RESET))
        for line in traceback.format_exc().splitlines()[-15:]:
            print("%s[dim]%s%s" % (C.GRAY, line, C.RESET))
        rc = 1
    finally:
        try:
            app.st["stop_ev"].set()
            app.st["pause_ev"].set()
        except Exception:
            pass
        if getattr(app, "_train_thread", None) is not None:
            app._train_thread.join(timeout=8.0)
        if getattr(app, "writer", None) is not None:
            try:
                app.writer.flush()
                app.writer.close()
            except Exception:
                pass
        terminal_reset()
    return int(rc or 0)


if __name__ == "__main__":
    sys.exit(main())
