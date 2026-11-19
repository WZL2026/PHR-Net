import sys
import os
import re
from datetime import datetime

# 匹配 ANSI 转义序列
_ANSI_RE = re.compile(r'\x1b\[[0-9;]*[a-zA-Z]')


class Logger:
    """将 stdout 同时输出到终端和时间戳日志文件。

    使用行缓冲：累积写入直到收到 \\n，然后向日志文件写入完整行。
    遇到 \\r（tqdm 进度条）时覆盖缓冲区中的当前行，进度条本身不写入日志。
    """

    def __init__(self, log_dir="logs", prefix="run"):
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(log_dir, f"{prefix}_{ts}.log")
        self.terminal = sys.stdout
        self.log_file = open(self.log_path, "a", encoding="utf-8", buffering=1)
        self._line_buf = ""          # 当前正在构建的行（尚未收到 \n）
        self._prev_line_was_progress = False

    def _is_progress_bar(self, line):
        """判断一行是否像 tqdm 风格的进度条。"""
        return bool(re.search(r'\d+%|█|it/s|it\/s', line))

    def _flush_line(self):
        """将缓冲的一行写入日志文件（跳过进度条行）。"""
        line = _ANSI_RE.sub('', self._line_buf)
        line = line.strip('\r')
        if line.strip() and not self._is_progress_bar(line):
            self.log_file.write(line + '\n')
            self._prev_line_was_progress = False
        elif self._is_progress_bar(line):
            self._prev_line_was_progress = True
        self._line_buf = ""

    def write(self, message):
        # 终端始终实时输出
        self.terminal.write(message)

        # 按字符处理以正确处理 \r 和 \n
        for ch in message:
            if ch == '\n':
                self._flush_line()
            elif ch == '\r':
                # 回车：覆盖当前行（用于进度条），重置缓冲区
                self._line_buf = ""
            else:
                self._line_buf += ch

    def flush(self):
        self.terminal.flush()
        if self._line_buf:
            self._flush_line()
        self.log_file.flush()

    def close(self):
        if self._line_buf:
            self._flush_line()
        self.log_file.flush()
        self.log_file.close()
        sys.stdout = self.terminal


def count_params(model):
    """返回 (总参数, 可训练参数) 数量，单位为百万。"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total / 1e6, trainable / 1e6


def section_header(title, width=80):
    """格式化的节标题。"""
    sep = "=" * width
    return f"\n{sep}\n  {title}\n{sep}"


def metric_table(headers, rows, col_widths=None):
    """渲染对齐的表格。headers: list[str], rows: list[list[str]]。"""
    if col_widths is None:
        col_widths = [max(len(str(r[i])) for r in [headers] + rows) + 2
                      for i in range(len(headers))]
    lines = []
    hdr = "".join(f"{h:<{w}}" for h, w in zip(headers, col_widths))
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for row in rows:
        lines.append("".join(f"{str(v):<{w}}" for v, w in zip(row, col_widths)))
    return "\n".join(lines)
