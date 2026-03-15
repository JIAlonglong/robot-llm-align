"""
ToolRegistry — 注册并分发 <tool_call> 请求
格式：<tool_call>tool_name(arg1=val1, arg2=val2)</tool_call>
"""
import re
from typing import Callable, Any


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Callable] = {}

    def register(self, name: str, func: Callable):
        self._tools[name] = func
        print(f"[ToolRegistry] 注册工具: {name}")

    def execute(self, tool_call_str: str) -> Any:
        """解析并执行 <tool_call>...</tool_call> 中的内容"""
        # 提取标签内容
        match = re.search(r"<tool_call>(.*?)</tool_call>", tool_call_str, re.DOTALL)
        if not match:
            return {"error": "未找到 <tool_call> 标签"}

        call_str = match.group(1).strip()

        # 解析函数名和参数：name(k=v, k=v)
        fn_match = re.match(r"(\w+)\((.*)\)$", call_str, re.DOTALL)
        if not fn_match:
            return {"error": f"无法解析调用: {call_str}"}

        name = fn_match.group(1)
        args_str = fn_match.group(2).strip()

        if name not in self._tools:
            return {"error": f"未知工具: {name}"}

        # 解析 kwargs（只支持简单字面量）
        kwargs = {}
        if args_str:
            for part in re.split(r",\s*(?=\w+=)", args_str):
                if "=" in part:
                    k, v = part.split("=", 1)
                    kwargs[k.strip()] = _parse_value(v.strip())

        try:
            return self._tools[name](**kwargs)
        except Exception as e:
            return {"error": str(e)}


def _parse_value(s: str) -> Any:
    """把字符串字面量转成 Python 类型"""
    if s in ("True", "true"):   return True
    if s in ("False", "false"): return False
    if s in ("None", "null"):   return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s.strip("\"'")
