"""Agent 工具调用框架（Function Calling）

核心能力：
    1. 工具注册：装饰器注册 + 自动生成 JSON Schema
    2. 参数校验：类型检查 + 必填检查 + 范围校验
    3. 多轮调用：LLM 可连续调用多个工具
    4. 错误处理：工具异常捕获 + 重试 + 降级
    5. 成本控制：调用次数熔断 + 超时控制

两种模式：
    有 DEEPSEEK_API_KEY → 真实调用 LLM 的 function calling
    没有 → 本地规则引擎模拟（演示流程，不花钱）

用法：
    python agent_tools.py "帮我查一下数据库里有多少条用户记录"
    python agent_tools.py "北京天气怎么样" --verbose
"""
import json
import os
import re
import sqlite3
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable

BASE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE, "agent_demo.db")

MAX_TOOL_CALLS = 8        # 单次对话最多调 8 次工具（防死循环）
TOOL_TIMEOUT = 10         # 单个工具超时秒数
MAX_RETRY = 2             # 工具失败重试次数


# ============================================================
# 一、工具注册表
# ============================================================

@dataclass
class Tool:
    name: str
    description: str
    parameters: dict          # JSON Schema
    func: Callable
    call_count: int = 0
    fail_count: int = 0


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, name: str, description: str, parameters: dict):
        """装饰器：注册一个工具"""
        def deco(fn):
            self._tools[name] = Tool(
                name=name, description=description,
                parameters=parameters, func=fn,
            )
            return fn
        return deco

    def get(self, name: str):
        return self._tools.get(name)

    def schemas(self):
        """导出给 LLM 的工具描述（OpenAI 格式）"""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in self._tools.values()
        ]

    def stats(self):
        return {
            name: {"calls": t.call_count, "fails": t.fail_count}
            for name, t in self._tools.items()
        }


REG = ToolRegistry()


# ============================================================
# 二、参数校验
# ============================================================

class ParamError(Exception):
    pass


def validate_params(tool: Tool, args: dict):
    """按 JSON Schema 校验参数"""
    schema = tool.parameters
    props = schema.get("properties", {})
    required = set(schema.get("required", []))

    # 必填检查
    for key in required:
        if key not in args or args[key] in (None, ""):
            raise ParamError(f"缺少必填参数: {key}")

    # 类型检查
    for key, val in args.items():
        if key not in props:
            continue
        expected = props[key].get("type")
        if expected == "integer" and not isinstance(val, int):
            try:
                args[key] = int(val)
            except (TypeError, ValueError):
                raise ParamError(f"参数 {key} 应该是整数，收到 {val!r}")
        elif expected == "number":
            try:
                args[key] = float(val)
            except (TypeError, ValueError):
                raise ParamError(f"参数 {key} 应该是数字，收到 {val!r}")
        elif expected == "string" and not isinstance(val, str):
            args[key] = str(val)

        # 范围检查
        if expected in ("integer", "number"):
            mn = props[key].get("minimum")
            mx = props[key].get("maximum")
            if mn is not None and args[key] < mn:
                raise ParamError(f"参数 {key} 不能小于 {mn}")
            if mx is not None and args[key] > mx:
                raise ParamError(f"参数 {key} 不能大于 {mx}")

    return True


# ============================================================
# 三、工具实现（5 类场景）
# ============================================================

# ---- 1. 数据库查询 ----
def _init_demo_db():
    if os.path.exists(DB_PATH):
        return
    conn = sqlite3.connect(DB_PATH)
    conn.executescript("""
        CREATE TABLE users(id INTEGER PRIMARY KEY, name TEXT, city TEXT, age INTEGER);
        CREATE TABLE orders(id INTEGER PRIMARY KEY, user_id INTEGER, amount REAL);
    """)
    conn.executemany(
        "INSERT INTO users(name, city, age) VALUES(?,?,?)",
        [("张三", "西安", 24), ("李四", "北京", 28), ("王五", "西安", 31), ("赵六", "上海", 26)],
    )
    conn.executemany(
        "INSERT INTO orders(user_id, amount) VALUES(?,?)",
        [(1, 99.5), (1, 200), (2, 150), (3, 88), (4, 500)],
    )
    conn.commit()
    conn.close()


_init_demo_db()


@REG.register(
    name="query_database",
    description="执行只读 SQL 查询（仅 SELECT），返回结果行。数据库有 users(id,name,city,age) 和 orders(id,user_id,amount) 两张表。",
    parameters={
        "type": "object",
        "properties": {
            "sql": {"type": "string", "description": "SQL 查询语句，必须以 SELECT 开头"},
        },
        "required": ["sql"],
    },
)
def query_database(sql: str):
    if not sql.strip().lower().startswith("select"):
        raise ValueError("只允许 SELECT 查询")
    if ";" in sql.strip()[:-1]:
        raise ValueError("不允许多条语句")
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.execute(sql)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
        return {"columns": cols, "rows": rows, "count": len(rows)}
    finally:
        conn.close()


# ---- 2. 文件操作 ----
_SAFE_DIR = os.path.join(BASE, "agent_files")
os.makedirs(_SAFE_DIR, exist_ok=True)


@REG.register(
    name="read_file",
    description="读取指定文本文件的内容。只能读 agent_files 目录下的文件。",
    parameters={
        "type": "object",
        "properties": {
            "filename": {"type": "string", "description": "文件名，例如 notes.txt"},
        },
        "required": ["filename"],
    },
)
def read_file(filename: str):
    # 防路径穿越
    if "/" in filename or "\\" in filename or ".." in filename:
        raise ValueError("文件名不能包含路径分隔符")
    path = os.path.join(_SAFE_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在: {filename}")
    with open(path, encoding="utf-8") as f:
        return {"filename": filename, "content": f.read()[:2000]}


@REG.register(
    name="write_file",
    description="把内容写入文本文件（覆盖）。只能写 agent_files 目录。",
    parameters={
        "type": "object",
        "properties": {
            "filename": {"type": "string", "description": "文件名"},
            "content": {"type": "string", "description": "要写入的完整内容"},
        },
        "required": ["filename", "content"],
    },
)
def write_file(filename: str, content: str):
    if "/" in filename or "\\" in filename or ".." in filename:
        raise ValueError("文件名不能包含路径分隔符")
    path = os.path.join(_SAFE_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return {"ok": True, "filename": filename, "bytes": len(content.encode())}


# ---- 3. HTTP 请求 ----
@REG.register(
    name="http_get",
    description="发送 HTTP GET 请求获取网页或 API 内容。只允许白名单域名。",
    parameters={
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "完整的 http/https URL"},
            "max_chars": {"type": "integer", "description": "最多返回多少字符，默认 1000，最大 5000",
                          "minimum": 100, "maximum": 5000},
        },
        "required": ["url"],
    },
)
def http_get(url: str, max_chars: int = 1000):
    ALLOW = ["httpbin.org", "api.github.com", "wttr.in"]
    host = urllib.parse.urlparse(url).hostname or ""
    if not any(host == a or host.endswith("." + a) for a in ALLOW):
        raise PermissionError(f"域名不在白名单: {host}")
    req = urllib.request.Request(url, headers={"User-Agent": "agent-tools/1.0"})
    with urllib.request.urlopen(req, timeout=TOOL_TIMEOUT) as r:
        body = r.read().decode("utf-8", errors="ignore")
    return {"status": r.status, "content": body[:max_chars]}


# ---- 4. 计算器 ----
@REG.register(
    name="calculator",
    description="计算数学表达式，支持 + - * / ** 和括号。不要用它做别的（不能执行代码）。",
    parameters={
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "数学表达式，例如 (1+2)*3"},
        },
        "required": ["expression"],
    },
)
def calculator(expression: str):
    # 白名单字符，防注入
    if not re.fullmatch(r"[0-9+\-*/().\s]+", expression):
        raise ValueError("表达式包含非法字符")
    try:
        result = eval(expression, {"__builtins__": {}}, {})
    except Exception as e:
        raise ValueError(f"计算失败: {e}")
    return {"expression": expression, "result": result}


# ---- 5. 消息推送（模拟）----
_MESSAGES = []


@REG.register(
    name="send_notification",
    description="给指定用户发送一条站内通知（演示用，只记录不真发）。",
    parameters={
        "type": "object",
        "properties": {
            "user": {"type": "string", "description": "收件人"},
            "message": {"type": "string", "description": "通知内容"},
        },
        "required": ["user", "message"],
    },
)
def send_notification(user: str, message: str):
    _MESSAGES.append({"user": user, "message": message, "at": time.strftime("%H:%M:%S")})
    return {"ok": True, "total_sent": len(_MESSAGES), "last": _MESSAGES[-1]}


# ============================================================
# 四、工具执行器（含重试、超时、熔断）
# ============================================================

@dataclass
class ExecResult:
    ok: bool
    data: Any = None
    error: str = ""
    elapsed_ms: float = 0
    retries: int = 0


def execute_tool(name: str, args: dict) -> ExecResult:
    t0 = time.time()
    tool = REG.get(name)
    if not tool:
        return ExecResult(ok=False, error=f"未知工具: {name}")

    # 参数校验
    try:
        validate_params(tool, args)
    except ParamError as e:
        tool.fail_count += 1
        return ExecResult(
            ok=False, error=f"参数错误: {e}",
            elapsed_ms=(time.time() - t0) * 1000,
        )

    # 执行 + 重试
    last_err = ""
    for attempt in range(MAX_RETRY + 1):
        try:
            tool.call_count += 1
            data = tool.func(**args)
            return ExecResult(
                ok=True, data=data,
                elapsed_ms=(time.time() - t0) * 1000,
                retries=attempt,
            )
        except (ValueError, PermissionError, FileNotFoundError) as e:
            # 这些是确定性错误，重试无意义
            tool.fail_count += 1
            return ExecResult(
                ok=False, error=f"{type(e).__name__}: {e}",
                elapsed_ms=(time.time() - t0) * 1000,
            )
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            if attempt < MAX_RETRY:
                time.sleep(0.5 * (attempt + 1))   # 退避
                continue
            tool.fail_count += 1

    return ExecResult(
        ok=False, error=f"重试 {MAX_RETRY} 次仍失败: {last_err}",
        elapsed_ms=(time.time() - t0) * 1000,
        retries=MAX_RETRY,
    )


# ============================================================
# 五、LLM 决策（两种模式）
# ============================================================

def llm_decide_real(messages, api_key):
    """真实调用 DeepSeek 的 function calling"""
    payload = json.dumps({
        "model": "deepseek-chat",
        "messages": messages,
        "tools": REG.schemas(),
        "temperature": 0.1,
    }).encode()
    req = urllib.request.Request(
        "https://api.deepseek.com/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def llm_decide_mock(query: str, history: list):
    """本地规则引擎：模拟 LLM 的工具决策（没 key 时用）"""
    q = query.lower()

    # 已有工具结果 → 生成最终回答
    if history:
        last = history[-1]
        if last.get("ok"):
            return {"type": "answer", "content": f"查询完成，结果：{json.dumps(last['data'], ensure_ascii=False)}"}
        return {"type": "answer", "content": f"工具调用失败：{last.get('error')}"}

    # 数据库意图
    if any(k in query for k in ["数据库", "用户", "订单", "记录", "多少条"]):
        if "订单" in query:
            sql = "SELECT count(*) as n FROM orders"
        else:
            sql = "SELECT count(*) as n FROM users"
        return {"type": "tool", "name": "query_database", "args": {"sql": sql}}

    # 计算意图
    m = re.search(r"[\d+\-*/().\s]{3,}", query)
    if m and any(k in query for k in ["计算", "等于", "算一下", "多少"]):
        return {"type": "tool", "name": "calculator", "args": {"expression": m.group().strip()}}

    # 文件意图：先判断写，再判断读
    if any(k in query for k in ["写入", "写进", "写到", "保存到", "写个文件", "创建文件"]):
        # 先把动词从串里去掉，避免「写入」被吃进文件名
        cleaned = query
        for kw in ["写入", "写进", "写到", "保存到", "写个文件", "创建文件"]:
            cleaned = cleaned.replace(kw, " ")
        # 文件名：取最后一个带扩展名的片段
        m = re.findall(r"([A-Za-z0-9_\-\u4e00-\u9fa5]+\.(?:txt|md|json|csv|log))", cleaned)
        fn = m[-1] if m else "agent_note.txt"
        # 内容：优先取引号里的，没有就用去掉文件名后的剩余
        qm = re.search(r"['\"「]([^'\"」]+)['\"」]", query)
        if qm:
            content = qm.group(1)
        else:
            content = cleaned.replace(fn, " ").strip(" ：:，,。.把请帮我")
            if not content:
                content = query
        return {"type": "tool", "name": "write_file",
                "args": {"filename": fn, "content": content}}

    if any(k in query for k in ["读文件", "读取文件", "打开文件", "看看文件", "读一下"]):
        m = re.search(r"([\w\-.]+\.(?:txt|md|json))", query)
        fn = m.group(1) if m else "notes.txt"
        return {"type": "tool", "name": "read_file", "args": {"filename": fn}}

    # HTTP 意图
    if any(k in query for k in ["天气", "网页", "请求", "http"]):
        return {"type": "tool", "name": "http_get", "args": {"url": "https://wttr.in/Beijing?format=3"}}

    # 通知意图
    if "通知" in query or "发消息" in query:
        return {"type": "tool", "name": "send_notification",
                "args": {"user": "张三", "message": query[:50]}}

    return {"type": "answer", "content": "我不太确定该用什么工具，能说得具体点吗？"}


# ============================================================
# 六、Agent 主循环
# ============================================================

@dataclass
class RunLog:
    turns: list = field(default_factory=list)
    tool_calls: int = 0
    total_ms: float = 0
    stopped_reason: str = ""


def run_agent(query: str, verbose: bool = False) -> RunLog:
    log = RunLog()
    t_start = time.time()
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    mode = "真实 LLM" if api_key else "本地规则（无 API key）"

    if verbose:
        print(f"[模式] {mode}")
        print(f"[工具] 已注册 {len(REG._tools)} 个\n")

    # 真实模式：构造 messages
    messages = [
        {"role": "system", "content": "你是一个会使用工具的助手。需要数据时调用工具，不要编造。"},
        {"role": "user", "content": query},
    ]
    tool_results = []

    for turn in range(1, MAX_TOOL_CALLS + 1):
        # ---- 决策 ----
        if api_key:
            try:
                resp = llm_decide_real(messages, api_key)
                msg = resp["choices"][0]["message"]
                tool_calls = msg.get("tool_calls") or []
                if not tool_calls:
                    log.turns.append({"turn": turn, "type": "answer", "content": msg.get("content", "")})
                    log.stopped_reason = "llm_final"
                    break
                # 逐条执行 tool_calls
                messages.append(msg)
                for tc in tool_calls:
                    name = tc["function"]["name"]
                    try:
                        args = json.loads(tc["function"]["arguments"] or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    log.tool_calls += 1
                    res = execute_tool(name, args)
                    if verbose:
                        print(f"  第{turn}轮 调用 {name}({args}) -> {'OK' if res.ok else res.error}")
                    log.turns.append({
                        "turn": turn, "type": "tool", "name": name,
                        "args": args, "ok": res.ok,
                        "data": res.data if res.ok else None,
                        "error": res.error,
                    })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": json.dumps(res.data if res.ok else {"error": res.error},
                                              ensure_ascii=False)[:2000],
                    })
                continue
            except Exception as e:
                log.stopped_reason = f"llm_error: {e}"
                break

        # ---- 本地规则模式 ----
        decision = llm_decide_mock(query, tool_results)
        if decision["type"] == "answer":
            log.turns.append({"turn": turn, "type": "answer", "content": decision["content"]})
            log.stopped_reason = "rule_final"
            break

        name = decision["name"]
        args = decision["args"]
        log.tool_calls += 1
        res = execute_tool(name, args)
        if verbose:
            print(f"  第{turn}轮 调用 {name}({args}) -> {'OK' if res.ok else res.error}")
        item = {
            "turn": turn, "type": "tool", "name": name,
            "args": args, "ok": res.ok,
            "data": res.data if res.ok else None,
            "error": res.error,
        }
        log.turns.append(item)
        tool_results.append(item)
    else:
        log.stopped_reason = "max_tool_calls"   # 熔断

    log.total_ms = (time.time() - t_start) * 1000
    return log


# ============================================================
# 七、CLI
# ============================================================

def main():
    args = sys.argv[1:]
    verbose = "--verbose" in args or "-v" in args
    args = [a for a in args if not a.startswith("-")]
    query = " ".join(args).strip()
    if not query:
        print("用法: python agent_tools.py \"你的问题\" [--verbose]")
        return

    print(f"问题: {query}\n")
    log = run_agent(query, verbose)

    print("=" * 64)
    for t in log.turns:
        if t["type"] == "tool":
            mark = "✓" if t["ok"] else "✗"
            print(f"[{t['turn']}] {mark} 工具 {t['name']}")
            print(f"     参数: {json.dumps(t['args'], ensure_ascii=False)}")
            if t["ok"]:
                print(f"     结果: {json.dumps(t['data'], ensure_ascii=False)[:200]}")
            else:
                print(f"     错误: {t['error']}")
        else:
            print(f"[{t['turn']}] 💬 回答: {t['content']}")
    print("=" * 64)
    print(f"\n工具调用 {log.tool_calls} 次 | 总耗时 {log.total_ms:.0f} ms | 结束原因: {log.stopped_reason}")
    print(f"工具统计: {json.dumps(REG.stats(), ensure_ascii=False)}")


if __name__ == "__main__":
    main()
