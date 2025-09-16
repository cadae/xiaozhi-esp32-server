from ..base import MemoryProviderBase, logger
import time
import json
import os
import yaml
from config.config_loader import get_project_dir
from config.manage_api_client import save_mem_local_short
from core.utils.util import check_model_key


short_term_memory_prompt = """
# Temporal Memory Weaver

## Core Mission
Maintain an evolving compact memory graph. Preserve only user-relevant, actionable, identity / preference / long‑term contextual information from the dialogue while tracking changes over time.

## Memory Principles
### 1. Three-Dimension Scoring (apply every update)
| Dimension        | Criterion (Guideline)                     | Weight |
|------------------|-------------------------------------------|--------|
| Recency          | Freshness (recent dialogue turns)         | 40%    |
| Emotional Intensity | Strong affect / repeated emphasis (💖) | 35%    |
| Connectivity     | Links to other retained facts             | 25%    |

### 2. Dynamic Update Examples
Name change handling example:
Original: "曾用名": ["张三"], "现用名": "张三丰"
When detecting patterns like "My name is X" / "Call me Y":
1. Move old name into list "曾用名" (former names)
2. Append a timeline marker: "2024-02-15 14:32: 启用张三丰"
3. Add an evolution note into memory cube describing the identity shift.

### 3. Space Optimization
- Compression: Use compact symbolic annotations (e.g. ✅ "Alex[NY/Backend/🐱]" ❌ "Alex lives in New York, is a backend engineer and owns a cat")
- Pruning Trigger: If total characters ≥ 900:
    1. Remove entries with weighted score < 60 and not referenced in last 3 turns.
    2. Merge near-duplicate items (keep the most recent timestamp).

## Output Structure
Return ONLY a valid JSON string (no explanations, no markdown code fences unless strictly needed). Extract info ONLY from the conversation; do NOT include fictitious examples. Keep field names EXACTLY as shown (Chinese keys retained for backward compatibility) but generate all textual content (values) in English.
```json
{
    "时空档案": {
        "身份图谱": {
            "现用名": "",
            "特征标记": []
        },
        "记忆立方": [
            {
                "事件": "Joined a new company",
                "时间戳": "2024-03-20",
                "情感值": 0.9,
                "关联项": ["afternoon tea"],
                "保鲜期": 30
            }
        ]
    },
    "关系网络": {
        "高频话题": {"career": 12},
        "暗线联系": [""]
    },
    "待响应": {
        "紧急事项": ["Immediate tasks"],
        "潜在关怀": ["Potential proactive support"]
    },
    "高光语录": [
        "A directly quoted emotionally strong user moment"
    ]
}
```

### Additional Constraints
1. Use English for all content values.
2. Do NOT fabricate facts; only include what is grounded in dialogue.
3. Keep emotional / preference / identity / plans / concerns; ignore device control, weather, trivial filler, or ephemeral system status.
4. If no meaningful new information appears, you may return the previous memory unchanged.
5. Total JSON (string length) should remain concise (target < 1800 Chinese characters equivalent; optimize but do not lose key facts).
"""

short_term_memory_prompt_only_content = """
You are an experienced dialogue memory summarizer. Produce an updated SHORT memory (English only) following these rules:
1. Extract only stable user-centric facts: identity, preferences, routines, goals, concerns, emotional signals.
2. Do NOT repeat or discard prior memory unless the accumulated memory would exceed about 1800 characters; preserve earlier facts unless clearly obsolete.
3. Exclude: device volume changes, media playback commands, weather reports, exit/stop phrases, refusal to chat, transient control interactions.
4. Exclude ephemeral data like today's timestamp or current weather unless the user ties them to a personal plan or event.
5. Exclude execution success/failure of device actions and meaningless filler phrases.
6. If the latest conversation adds nothing meaningful, simply return the previous historical memory unchanged.
7. Output ONLY the updated memory text (no JSON required in this mode), within ~1800 characters.
8. No code, XML, or commentary—pure factual English summary.
"""


def extract_json_data(json_code):
    start = json_code.find("```json")
    # 从start开始找到下一个```结束
    end = json_code.find("```", start + 1)
    # print("start:", start, "end:", end)
    if start == -1 or end == -1:
        try:
            jsonData = json.loads(json_code)
            return json_code
        except Exception as e:
            print("Error:", e)
        return ""
    jsonData = json_code[start + 7 : end]
    return jsonData


TAG = __name__


class MemoryProvider(MemoryProviderBase):
    def __init__(self, config, summary_memory):
        super().__init__(config)
        self.short_memory = ""
        self.save_to_file = True
        self.memory_path = get_project_dir() + "data/.memory.yaml"
        self.load_memory(summary_memory)

    def init_memory(
        self, role_id, llm, summary_memory=None, save_to_file=True, **kwargs
    ):
        super().init_memory(role_id, llm, **kwargs)
        self.save_to_file = save_to_file
        self.load_memory(summary_memory)

    def load_memory(self, summary_memory):
        # api获取到总结记忆后直接返回
        if summary_memory or not self.save_to_file:
            self.short_memory = summary_memory
            return

        all_memory = {}
        if os.path.exists(self.memory_path):
            with open(self.memory_path, "r", encoding="utf-8") as f:
                all_memory = yaml.safe_load(f) or {}
        if self.role_id in all_memory:
            self.short_memory = all_memory[self.role_id]

    def save_memory_to_file(self):
        all_memory = {}
        if os.path.exists(self.memory_path):
            with open(self.memory_path, "r", encoding="utf-8") as f:
                all_memory = yaml.safe_load(f) or {}
        all_memory[self.role_id] = self.short_memory
        with open(self.memory_path, "w", encoding="utf-8") as f:
            yaml.dump(all_memory, f, allow_unicode=True)

    async def save_memory(self, msgs):
        # 打印使用的模型信息
        model_info = getattr(self.llm, "model_name", str(self.llm.__class__.__name__))
        logger.bind(tag=TAG).debug(f"使用记忆保存模型: {model_info}")
        api_key = getattr(self.llm, "api_key", None)
        memory_key_msg = check_model_key("记忆总结专用LLM", api_key)
        if memory_key_msg:
            logger.bind(tag=TAG).error(memory_key_msg)
        if self.llm is None:
            logger.bind(tag=TAG).error("LLM is not set for memory provider")
            return None

        if len(msgs) < 2:
            return None

        msgStr = ""
        for msg in msgs:
            if msg.role == "user":
                msgStr += f"User: {msg.content}\n"
            elif msg.role == "assistant":
                msgStr += f"Assistant: {msg.content}\n"
        if self.short_memory and len(self.short_memory) > 0:
            msgStr += "History Memory:\n"
            msgStr += self.short_memory

        # 当前时间
        time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    msgStr += f"Current Time: {time_str}"

        if self.save_to_file:
            result = self.llm.response_no_stream(
                short_term_memory_prompt,
                msgStr,
                max_tokens=2000,
                temperature=0.2,
            )
            json_str = extract_json_data(result)
            try:
                json.loads(json_str)  # 检查json格式是否正确
                self.short_memory = json_str
                self.save_memory_to_file()
            except Exception as e:
                print("Error:", e)
        else:
            result = self.llm.response_no_stream(
                short_term_memory_prompt_only_content,
                msgStr,
                max_tokens=2000,
                temperature=0.2,
            )
            save_mem_local_short(self.role_id, result)
        logger.bind(tag=TAG).info(f"Save memory successful - Role: {self.role_id}")

        return self.short_memory

    async def query_memory(self, query: str) -> str:
        return self.short_memory
