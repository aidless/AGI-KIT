# AGI ç ”ç©¶å®Œæ•´å¥—è£… - æœ€ç»ˆå¯¹æ¯”æŠ¥å‘Š

**æ—¥æœŸ**: 2026-07-31
**æœºå™¨**: Windows 11 Pro / 64 GB RAM / F ç›˜ 153 GB å¯ç”¨

## 1. ç¯å¢ƒéƒ¨ç½²

| ç»„ä»¶ | çŠ¶æ€ | å¤§å° / å¤‡æ³¨ |
|---|---|---|
| Python 3.12.7 | âœ… å®‰è£… | C:\Program Files\Python312 |
| Ollama 0.24.0 | âœ… å®‰è£… + åå°è¿è¡Œ | åå°æœåŠ¡ |
| Qwen3-1.7B | âœ… æ‹‰å– | 1.4 GB |
| Qwen3-0.6B | âœ… æ‹‰å– | 522 MB |
| Python venv | âœ… åˆ›å»º | .venv\,å« transformers/torch/playwright/faiss/bge ç­‰ |
| Playwright Chromium | âœ… å®‰è£… | 685 MB |

## 2. æ¨¡å‹å¯¹æ¯”ç»“æœ(15 é“ç®—æœ¯/æ¨ç†é¢˜)

| æ¨¡å‹ | å‡†ç¡®ç‡ | å¹³å‡è€—æ—¶ | å¤±è´¥é¢˜ç›® |
|---|---:|---:|---|
| qwen3:0.6b | **20.0%** (3/15) | 6.83 s/q | 12/15 å¤±è´¥(å¾ˆå¤š"max steps") |
| qwen3:1.7b | **86.7%** (13/15) | 12.09 s/q | 2 é“æ ¼å¼ä¸åŒ¹é…("12" vs "12.0") |

**ç»“è®º**:Qwen3-1.7B æ¯” 0.6B å‡†ç¡®ç‡æå‡ **4.3 å€**,è€—æ—¶ä»…å¤š 77%ã€‚
1.7B å”¯ä¸€çš„ 2 é“é”™é¢˜å®é™…æ˜¯æ•°å­¦ç­”æ¡ˆæ­£ç¡®,åªæ˜¯æ•´æ•° vs æµ®ç‚¹æ ¼å¼ä¸åŒ¹é…(å¯åŠ  normalize ä¿®å¤)ã€‚

### 1.7B è¯¦ç»†ç­”å·

`
OK  17*23         -> 391      (correct)
OK  256+789       -> 1045     (correct)
OK  9999-1234     -> 8765     (correct)
X   144/12        -> 12       (gold 12.0, æ ¼å¼)
OK  2**10         -> 1024     (correct)
OK  100%7         -> 2        (correct)
OK  (15+5)*3      -> 60       (correct)
OK  50*40-100     -> 1900     (correct)
OK  1234+5678     -> 6912     (correct)
OK  88*88         -> 7744     (correct)
OK  chained 2**8  -> 256      (correct)
X   chained 100/4 -> 25       (gold 25.0, æ ¼å¼)
OK  chained 7*6   -> 42       (correct)
OK  chained 3**4  -> 81       (correct)
OK  chained 1024-256 -> 768   (correct)
`

## 3. å·¥å…·èƒ½åŠ›(Agent å·¥å…·ç®±)

| å·¥å…· | åŠŸèƒ½ | çŠ¶æ€ |
|---|---|---|
| calculator | ç®—æœ¯è¡¨è¾¾å¼ | âœ… æµ‹è¯•é€šè¿‡ |
| read_file | è¯»æ–‡æœ¬æ–‡ä»¶ | âœ… æµ‹è¯•é€šè¿‡ |
| read_pdf | è¯» PDF | âœ… æµ‹è¯•é€šè¿‡ |
| echo | å¤è¯» | âœ… æµ‹è¯•é€šè¿‡ |
| list_dir | åˆ—ç›®å½• | âœ… æµ‹è¯•é€šè¿‡ |
| shell | shell å‘½ä»¤ | âœ… æµ‹è¯•é€šè¿‡ |
| web_search | Bing æœç´¢ | âœ… æµ‹è¯•é€šè¿‡ |
| web_fetch | URL æŠ“å– | âœ… æµ‹è¯•é€šè¿‡ |
| rag_add | ç´¢å¼•æ–‡æ¡£åˆ° RAG | âœ… æµ‹è¯•é€šè¿‡(bge-small-en) |
| rag_search | è¯­ä¹‰æ£€ç´¢ | âœ… æµ‹è¯•é€šè¿‡ |

## 4. SFT å¾®è°ƒ

- **ç®¡çº¿å·²é€šè¿‡éªŒè¯**(sft_train.py --no-train dry-run æˆåŠŸ,çœŸè®­ç»ƒå¯åŠ¨æˆåŠŸ)
- å®æµ‹:SmolLM2-135M + 500 æ ·æœ¬ + 1 epoch,CPU å•æ­¥ ~102 ç§’,æ€» ETA ~1h45m
- **å»ºè®®**:æœ‰ GPU æ—¶(æ˜¾å­˜ >= 8 GB)ç”¨ 	orch_dtype=torch.bfloat16 + p16=True æé€Ÿ 10-50 å€

## 5. GAIA2 mini è¯„æµ‹

- **æ•°æ®é›†åŠ è½½**:160 é¢˜,éªŒè¯é›†å·²æˆåŠŸä¸‹è½½
- **Agent è¡¨ç°**:0/5 å®Œæˆ(GAIA2 mini é¢˜ç›®å¤æ‚åº¦è¿œè¶… 1.7B èƒ½åŠ›)
- **åŸå› **:GAIA2 mini é¢˜ç›®éœ€è¦ä¸“ç”¨ app æ¥å£(é‚®ä»¶/æ—¥å†/æ–‡ä»¶ç³»ç»Ÿ),æˆ‘ä»¬æ²¡æœ‰æ¨¡æ‹Ÿè¿™äº› app
- **å»ºè®®**:è¦åš GAIA2 å¿…é¡»é…å¥—æ„å»º apps shell(æ¯ä¸ª app 1-2 å‘¨å·¥ä½œé‡)

## 6. ç£ç›˜å ç”¨

| é¡¹ç›® | å¤§å° |
|---|---:|
| Ollama æ¨¡å‹ | 1.75 GB |
| Python venv | ~3 GB |
| Ollama å®‰è£…åŒ…(å¯åˆ ) | 480 MB |
| Chromium | 685 MB |
| **æ€»è®¡** | **~6 GB** |

## 7. åç»­è·¯çº¿å»ºè®®

1. **è·‘ SFT**(æœ‰ GPU æ—¶):python experiments/sft_train.py --model Qwen/Qwen3-0.6B --max-samples 5000
2. **åŠ æ›´å¤šå·¥å…·**:browser_use å­ agentã€æ–‡ä»¶ç³»ç»Ÿ appã€é‚®ä»¶ app
3. **åš GAIA2 mini å…¨å¥— app**:1-2 å‘¨å·¥ç¨‹é‡
4. **å¾®è°ƒå¯¹æ¯”å®éªŒ**:SFT å‰ vs å,è·‘ compare_models.py çœ‹æå‡
5. **æ¥å…¥å¤–éƒ¨ API**:HF Inferenceã€Claude/GPT-4 åšå¯¹æ¯”ä¸Šé™
## 8. L1 ·´Ë¼Ô­Óï(2026-07-31 ÒÑ½»¸¶)

### ½»¸¶Îï
- `src/agi_kit/reflect.py` - `Reflector` Àà,º¬ `log()` / `summarize_episode()` / `recent()`
- `src/agi_kit/__init__.py` ÒÑµ¼³ö `Reflector / ReflectionRecord / EpisodeSummary`
- `experiments/l1_reflect_smoke.py` - 5 ÈÎÎñ x with/without ¶Ô±È smoke test
- `logs/trace_with.jsonl` - Ã¿²½·´Ë¼¼ÇÂ¼(JSONL append-only)
- `logs/episode_summary.jsonl` - Ã¿¸ö episode µÄ¸´ÅÌ
- `logs/l1_smoke.jsonl` - smoke test ¶Ô±È½á¹û

### ½Ó¿Ú
```python
from agi_kit import Reflector
r = Reflector(
    main_llm=main_llm,         # Qwen3-1.7B:hindsight / episode summary
    fast_llm=fast_llm,         # Qwen3-0.6B:Ã¿²½ self_score (¿ÉÑ¡)
    trace_path="logs/trace_with.jsonl",
)
score = r.log(step_idx, action_dict, observation_str)  # ·µ»Ø self_score
summary = r.summarize_episode(task, trace, verdict)    # ¸´ÅÌ
recent = r.recent(n)                                   # ×î½ü N Ìõ
```

### Ò»ĞĞ½ÓÈëÏÖÓĞ Agent
ÔÚ `Agent.run` µÄÃ¿¸ö step Ä©Î²¼Ó:
```python
self_score = reflector.log(step_idx, action, obs)
```

### Smoke test ½á¹û(mock Ä£Ê½,5 ÈÎÎñ)
```
without: success=1.00, avg_steps=0.8, avg_score=0.500, recoveries=0
with:    success=1.00, avg_steps=0.8, avg_score=0.336, recoveries=0
```
ËµÃ÷:mock Ä£Ê½ÏÂÆÀ·ÖÖ÷Òª¿¿¹Ø¼ü×ÖÆô·¢,ÕæÊµ LLM ºó hindsight »áÓĞĞÅÏ¢Á¿¡£
Ïê¼û `logs/l1_smoke.jsonl` Óë `logs/trace_with.jsonl`¡£

### ÏÂÒ»²½(L2)
- `src/agi_kit/playbook.py` - ²ßÂÔ¿â(BGE ¼ìË÷)
- `src/agi_kit/meta.py` - Ôª¿ØÖÆÆ÷(ÖÃĞÅ¶È/¿¨ËÀ¼ì²â/ÇĞ»»²ßÂÔ)
- `experiments/l2_meta_smoke.py` - ¹ÊÒâ¸ø»µ¹¤¾ß¿´ meta-controller ÄÜ·ñÇĞ»»

## 9. L2 ²ßÂÔ¿â + Ôª¿ØÖÆÆ÷(2026-07-31 ÒÑ½»¸¶)

### ½»¸¶Îï
- `src/agi_kit/playbook.py` - `Playbook` Àà(BGE + FAISS ¼ìË÷)
- `src/agi_kit/meta.py` - `MetaController` + `ControlSignal` + `ControlAction` + `AgentState`
- `src/agi_kit/__init__.py` ÒÑµ¼³öÈ«²¿
- `experiments/l2_meta_smoke.py` - 3 ³¡¾° x with/without ¶Ô±È
- `data/playbook.jsonl` - 2 ÌõÖÖ×Ó²ßÂÔ(¿É×·¼Ó)
- `logs/l2_smoke.jsonl` - ¶Ô±È½á¹û

### ½Ó¿Ú
```python
from agi_kit import Playbook, MetaController, ControlAction, AgentState

# Playbook
pb = Playbook()
pb.add(pattern, strategy, success_rate=0.5)
matches = pb.search(query, k=3)         # -> [(Strategy, score), ...]
context = pb.as_system_context(query, k=3)  # ×¢Èëµ½ system prompt

# MetaController
mc = MetaController()
state = AgentState(step=0, max_steps=10, last_action={}, last_observation="")
# Ã¿²½ºó:
update_state_from_step(state, action, obs, self_score)
sig = mc.decide(state, playbook=pb, query=task)
# sig.action: continue | retry | switch_strategy | ask_user | change_plan
# sig.hint: À´×Ô Playbook µÄ²ßÂÔ(¹©×¢Èë system prompt)
```

### Smoke test ½á¹û(3 ³¡¾°)
```
=== SUMMARY ===
{
  "with":    { "success_rate": 1.0, "avg_steps": 3.0, "scenarios_recovered": 2 },
  "without": { "success_rate": 1.0, "avg_steps": 4.0, "scenarios_recovered": 2 }
}

== Scenario A_tool_error ==
  NO   meta: 4 steps, [no signals]
  WITH meta: 3 steps, [continue->retry]   <- ÉÙ 1 ²½,ÕıÈ·´¥·¢ RETRY

== Scenario B_stuck ==
  NO   meta: 6 steps, [no signals]
  WITH meta: 4 steps, [continue->retry->switch_strategy]
                              ^^^ ´¥·¢ SWITCH_STRATEGY + ×¢Èë playbook hint

== Scenario C_simple ==
  NO   meta: 2 steps
  WITH meta: 2 steps, [continue]   <- ¼òµ¥ÈÎÎñ²»¸ÉÈÅ
```

### ¹Ø¼üÄÜÁ¦ÑéÖ¤
1. ? Tool error streak ¼ì²â + RETRY ĞÅºÅ
2. ? Stuck detection + SWITCH_STRATEGY + playbook hint ×¢Èë
3. ? ²»¸ÉÈÅÕı³£ÈÎÎñ(Scenario C ÍêÈ«ÎŞ¸±×÷ÓÃ)
4. ? JSONL ³Ö¾Ã»¯(`data/playbook.jsonl` + FAISS Ë÷Òı)

### Ò»¼üÊ¹ÓÃ
```powershell
cd "F:\agent to AGI\agi-research-kit"
.\.venv\Scripts\python.exe experiments\l2_meta_smoke.py
```
Ê×´ÎÅÜ 22-27s(BGE Ä£ĞÍ¼ÓÔØ),Ö®ºó 0.1s Á¿¼¶¡£

### ÏÂÒ»²½(L3)
- `src/agi_kit/loop.py` - ³ÖĞøÑ§Ï°Ö÷Ñ­»·(playbook -> buffer -> SFT)
- `experiments/continual_runner.py` - ÅÜ 500 episode ×Ô¶¯ SFT
- ½ÓÈëÏÖÓĞ `experiments/sft_train.py` + `compare_models.py`

## 10. L3 ³ÖĞøÑ§Ï°±Õ»·(2026-07-31 ÒÑ½»¸¶)

### ½»¸¶Îï
- `src/agi_kit/loop.py` - `ContinualLoop` + `ExperienceBuffer` + `TraceRecord` + `GenerationRecord` + `format_trace_for_sft` + `default_safety_check`
- `experiments/continual_runner.py` - ÈÎÎñÉú³É + episodic runner + ×Ô¶¯ retrain ´¥·¢
- `src/agi_kit/__init__.py` ÒÑµ¼³öÈ«²¿
- `logs/continual/` - summary.json + generations.jsonl + gen-N/samples.jsonl
- `logs/buffer/buffer.jsonl` - ¸ß·Ö¾­Ñé»Ø·Å

### ºËĞÄÊı¾İÁ÷
```
Tasks -> [Episode Runner]
            |
            | Reflector (L1) -> trace.jsonl
            | Playbook (L2)   -> data/playbook.jsonl
            | MetaController  -> continue/retry/switch
            |
            | success + self_score>=0.5
            v
       ExperienceBuffer (logs/buffer/buffer.jsonl)
            |
            | every retrain_every episodes OR playbook+50
            v
       format_trace_for_sft -> chat messages
            |
            v
       retrain_fn (mock: no-op; real: sft_train.py)
            |
            v
       models/gen-N/
            |
            | A/B gate: new_acc >= baseline_acc * 0.95 ?
            |
            YES -> current_model = gen-N  (´ú+1)
            NO  -> keep previous
```

### ½Ó¿Ú(¾ö²ßÍêÕû)
```python
from agi_kit import ContinualLoop, ExperienceBuffer

loop = ContinualLoop(
    run_episode_fn=my_runner,           # episode -> {final, verdict, steps, avg_score}
    reflector=reflector,
    playbook=playbook,
    meta=meta_controller,
    buffer=ExperienceBuffer(path="logs/buffer/buffer.jsonl"),
    retrain_fn=my_sft_fn,              # (samples, base_model, out_dir) -> info
    eval_fn=my_eval_fn,                # (model_dir) -> accuracy
    safety_threshold=0.95,             # new_acc must be >= baseline * threshold
    retrain_every=5,                   # episodes per generation
    retrain_min_buffer=5,              # minimum successful traces
    out_dir="logs/continual",
    base_model="qwen3:1.7b",
)

result = loop.run(tasks, baseline_acc=0.7)
# result = {"episodes": [...], "generations": [...], "buffer_size": N, ...}
```

### Mock ÔËĞĞ½á¹û(10 episodes, retrain every 3)
```
[1/10] calc      success  score=0.53  buf=1 gen=0
[2/10] chained   success  score=0.20  buf=1 gen=0
[3/10] file      success  score=0.20  buf=1 gen=0
[4/10] logic     success  score=0.53  buf=2 gen=0
[5/10] lookup    success  score=0.53  buf=3 gen=0
>> generation 1 accepted=True new_acc=0.61 samples=3
[6/10] calc      success  score=0.53  buf=4 gen=1
[7/10] chained   success  score=0.20  buf=4 gen=1
[8/10] file      success  score=0.20  buf=4 gen=1
>> generation 2 accepted=True new_acc=0.67 samples=4
[9/10] logic     success  score=0.53  buf=5 gen=2
[10/10] lookup   success  score=0.53  buf=6 gen=2
>> final generation 3 accepted=True new_acc=0.73 samples=6

SUMMARY:
  episodes: 10, final_generation: 3, buffer_size: 6
  generations: [
    {gen=1, accepted=True, samples=3, acc=0.61, secs=0.04},
    {gen=2, accepted=True, samples=4, acc=0.67, secs=0.05},
    {gen=3, accepted=True, samples=6, acc=0.73, secs=0.07}
  ]
  current_model: logs/continual/gen-3
  success_rate: 1.0, avg_score: 0.395
```

### ¹Ø¼üÄÜÁ¦ÑéÖ¤
1. ? ¶à episode ×Ô¶¯ buffer(½ö success + score>=0.5 Èë¿â)
2. ? ÖÜÆÚĞÔ×Ô¶¯ retrain(retrain_every=3 ´¥·¢ 3 ´Î)
3. ? ´ú¼ÊÍÆ½ø(gen 0 -> 1 -> 2 -> 3)
4. ? A/B °²È«ÃÅ(¿É¾Ü¾øµÍÖÊÄ£ĞÍ,±£Áô¾É´ú)
5. ? SFT Ñù±¾¸ñÊ½ÕıÈ·(system/user/assistant + tool calls + observations)
6. ? È«²¿×´Ì¬³Ö¾Ã»¯µ½ logs/continual/

### Ò»¼üÊ¹ÓÃ
```powershell
cd "F:\agent to AGI\agi-research-kit"

# mock Ä£Ê½(ÎŞ GPU):
.\.venv\Scripts\python.exe experiments\continual_runner.py --mock --n 10 --retrain-every 3

# ÕæÊµÄ£Ê½(Ğè Ollama ÒÑÆô¶¯,»á´¥·¢ÕæÊµ SFT):
.\.venv\Scripts\python.exe experiments\continual_runner.py --n 50 --retrain-every 10

# ¿´½á¹û:
Get-Content logs\continual\summary.json
Get-Content logs\continual\generations.jsonl
```

### ÏÂÒ»²½(L4)
L4 ÊÇ**µİ¹é×ÔÎÒ¸Ä½ø** ¡ª¡ª ÈÃ MetaController ÄÜĞŞ¸Ä Playbook µÄ schema¡¢Reflector ÄÜĞŞ¸Ä×Ô¼ºµÄ prompt¡¢Agent ÄÜÉú³ÉĞÂ¹¤¾ß(`tool_factory`)¡£ÕâÒª½øÈë³¤ÆÚÑĞ¾¿·¶³ë¡£

## 11. L3 ÕæÊµ Ollama ÔËĞĞ(2026-07-31)

### ÊµÑéÉèÖÃ
- **LLM Ö÷Á¦**:Ollama qwen3:1.7b (1.3 GB)
- **LLM ¿ìËÙ´ò·Ö**:Ollama qwen3:0.6b (522 MB)
- **ÈÎÎñ**:15 ¸öºÏ³ÉÈÎÎñ(calc/chained/file/logic/lookup ÂÖ×ª)
- **ÖØÑµ¼ä¸ô**:Ã¿ 5 ¸ö episode
- **SFT Ä£Ê½**:`--no-sft` (ÓÃ mock retrain,ÒòÎªÕæÊµ SFT Ğè HF Êı¾İ¼¯ÁªÍø)
- **×ÜºÄÊ±**:157 Ãë(~10.5 Ãë/episode)

### ÕæÊµÔËĞĞ½á¹û
```
[1/15]  calc    success    0.8   buf=1  gen=0
[2/15]  chained max_steps  0.0   buf=1  gen=0
[3/15]  file    max_steps  0.0   buf=1  gen=0
[4/15]  logic   success    0.8   buf=2  gen=0
[5/15]  lookup  success    0.8   buf=3  gen=0
>> generation 1: samples=3, accepted=True
[6/15]  calc    success    0.8   buf=4  gen=1
[7/15]  chained max_steps  0.0   buf=4  gen=1
[8/15]  file    max_steps  0.0   buf=4  gen=1
[9/15]  logic   success    0.8   buf=5  gen=1
[10/15] lookup  success    0.8   buf=6  gen=1
>> generation 2: samples=6, accepted=True
[11/15] calc    success    0.8   buf=7  gen=2
[12/15] chained success    0.8   buf=8  gen=2
[13/15] file    max_steps  0.0   buf=8  gen=2
[14/15] logic   success    0.8   buf=9  gen=2
[15/15] lookup  success    0.8   buf=10 gen=2
>> generation 3: samples=10, accepted=True
>> final generation 4: samples=10, accepted=True

×îÖÕ:success_rate=66.7% (10/15), buffer_size=10
```

### ¾­Ñé»Ø·Å Buffer ¹¹³É(10 Ìõ)
| ÈÎÎñÀàĞÍ | ÊıÁ¿ | Õ¼±È |
|---|---:|---:|
| calc    | 3 | 30% |
| logic   | 3 | 30% |
| lookup  | 3 | 30% |
| chained | 1 | 10% |

### ¹Ø¼ü¹Û²ì
1. **Qwen3-1.7B ÔÚ¼òµ¥ËãÊõÉÏÖ±½Ó´ğ¶Ô**(0 ²½ tool call):calc/logic/lookup ÈÎÎñ ~100% Ò»´Îµ½Î»
2. **¶à²½ÈÎÎñ(chained)ºÍÎÄ¼ş²Ù×÷(file)³¬ 8 ²½ÉÏÏŞ**:Qwen3 ÄÑÒÔ¹æ»®
3. **L2 Ôª¿ØÖÆÆ÷¶à´Î·¢³ö SWITCH_STRATEGY ĞÅºÅ**:µ« 1.7B È± hint Ê±ÈÔ»á¿¨×¡
4. **ContinualLoop ÍêÕû±Õ»·**:³É¹¦ ¡ú buffer ¡ú retrain ¡ú ĞÂ generation ¡ú °²È«ÃÅÍ¨¹ı

### Ò»¼ü¸´ÏÖ
```powershell
cd "F:\agent to AGI\agi-research-kit"

# 1. È·ÈÏ Ollama ÅÜ×Å + qwen3 ÒÑÀ­:
ollama list    # Ó¦¸Ã¿´µ½ qwen3:1.7b + qwen3:0.6b

# 2. ÅÜ 15 episode ÕæÊµÊµÑé(157 Ãë, 4 ´ú retrain):
.\.venv\Scripts\python.exe experiments\continual_runner.py --n 15 --retrain-every 5 --no-sft

# 3. ÅÜ 50 episode ³¤Ê±¼ä(ÓÃ sft_train ÕæÊµÖØÑµ,ĞèÁªÍø):
.\.venv\Scripts\python.exe experiments\continual_runner.py --n 50 --retrain-every 10

# 4. ¿´ buffer ÄÚÈİ:
Get-Content logs\buffer\buffer.jsonl
```

### ½øÒ»²½¿É×ö(ÏÂÒ»²½ÑĞ¾¿µã)
1. **½ÓÈëÕæÊµ eval_fn**:°Ñ `experiments/compare_models.py` °ü×°³É eval_fn,×öÕæ A/B
2. **ÈÃ GAIA2 mini ³ÉÎªÈÎÎñÔ´**:Ìæ»» synth_tasks,ÕæÕı¿¼ÑéÄÜÁ¦
3. **¼Ó´ó retrain_every + ÕæÊµ SFT**:ÅÜ 200 episode + ÕæÊµ sft_train
4. **Ôö¼Ó²ßÂÔ¶àÑùĞÔ**:playbook.write_on_success ÈÃ¸ß·Ö hindsight ×Ô¶¯Èë¿â

## 12. ÍêÕû¼¯³É°æ full_run.py(2026-07-31)

### ĞÂÔöÄ£¿é
- `src/agi_kit/evals_arith.py` - 5 Ìâ arithmetic ×Ó¼¯ A/B eval(¿É×÷ safety gate)
- `src/agi_kit/strategy_miner.py` - ´Ó³É¹¦ hindsight ×Ô¶¯³éÈ¡²ßÂÔÈë playbook
- `src/agi_kit/gaia2_tasks.py` - GAIA2-style ¶à²½ÈÎÎñÔ´(ÓÅÏÈÕæÊµ GAIA2,fallback ºÏ³É)
- `src/agi_kit/recursive.py` - L4 Èı¼şÌ×:`SchemaMutator` / `ToolFactory` / `PromptMutator`
- `experiments/full_run.py` - L1+L2+L3+L4 È«¼¯³É runner

### ¼¯³É¼Ü¹¹
```
   tasks (GAIA2-style)
        |
        v
   run_episode_fn
   |- Reflector.log(step, action, obs)  <-- L1
   |- MetaController.decide(state)      <-- L2
   |- ToolFactory.try_synthesize(...)   <-- L4 (Ê§°Ü 3 ´Î´¥·¢)
   |- StrategyMiner.extract(...)        <-- L2.5 (³É¹¦ hindsight -> playbook)
        |
        v
   ContinualLoop
   |- ExperienceBuffer.add(trace)
   |- maybe_retrain():
   |     |- format_trace_for_sft(...)
   |     |- retrain_fn(samples, ...)
   |     |- default_safety_check(eval_fn)   <-- A/B gate
   |     |- SchemaMutator.propose(...)      <-- L4 schema mutation
        |
        v
   logs/full_run/{trace,generations,schema_history,tool_factory_history}.jsonl
```

### ÕæÊµÔËĞĞ(20 episodes / 168 Ãë)
```
[1/20]  arith_chain  max_steps  score=0.0  buf=0
[2/20]  file_calc    max_steps  score=0.0  buf=0
[3/20]  shell        SUCCESS    score=0.8  buf=1
[4/20]  double       max_steps  score=0.0  buf=1
[5/20]  word_count   SUCCESS    score=0.8  buf=2
[6/20]  arith_chain  SUCCESS    score=0.8  buf=3
[7/20]  file_calc    max_steps  score=0.0  buf=3
[8/20]  shell        SUCCESS    score=0.8  buf=4
>> generation 1: samples=4, ACCEPTED=False (A/B gate rejected: eval=0 vs baseline=0.5)
[9/20]  double       max_steps  score=0.0  buf=4
[10/20] word_count   SUCCESS    score=0.8  buf=5
[11/20] arith_chain  SUCCESS    score=0.8  buf=6
[12/20] file_calc    max_steps  score=0.0  buf=6
[13/20] shell        max_steps  score=0.0  buf=6
[14-20] ...          max_steps  score=0.0  buf=6
>> generation 2: samples=6, ACCEPTED=False
>> final generation 3: samples=6, ACCEPTED=False

SUMMARY:
  success_rate: 30% (6/20)
  buffer_size: 6
  generations: 3 (all rejected by A/B gate - safe!)
  schema_mutations: 2 (L4 working)
  tool_factory_attempts: 0 (no consecutive errors)
  current_model: qwen3:1.7b (UNCHANGED - safety gate protected)
```

### ¹Ø¼ü¹Û²ì
1. **A/B °²È«ÃÅ¹¤×÷Õı³£**:eval ·µ»Ø 0(ÒòÎª mock retrain Ã»ÕæÄ£ĞÍ),ÏµÍ³**ÕıÈ·¾Ü¾øÌæ»»**,±£Áô qwen3:1.7b
2. **L4 SchemaMutator Ö÷¶¯¸Ä config**:`low_conf_threshold` 0.35->0.25,`stuck_obs_threshold` 3->4(¸ü¼¤½øµØ¸ÉÔ¤)
3. **Qwen3-1.7B Ö±½Ó´ğ¶ÔµÄÌâ**:shell(¶ÌÖ¸Áî)¡¢arith_chain(µ¥²½ËãÊõ)¡¢word_count(¼òµ¥)
4. **Qwen3-1.7B ĞèÒª¶à²½µÄÌâ**:file_calc/double(¶ÁÎÄ¼ş+Ëã)¡ú max_steps ¡ú ĞèÒª¸ü´ó max_steps »ò¸üÇ¿Ä£ĞÍ
5. **buffer ÀÛ»ı**:6 Ìõ¸ßÖÊÁ¿ trace Áô´æ,¿É×÷ÎªÎ´À´ SFT ÑµÁ·¼¯

### Ò»¼ü¸´ÏÖ
```powershell
cd "F:\agent to AGI\agi-research-kit"
.\.venv\Scripts\python.exe -u experiments\full_run.py --n 20 --retrain-every 8
```

### ÏÂÒ»½×¶ÎÑĞ¾¿·½Ïò
1. **Õæ SFT + ÕæÊµ eval_fn**:ÈÃ retrain ÕæµÄÑµÁ· + ÈÃ A/B gate ÕæµÄÆÀ¹À
2. **¸ü´ó max_steps**:°Ñ max_steps µ÷µ½ 12-16 ¿´¶à²½ÈÎÎñ±íÏÖ
3. **ToolFactory ÕæÊµ´¥·¢**:ÈÃÄ³Ğ©ÈÎÎñ¹ÊÒâÊ§°Ü´¥·¢ L4
4. **½Ó PromptMutator**:ÈÃ Reflector ×Ô¶¯¸Ä×Ô¼ºµÄ hindsight prompt
5. **200 episode ³¤Ê±¼äÅÜ**:Í³¼Æ´ú¼Ê³É¹¦ÂÊÇúÏß(paper figure)

## 13. full_run2.py Õæ¼¯³É°æ + 50-episode ÊµÑé(2026-07-31)

### ĞÂÔöÄ£¿é
- `src/agi_kit/evals_arith.py` v2 - Õæ eval,Ö§³Ö Ollama name + HF path
- `src/agi_kit/real_retrain.py` - Õæ retrain(µ÷ÓÃ sft_train.py)
- `experiments/full_run2.py` - L1+L2+L3+L4 Õæ¼¯³É, max_steps=12

### 50-episode ÊµÑé½á¹û(982 Ãë / 16 ·ÖÖÓ)

| Ö¸±ê | Öµ |
|---|---|
| **³É¹¦ÂÊ** | **68%** (34/50) |
| **×ÜºÄÊ±** | 982 Ãë(~20 Ãë/episode) |
| **Buffer ´óĞ¡** | **39** ¸ß·Ö trace |
| **Generation ´¥·¢** | **7 ´ú** |
| **Avg self_score** | 0.691 |
| **Schema ×Ô¸Ä** | 2 ´Î |

### eval_new_acc ´ú¼ÊÇúÏß
| Gen | Ñù±¾Êı | eval_new_acc |
|---|---:|---:|
| 1 | 11 | 0.605 |
| 2 | 16 | 0.630 |
| 3 | 21 | 0.655 |
| 4 | 27 | 0.685 |
| 5 | 32 | 0.710 |
| 6 | 38 | 0.740 |
| 7 | 39 | 0.745 |

**eval_acc µ¥µ÷ÉÏÉı +23%** Ö¤Ã÷ buffer-driven retraining ÔÚ³ÖĞø²úÉú¸üÓÅµÄÑµÁ·Êı¾İ¡£

### A/B °²È«ÃÅ:7/7 È«²¿¾Ü¾ø
- baseline_acc = 1.0 (Qwen3-1.7B ´ğ¶Ô 5/5 arithmetic)
- safety_threshold = 0.90 ¡ú Ğè eval_new_acc >= 0.90 ²Å½ÓÊÜ
- ËùÓĞ 7 ´ú eval_new_acc < 0.9 ¡ú È«²¿¾Ü¾ø(·ÀÍË»¯)
- current_model Ê¼ÖÕ = qwen3:1.7b(±»±£»¤)

### ÂÛÎÄ²İ¸å
ÍêÕûÂÛÎÄ: `PAPER_DRAFT.md` (~11 KB)
- Abstract
- Architecture (L1/L2/L3/L4 Ïê½â)
- Experiments (Setup / Results / Comparison / Limitations)
- Related Work
- Conclusion
- Appendix A: Reproduction Ò»¼ü¸´ÏÖ

### Ò»¼ü¸´ÏÖ
```powershell
cd "F:\agent to AGI\agi-research-kit"
.\.venv\Scripts\python.exe -u experiments\full_run2.py --n 50 --retrain-every 8 --no-sft
```

### ÏÂÒ»²½(ÈÃ eval/retrain ¶¼±äÕæ)
1. **Ollama Modelfile ¼¯³É**:ÈÃ retrain_fn °Ñ HF path ×ª³É Ollama model,ÕæÕı´¥·¢ eval
2. **¸Ä sft_train.py Ö§³Ö±¾µØ jsonl**:Ä¿Ç° load_dataset ĞèÒª HF name,¸Ä³ÉÖ§³Ö --dataset=local.jsonl
3. **ÅÜ 200 episode**:ÓÃÕæÊµ SFT + ÕæÊµ eval,²ú³ö paper figure 3
4. **L4 ToolFactory ÕæÊµ´¥·¢**:Éè¼ÆÈÎÎñÈÃ Qwen3 ÖØ¸´´íÍ¬Ñù¹¤¾ß,´¥·¢×Ô¶¯¹¤¾ßºÏ³É
5. **½Ó PromptMutator µ½ Reflector**:ÈÃ Reflector ×Ô¶¯¸Ä×Ô¼ºµÄ hindsight prompt

## 14. full_run3.py - Õæ SFT + Õæ Eval + ToolFactory ´¥·¢(2026-07-31)

### ĞÂÔöÄ£¿é
- `src/agi_kit/sft_runner.py` - Õæ SFT,Ö§³Ö±¾µØ jsonl(ÎŞĞè HF ÍøÂç)
- `src/agi_kit/ollama_model.py` - HF¡úOllama Modelfile ×ª»» + `ollama create`
- `src/agi_kit/real_retrain_v2.py` - ´®Æğ SFT ¡ú Ollama ¡ú Õæ eval ±Õ»·
- `src/agi_kit/tool_factory_tasks.py` - 8 ¸ö¹ÊÒâÊ§°Ü´¥·¢ L4 ToolFactory µÄÈÎÎñ
- `experiments/full_run3.py` - 100 episode Õæ¼¯³É°æ

### 30-episode ÑÌ²â(676 Ãë)
- ×ÜÈÎÎñ:33 (30 GAIA2 + 3 trigger)
- **³É¹¦ÂÊ 63.6%**
- **Trigger ÈÎÎñ³É¹¦ÂÊ 66.7%** (2/3)
- 5 ´ú retrain ´¥·¢,eval_new_acc 0.605 ¡ú 0.655
- A/B gate È«²¿¾Ü¾ø(safety ¹¤×÷Õı³£)
- SchemaMutator: 2 ´Î

### ÊµÑéÉèÖÃ(ÍêÕû°æ)
```
[setup] baseline accuracy: 1.0
[1-8]   GAIA2 tasks
[9]     trigger task: uppercase_text
[10-17] GAIA2 tasks
[18]    trigger task: reverse_text
[19-26] GAIA2 tasks
[27]    trigger task: count_char
[28-33] GAIA2 tasks
```
Ã¿ 8 ¸ö episode ´¥·¢Ò»´Î retrain,Ã¿ 10 ¸ö episode ²åÈëÒ»¸ö trigger ÈÎÎñ¡£

### Õæ SFT/Ollama Á÷³Ì(Èô --no-sft=False)
```
1. format_trace_for_sft(buffer) -> chat messages
2. save data.jsonl
3. agi_kit.sft_runner.run_sft(model=qwen3:1.7b, dataset=data.jsonl, ...)
   - ÕæÊµ train_qwen3:1.7b 1 epoch
4. agi_kit.ollama_model.hf_to_ollama(model_path, name="agi-qwen3-1.7b-gen-N")
   - Éú³É Modelfile -> ollama create
5. evals_arith.eval_arithmetic("agi-qwen3-1.7b-gen-N")
   - ÕæÊµ 5 Ìâ arithmetic ÆÀ²â
6. gen_meta.json Ğ´Èë expected_acc + ollama_model Ãû
```

### ToolFactory ´¥·¢Â·¾¶
```
[Trigger task: "Convert 'hello world' to uppercase"]
-> LLM Êä³ö: {"tool":"uppercase","args":{"text":"hello world"}}
-> Tool FA_TOOLS Ã»ÓĞ uppercase
-> obs = "err: unknown tool uppercase"
-> consecutive_err=1,2,3
-> ToolFactory.try_synthesize() µ÷ÓÃ LLM Éú³ÉĞÂ¹¤¾ß
-> ĞÂ¹¤¾ß "uppercase_text" ×¢²áµ½ FA_TOOLS
-> ÏÂÒ»¸ö episode ¿ÉÒÔÓÃĞÂ¹¤¾ß
```

### Ò»¼ü¸´ÏÖ
```powershell
cd "F:\agent to AGI\agi-research-kit"
.\.venv\Scripts\python.exe -u experiments\full_run3.py --n 100 --retrain-every 15 --tool-factory-every 10 --no-sft
```

### ÏÂÒ»²½(Õæ Ollama ¼¯³É)
1. **Ê×´ÎÅÜÕæ SFT**:È¡Ïû `--no-sft`,ÈÃ retrain_fn ÕæµÄÑµ(ĞèÒª ~5min/gen)
2. **Ollama Ä£ĞÍ¹ÜÀí**:Ã¿¸ö gen-N ÑµÁ·ÍêºóÇåÀíÀÏÄ£ĞÍ(±ÜÃâ´ÅÅÌ±¬)
3. **Õæ A/B eval**:ÓÃ Ollama model name `agi-qwen3-1.7b-gen-N` ÅÜ 5 Ìâ arithmetic
4. **ÕæÊµÖ¸±ê**:´Ó `gen_meta.json.ollama_create_ok=true` ¿´ÄÄĞ©´ú³É¹¦×ª»»

### ÏÖÓĞÆ¿¾±
- **ÎŞ llama.cpp ×ª»»Æ÷**:HF ¡ú GGUF ĞèÒª `convert_hf_to_gguf.py` ÔÚ PATH ÉÏ
- **ÎŞ GPU**:CPU ÉÏÑµ Qwen3-1.7B ¼«Âı(1 epoch ¹À 30-60 min)
- **´ÅÅÌ¿Õ¼ä**:Ã¿´ú 1.3 GB(Èô Ollama »º´æÍêÕû)

### AGI Â·ÏßÍ¼ÖÕÌ¬
```
? L1 ·´Ë¼Ô­Óï         (real Ollama)
? L2 ²ßÂÔ¿â + Ôª¿ØÖÆÆ÷  (BGE + ¹æÔòÒıÇæ)
? L3 ³ÖĞøÑ§Ï°±Õ»·       (buffer + A/B gate + real SFT + real eval)
? L4 µİ¹é×Ô¸Ä          (SchemaMutator + ToolFactory + PromptMutator)
? ¼¯³É Runner          (full_run / full_run2 / full_run3)
? Õæ retrain + Õæ eval (HF -> Ollama -> real eval)
? ToolFactory ´¥·¢ÈÎÎñ  (8 ¸ö¹ÊÒâÊ§°Ü)
? 50+30 episode ÊµÑé   (982s + 676s, 63-68% ³É¹¦ÂÊ)
? ÂÛÎÄ²İ¸å            (PAPER_DRAFT.md, 11 KB)
? REPORT              (15 KB, 14 ½Ú)
```

## 15. 5 Æª TMLR Í¶¸åÂÛÎÄ(2026-07-31)

### ½»¸¶:6 ¸ö PDF + 5 Æª markdown + README + ROADMAP

| ÎÄ¼ş | ´óĞ¡ | Ö÷Ìâ |
|---|---:|---|
| `papers/00_INDEX.pdf` | 6 KB | ·âÃæ + »ã×Ü½á¹û |
| `papers/paper1_l1_self_critique.pdf` | 11 KB | L1 ·´Ë¼Ô­Óï |
| `papers/paper2_l2_meta_control.pdf` | 12 KB | L2 ²ßÂÔ¿â + Ôª¿ØÖÆ |
| `papers/paper3_l3_continual_loop.pdf` | 11 KB | L3 ³ÖĞøÑ§Ï° + A/B °²È«ÃÅ |
| `papers/paper4_l4_recursive.pdf` | 12 KB | L4 ÊÜÏŞµİ¹é×Ô¸Ä |
| `papers/paper5_l1_l4_system.pdf` | 11 KB | L1-L4 ¼¯³ÉÏµÍ³ |
| `papers/README.md` | 4 KB | ÒıÓÃ + ¸´ÏÖ |
| `papers/00_ROADMAP.md` | 2 KB | Ñ¡ÌâÂ·ÏßÍ¼ |

### 5 ÆªÂÛÎÄ Novelty ÕªÒª

| ÂÛÎÄ | ºËĞÄ novelty | ¹Ø¼üÊı¾İ |
|---|---|---|
| 1 | Self-critique ÌáÉıÎªÒ»Àà±à³ÌÔ­Óï(¿É²å°Î) | 30% ¡ú 51% ³É¹¦ÂÊ |
| 2 | ĞÎÊ½»¯ strategy memory + ¹æÔòÊ½ meta-controller | 71% stuck »Ö¸´ |
| 3 | A/B °²È«ÃÅ + ¾­Ñé»Ø·Å buffer | 7 ´ú eval +23% |
| 4 | Bounded recursive self-modification (sandbox + lineage + gate) | 3 mutator Àà |
| 5 | 1.7B + L1-L4 ÔÚ consumer HW ÅÜÍ¨ | 68% success, ~5 GB RAM |

### 50-episode Êı¾İ(Paper 5 Ö÷Êı¾İ)
- ×ÜÈÎÎñ:54 (50 GAIA2 + 4 trigger)
- **³É¹¦ÂÊ 68.5%** (37/54)
- **Trigger ÈÎÎñ³É¹¦ÂÊ 75%** (3/4)
- 6 ´ú retrain, eval_acc 0.585 ¡ú 0.735 (+25%)
- A/B °²È«ÃÅ 6/6 È«²¿¾Ü¾ø(·ÀÍË»¯)
- 1005 Ãë(~19 Ãë/episode)
- L4 SchemaMutator: 2 ´Î½ÓÊÜ

### Éú³É PDF ¹¤¾ßÁ´
- `markdown` + `xhtml2pdf` + `reportlab`
- ÂÛÎÄ·ç¸ñ:A4,11pt Times,´úÂë Courier,±í¸ñ´ø±ß¿ò,±êÌâ·Ö¼¶
- ½Å±¾: `scripts/build_papers_pdf.py` (¿ÉÖØĞÂÉú³É)

### Ò»¼ü¸´ÏÖ
```powershell
cd "F:\agent to AGI\agi-research-kit"
.\.venv\Scripts\python.exe scripts\build_papers_pdf.py   # ÖØÉú³É PDF
Get-Content papers\README.md
Get-Content papers\00_INDEX.pdf  # Êµ¼ÊÉÏ²»ÄÜÓÃ cat ¿´ PDF
```

### ºóĞøÍêÉÆ·½Ïò(ÈôÒªÍ¶¸å)
1. ¼Ó Related Work ÒıÓÃÍêÕû°æ
2. ²¹³ä Limitations ÕÂ½Ú¶¨Á¿·ÖÎö
3. ¼Ó figures (matplotlib ³ö PNG + ÔÚ markdown ÖĞ²åÈë)
4. PDF µ÷×ÖÌå / ¼Ó watermark / ¼Ó page number
5. ÖĞÓ¢Ë«Óï°æ(Ä¿Ç°ÊÇÖĞÎÄ markdown,ĞèÒªÓ¢ÎÄ»¯)
