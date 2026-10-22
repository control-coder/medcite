"""构造多轮追问评测集 ``examples/public_health_v3_multiturn``：同一份 v3 语料，不新增正确片段标注。

每条对话两轮：第一轮是点明话题的提问，第二轮是省略了话题或用代词指代的追问。追问补全后就是一条已标注的
v3 或 v3-ext 查询（“源查询”），所以追问的正确片段、类别和划分全部继承自源查询：

- 可回答的追问继承源查询的正确片段；不可回答的追问来自源查询的近似主题、语料未覆盖、超出范围类，正确片段为空；
- 划分沿用源查询：test 对话只从 v3 test 与 v3-ext（全部为 test）取源查询，dev 对话只从 v3 dev 取，互不交叉；
- 第一轮的回答在评测中固定下来，保证可复现：话题在语料里时，取该话题页面中第一段不属于源查询正确片段的原文
  （代表系统上一轮交付的一段摘录，不泄露本轮答案）；话题不在语料里（语料未覆盖、超出范围），或页面里只有
  正确片段时，记为拒答。

挑选规则：在源查询里人工挑出问题中有明确话题词（病名、物品、地名）、能换成代词或省略的；话题没有出现在问题里的
（例如“孩子身上最常见的长期疾病是哪个？”）不选。对话内容（第一轮问题、追问、话题词）全部手写在下面的 ``DIALOGUES``，
由 AI 撰写，只有一位复核人，没有医学专家审核。构造时检查：第一轮包含至少一个话题词，追问不包含任何话题词，
追问不等于源查询，第一轮回答不是正确片段，并对全部问题运行与应用相同的泄露检查。

用法::

    python scripts/build_multiturn_set.py            # 生成或核对，已存在时内容不同即报错
    python scripts/build_multiturn_set.py --write    # 覆盖写入
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from medidiag.rag.leakage import run_leakage_check_records

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {"v3": ROOT / "examples/public_health_v3", "v3_ext": ROOT / "examples/public_health_v3_ext"}
TARGET = ROOT / "examples/public_health_v3_multiturn"
ORIGIN = "ai_claude_session"

# (源数据集, 源查询 sample_id, 第一轮问题, 话题页面 source_id 或 None, 话题词, 追问)
DIALOGUES: list[tuple[str, str, str, str | None, tuple[str, ...], str]] = [
    # ---------------- test：v3 test 可回答
    ("v3", "pv2_C_07", "喝酒对健康有哪些危害？", "alcohol", ("酒",), "它里面含有什么有毒物质？"),
    ("v3", "pv2_D_10", "抗微生物药物耐药性严重吗？", "amr", ("耐药",), "这个词到底是什么意思？"),
    ("v3", "pv2_C_11", "贫血常见吗？", "anaemia", ("贫血",), "那它是怎么定义的？"),
    ("v3", "pv2_B_09", "哮喘是一种什么病？", "asthma", ("哮喘",), "它发作的诱发因素有哪些？"),
    ("v3", "pv2_A_21", "霍乱是怎么回事？", "cholera", ("霍乱",), "为什么说喝干净水、用卫生厕所就能防住它？"),
    ("v3", "public_q11", "新冠哪些人容易得重症？", "covid_risk", ("新冠",), "没打过疫苗的人算不算这类高风险人群？"),
    ("v3", "pv2_B_06", "心脏病发作有多危险？", "cvd", ("心脏病",), "它发作主要是什么东西堵塞造成的？"),
    ("v3", "pv2_B_10", "痴呆症是怎么回事？", "dementia", ("痴呆",), "它最常见的类型是什么？"),
    ("v3", "pv2_A_05", "登革热是怎么传播的？", "dengue", ("登革",), "得了它常见有哪些症状？"),
    ("v3", "pv2_D_05", "抑郁症常见吗？", "depression", ("抑郁",), "它是怎么引起的？"),
    ("v3", "pv2_B_02", "2型糖尿病是什么？", "diabetes", ("糖尿病",), "为什么它的症状很多年才被注意到？"),
    ("v3", "pv2_A_12", "埃博拉病的病死率高吗？", "ebola", ("埃博拉",), "这种病毒的天然宿主是哪种动物？"),
    ("v3", "pv2_D_17", "发霉的花生和玉米还能吃吗？", "food_safety", ("发霉", "花生", "玉米"), "吃了这些为什么会伤肝？"),
    ("v3", "pv2_C_02", "游离糖指的是什么？", "healthy_diet", ("游离糖",), "那它每天应该限制在总能量的多少？"),
    ("v3", "pv2_A_20", "乙肝能治好吗？", "hepb", ("乙肝", "B肝"), "这种病毒最常见是通过哪些途径传开的？"),
    ("v3", "pv2_A_03", "艾滋病毒感染后会怎样？", "hiv", ("艾滋", "HIV"), "它通过哪些体液传播？"),
    ("v3", "pv2_D_02", "家里烧煤烧柴做饭，空气污染严重吗？", "household_air", ("空气污染",), "这种污染会引起哪些疾病？"),
    ("v3", "pv2_B_03", "高血压有什么危害？", "hypertension", ("高血压",), "那血压多高才算？"),
    ("v3", "pv2_D_11", "人乳头瘤病毒疫苗是打什么的？", "immunization", ("人乳头瘤", "HPV"), "感染了这种病毒会引起什么疾病？"),
    ("v3", "pv2_A_07", "麻疹的传染性强吗？", "measles", ("麻疹",), "它有哪些症状？"),
    ("v3", "pv2_A_09", "狂犬病是怎么传给人的？", "rabies", ("狂犬",), "要是已经出现临床症状了，会致命吗？"),
    ("v3", "pv2_C_16", "二手烟是什么？", "tobacco", ("二手烟", "烟"), "家里人抽，我们自己不抽也会受害吗？有没有无害的量？"),
    ("v3", "pv2_A_13", "结核病是怎么传播的？", "tb", ("结核", "肺痨"), "全世界大概有多少人带着这种菌却没症状？"),
    ("v3", "pv2_B_25", "糖尿病和癌症哪个更常见？", "diabetes", ("糖尿病", "癌"), "这两种病有没有办法提前预防？"),
    ("v3", "pv2_A_25", "艾滋病毒和乙肝病毒有什么不同？", "hiv", ("艾滋", "乙肝"), "这两种病毒的传播途径有什么相同点？"),
    ("v3", "pv2_A_23", "结核病和艾滋病毒感染都很常见吗？", "tb", ("结核", "艾滋"), "这两者之间有什么关系？"),
    # ---------------- test：v3-ext 可回答
    ("v3_ext", "pv3x_P_017", "疟疾在哪些地方多见？", "malaria", ("疟疾", "打摆子"), "这病是怎么得上的？能治好吗？"),
    ("v3_ext", "pv3x_P_021", "慢阻肺是什么病？", "copd", ("慢阻肺",), "得了这病的人平时最常见哪些难受的表现？"),
    ("v3_ext", "pv3x_P_042", "结直肠癌常见吗？", "colorectal_cancer", ("结直肠", "肠"), "要是发现得早，是不是更容易治好？"),
    ("v3_ext", "pv3x_P_066", "汉坦病毒是什么？", "hantavirus", ("汉坦", "老鼠"), "这种病毒是怎么传到人身上的？"),
    ("v3_ext", "pv3x_P_030", "单纯疱疹病毒感染常见吗？", "herpes_simplex_virus", ("疱疹",), "它一般是怎么传给别人的？"),
    ("v3_ext", "pv3x_P_012", "HIV 是怎么传播的？", "hiv", ("HIV", "艾滋"), "感染了能彻底治好吗？以后还能正常活很久吗？"),
    ("v3_ext", "pv3x_P_118", "宫内节育器是什么？", "intrauterine_devices", ("宫内节育器", "上环", "环"),
     "没生过孩子的小姑娘能用吗？"),
    ("v3_ext", "pv3x_P_041", "利什曼病有哪几种？", "leishmaniasis", ("利什曼", "黑热病"), "怀疑得了这种病该怎么办，怎么查出来？"),
    ("v3_ext", "pv3x_P_080", "腰痛常见吗？", "low_back_pain", ("腰",), "有这毛病的人做理疗锻炼有什么用？"),
    ("v3_ext", "pv3x_P_096", "中东呼吸综合征是怎么传给人的？", "middle_east_respiratory_syndrome_coronav",
     ("中东呼吸", "骆驼"), "它有没有可能像新冠一样蔓延全世界？"),
    ("v3_ext", "pv3x_P_088", "尼帕病毒是从哪里来的？", "nipah_virus", ("尼帕", "蝙蝠"), "感染后会有什么表现？"),
    ("v3_ext", "pv3x_P_098", "河盲症是什么病？", "onchocerciasis", ("河盲", "盘尾"), "它为什么会让眼睛和皮肤都出问题？"),
    ("v3_ext", "pv3x_P_008", "口服避孕药是怎么起作用的？", "oral_contraceptives", ("避孕", "防怀孕"),
     "长期吃这种药，会不会让人更容易长肿瘤？"),
    ("v3_ext", "pv3x_P_117", "鼠疫现在还有吗？", "plague", ("鼠疫", "黑死病"), "它最多见的那一类，是被什么叮了得上的？"),
    ("v3_ext", "pv3x_P_054", "足部象皮肿是怎么得的？", "podoconiosis_non_filarial_lymphoedema", ("象皮肿", "粗腿"),
     "这种病怎么治？"),
    ("v3_ext", "pv3x_P_076", "血吸虫病是怎么感染的？", "schistosomiasis", ("血吸虫", "大肚子"), "这种病每年要死多少人？"),
    ("v3_ext", "pv3x_P_064", "假药劣药有哪些危害？", "substandard_and_falsified_medical_produc", ("假药", "劣药"),
     "要管住这些，政府该怎么做？"),
    ("v3_ext", "pv3x_P_123", "雅司病是什么病？", "yaws", ("雅司", "螺旋体"), "它有疫苗可以打吗？"),
    ("v3_ext", "pv3x_P_105", "黄热病是通过什么传播的？", "yf", ("黄热",), "得了它刚开始有什么症状、多久会退？"),
    ("v3_ext", "pv3x_P_127", "B族链球菌对新生儿有什么危害？", "group_b_streptococcus_gbs", ("链球菌", "B型", "B族"),
     "宝宝出生后晚些时候才发作的那种，家里怎么防？"),
    # ---------------- test：不可回答（近似主题、语料未覆盖、超出范围）
    ("v3", "pv2_A_28", "麻疹是怎么传播的？", "measles", ("麻疹",), "中国去年新增了多少例？"),
    ("v3", "pv2_A_29", "被狗咬了要打狂犬病疫苗吗？", "rabies", ("狂犬",), "我家附近哪个医院的这种疫苗最便宜？"),
    ("v3", "pv2_B_28", "哮喘发作时有什么表现？", "asthma", ("哮喘",), "它每年在全球造成多少人死亡？"),
    ("v3", "pv2_B_29", "癌症能早发现吗？", "cancer", ("癌",), "那早期筛查应该多久做一次？"),
    ("v3", "pv2_C_26", "喝啤酒也伤身体吗？", "alcohol", ("啤酒",), "每瓶大概含多少克酒精？"),
    ("v3", "pv2_C_28", "吸烟对肺有什么伤害？", "tobacco", ("烟",), "要是戒掉了，多久肺功能能恢复？"),
    ("v3", "pv2_C_30", "贫血有哪些表现？", "anaemia", ("贫血",), "成年女性血红蛋白低于多少就算？"),
    ("v3", "pv2_D_27", "抑郁症能治好吗？", "depression", ("抑郁",), "一般要治疗多长时间才能好？"),
    ("v3", "pv2_U_02", "湿疹是怎么引起的？", None, ("湿疹",), "它反复发作的时候怎么护理皮肤？"),
    ("v3", "pv2_U_06", "肾结石是怎么形成的？", None, ("肾结石", "结石"), "它疼起来一般持续多久？"),
    ("v3", "pv2_U_10", "帕金森病是什么病？", None, ("帕金森",), "它早期有什么征兆？"),
    ("v3", "pv2_U_12", "脂肪肝是怎么得的？", None, ("脂肪肝",), "这病还能逆转吗？"),
    ("v3", "pv2_U_16", "水痘是怎么传染的？", None, ("水痘",), "出疹之后多久就不再传染了？"),
    ("v3", "pv2_U_20", "慢性胃炎有什么症状？", None, ("胃炎",), "得了它平时不能吃什么？"),
    ("v3", "pv2_A_32", "上海这周天气怎么样？", None, ("上海",), "明天那边会下雨吗？"),
    ("v3", "pv2_D_35", "新买的笔记本电脑需要装杀毒软件吗？", None, ("笔记本", "电脑"), "那它怎么重装系统？"),
    ("v3_ext", "pv3x_N_134", "高血压患者要经常量血压吗？", "hypertension", ("血压",), "自己在家量的话，一天量几次比较合适？"),
    ("v3_ext", "pv3x_N_145", "沙尘暴对健康有什么影响？", "sand_and_dust_storms", ("沙尘暴",),
     "刮这个的时候出门，戴什么级别的口罩才够？"),
    ("v3_ext", "pv3x_N_160", "疟疾是怎么传播的？", "malaria", ("疟疾", "打摆子"), "去非洲出差，防这个病的药要提前几天开始吃？"),
    ("v3_ext", "pv3x_N_169", "埃博拉有疫苗吗？", "ebola", ("埃博拉",), "这种疫苗打一针，能管多少年？"),
    # ---------------- dev：v3 dev 可回答
    ("v3", "pv2_B_07", "癌症是怎么发生的？", "cancer", ("癌",), "它说的转移是什么意思？"),
    ("v3", "pv2_A_08", "霍乱有什么症状？", "cholera", ("霍乱",), "它是由什么引起的？"),
    ("v3", "pv2_B_08", "慢性阻塞性肺病有哪些表现？", "copd", ("阻塞性肺", "慢阻肺"), "它最常见的原因是什么？"),
    ("v3", "public_q04", "新冠期间要戴口罩吗？", "covid_masks", ("口罩",), "戴了可以保护周围的人吗？"),
    ("v3", "pv2_D_04", "抑郁症常见吗？", "depression", ("抑郁",), "它和正常的情绪低落有什么区别？"),
    ("v3", "pv2_B_01", "糖尿病有哪几种？", "diabetes", ("糖尿病",), "它是怎么引起的，和胰岛素有什么关系？"),
    ("v3", "pv2_D_09", "溺水每年造成多少人死亡？", "drowning", ("溺水",), "它到底是指什么情况？"),
    ("v3", "pv2_C_03", "多吃水果蔬菜有什么好处？", "healthy_diet", ("水果", "蔬菜"), "十岁以上的人每天应该吃多少？"),
    ("v3", "pv2_A_04", "艾滋病毒是怎么传播的？", "hiv", ("艾滋",), "感染后的窗口期是多久？"),
    ("v3", "pv2_B_04", "高血压有什么危害？", "hypertension", ("高血压",), "得了它一般有症状吗，怎么才能知道自己得了？"),
    ("v3", "pv2_A_02", "疟疾是怎么传播的？", "malaria", ("疟疾",), "它会通过输血传播吗？"),
    ("v3", "pv2_C_10", "儿童营养不良严重吗？", "malnutrition", ("营养不良",), "这个词具体指什么？"),
    ("v3", "pv2_B_12", "肥胖有什么危害？", "obesity", ("肥胖",), "成人体重指数多少就算？"),
    ("v3", "pv2_B_11", "牙周病常见吗？", "oral", ("牙周",), "它有什么表现？"),
    ("v3", "pv2_C_05", "久坐对健康有害吗？", "physical_activity", ("久坐",), "具体什么样才算这种行为？"),
    ("v3", "pv2_D_08", "开车超速很常见吗？", "road_traffic", ("超速",), "为什么它会让车祸更严重？"),
    ("v3", "pv2_A_01", "结核病有什么症状？", "tb", ("结核",), "它是怎么通过空气传播的？"),
    ("v3", "pv2_C_06", "二手烟对孩子有害吗？", "tobacco", ("二手烟",), "它有安全的接触水平吗？"),
    ("v3", "pv2_C_09", "反式脂肪有什么危害？", "trans_fat", ("反式脂肪",), "哪些食物里含有工业生产的这种脂肪？"),
    ("v3", "pv2_A_11", "黄热病有什么症状？", "yf", ("黄热",), "它是通过什么蚊子传播的？"),
    # ---------------- dev：不可回答
    ("v3", "pv2_A_26", "结核病能治好吗？", "tb", ("结核",), "住院治疗一般要花多少钱？"),
    ("v3", "pv2_A_27", "登革热有疫苗吗？", "dengue", ("登革",), "这种疫苗一针要多少钱？"),
    ("v3", "pv2_B_26", "糖尿病患者饮食要注意什么？", "diabetes", ("糖尿病",), "这类患者的空腹血糖应该控制在多少？"),
    ("v3", "pv2_B_30", "痴呆症有哪些早期表现？", "dementia", ("痴呆",), "得了这病的人一般还能活多少年？"),
    ("v3", "pv2_D_29", "HPV疫苗是预防什么的？", "immunization", ("HPV", "人乳头瘤"), "这种疫苗一共要打几针，费用大概多少？"),
    ("v3", "pv2_U_01", "痛风是怎么引起的？", None, ("痛风",), "它发作的时候能喝啤酒吗？"),
    ("v3", "pv2_U_07", "甲状腺功能亢进是什么病？", None, ("甲状腺", "甲亢"), "它有哪些表现？"),
    ("v3", "pv2_U_15", "手足口病是怎么传染的？", None, ("手足口",), "它一般几天能好？"),
    ("v3", "pv2_A_31", "车险一般买哪几种？", None, ("车险",), "理赔需要准备哪些材料？"),
    ("v3", "pv2_C_33", "Python 适合做数据处理吗？", None, ("Python",), "用它怎么读取Excel文件？"),
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build() -> list[dict[str, Any]]:
    chunks = read_jsonl(SOURCES["v3"] / "chunks.jsonl")
    queries = {name: {q["sample_id"]: q for q in read_jsonl(path / "queries.jsonl")} for name, path in SOURCES.items()}
    rows: list[dict[str, Any]] = []
    numbers: Counter[str] = Counter()
    seen: set[tuple[str, str]] = set()
    for source, sample_id, first, page, terms, follow_up in DIALOGUES:
        query = queries[source][sample_id]
        if (source, sample_id) in seen:
            raise ValueError(f"{sample_id} 重复使用")
        seen.add((source, sample_id))
        gold = list(query["gold_evidence_ids"])
        if not any(t in first for t in terms):
            raise ValueError(f"{sample_id}：第一轮没有点明话题 {terms}")
        if any(t in follow_up for t in terms):
            raise ValueError(f"{sample_id}：追问里仍有话题词 {terms}")
        if follow_up == query["question"]:
            raise ValueError(f"{sample_id}：追问与源查询相同")
        if page is not None and not any(c["source_id"] == page for c in chunks):
            raise ValueError(f"{sample_id}：语料里没有页面 {page}")
        answer = next((c for c in chunks if c["source_id"] == page and c["chunk_id"] not in gold), None)
        history = ({"question": first, "outcome": "answered", "answer": [answer["text"]],
                    "answer_chunk_ids": [answer["chunk_id"]]} if answer is not None
                   else {"question": first, "outcome": "abstained", "answer": [], "answer_chunk_ids": []})
        split = query["split"]
        numbers[split] += 1
        rows.append({
            "sample_id": f"pmt_{'T' if split == 'test' else 'D'}_{numbers[split]:02d}",
            "split": split, "bucket": query["bucket"],
            "source_dataset": source, "source_sample_id": sample_id, "source_question": query["question"],
            "gold_evidence_ids": gold, "history": [history], "follow_up": follow_up, "topic_terms": list(terms),
            "origin": ORIGIN,
        })
    records = [{"sample_id": r["sample_id"], "question": q} for r in rows
               for q in (r["follow_up"], r["history"][0]["question"])]
    hits = run_leakage_check_records(records, chunks, {"chunk_fields_to_check": ["source", "source_id", "metadata.raw_id"],
                                                       "check_question_text": True, "check_answer_key": True})
    if hits:
        raise ValueError(f"泄露检查命中：{hits[:3]}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true", help="覆盖已有的 dialogues.jsonl")
    args = parser.parse_args()
    rows = build()
    content = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    target = TARGET / "dialogues.jsonl"
    if target.exists() and not args.write and target.read_text(encoding="utf-8") != content:
        raise SystemExit(f"{target} 与重新生成的内容不同，确认后加 --write 覆盖")
    TARGET.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8", newline="\n")
    counts = Counter((r["split"], r["bucket"]) for r in rows)
    print("对话数：", len(rows), dict(sorted(counts.items())))
    print("dialogues.jsonl sha256", hashlib.sha256(target.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
