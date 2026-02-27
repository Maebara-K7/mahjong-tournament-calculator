# practice.py
"""
麻将终局条件计算练习程序

这个程序会自动生成一个随机的麻将终局场景，
并要求你计算在某种情况下的和牌条件。
"""

import random
import json
import os
import multiprocessing
import time
import math
from cond_lib import *
from rulesets import RULES
import calculation

QBANK_FILE = 'devil_mode_qbank.json'
GENERATION_ATTEMPT_LIMIT = 100
MIN_OR_COUNT_FOR_DEVIL = 2


def _generate_unbounded_geom():
    """生成一个随机整数 k >= 0，其概率与 1/(2^k) 成正比。"""
    k = 0
    # 返回k的概率是(1/2)^(k+1)，与1/(2^k)成正比。
    # 这为每次增量提供了所需的“减半概率”。
    while random.random() < 0.5:
        k += 1
    return k


def _generate_geom_with_max(max_val):
    """
    生成一个在[0, max_val]范围内的随机整数k，其概率与1/(2^k)成正比。
    这是通过对无界生成器进行拒绝抽样来完成的。
    """
    if max_val < 0:
        return 0  # 不应该发生，但作为安全措施
    while True:
        k = _generate_unbounded_geom()
        if k <= max_val:
            return k


def _get_high_fu_cheat_sheet(player_to_ask, oya, question_type, rules):
    """为高符数（70, 90, 110）生成一个提示单。"""
    is_oya = player_to_ask == oya
    is_tsumo = question_type == "tsumo"

    fu_to_display = [70, 90, 110]
    han_to_display = [1, 2]

    type_str = ("亲家" if is_oya else "子家") + ("自摸" if is_tsumo else "荣和")
    header = f"--- 高符数点数速查表 ({type_str}) ---"

    table_lines = [header]
    col_headers = ["符数"] + [f"{h}番" for h in han_to_display]
    table_lines.append("{:<5} | {:<9} | {:<9}".format(*col_headers))
    table_lines.append("-" * (len(table_lines[1]) + 2))

    for fu in fu_to_display:
        row_data = [f"{fu}符"]
        for han in han_to_display:
            # 不存在 1 番 110 符自摸
            if is_tsumo and han == 1 and fu == 110:
                row_data.append("--")
                continue

            if is_tsumo:
                if is_oya:
                    # 亲家自摸
                    pts = ceil_to_hundred(2 * fu * 2 ** (han + 2))
                    row_data.append(f"{pts} all")
                else:
                    # 子家自摸
                    ko_pts = ceil_to_hundred(fu * 2 ** (han + 2))
                    oya_pts = ceil_to_hundred(2 * fu * 2 ** (han + 2))
                    row_data.append(f"{ko_pts}/{oya_pts}")
            else:
                if is_oya:
                    # 亲家荣和
                    pts = ceil_to_hundred(6 * fu * 2 ** (han + 2))
                    row_data.append(str(pts))
                else:
                    # 子家荣和
                    pts = ceil_to_hundred(4 * fu * 2 ** (han + 2))
                    row_data.append(str(pts))

        table_lines.append("{:<5} | {:<9} | {:<9}".format(*row_data))

    table_lines.append("-" * (len(table_lines[1]) + 2))
    return "\n".join(table_lines)


def normalize_student_answer(answer_str):
    """
    规范化学生的答案字符串，以处理不连续条件的各种分隔符，
    同时正确忽略括号内的逗号（例如，在自摸点数格式中）。
    """
    # 首先，将明确的分隔符规范化为标准的'或'。
    # 我们处理带空格的' or '以及其他常见分隔符。
    norm_str = answer_str.strip().lower()
    norm_str = norm_str.replace(' or ', '或')
    norm_str = norm_str.replace('||', '或')
    norm_str = norm_str.replace('|', '或')
    norm_str = norm_str.replace(';', '或')

    # 删除所有空格以便进行一致的比较
    norm_str = norm_str.replace(' ', '')
    # 同时替换没有空格的'or'，这可能是删除空格后产生的结果
    norm_str = norm_str.replace('or', '或')

    # 手动迭代，只替换括号外的逗号
    result = []
    paren_level = 0
    for char in norm_str:
        if char == '(':
            paren_level += 1
        elif char == ')':
            paren_level = max(0, paren_level - 1)  # 避免变为负数

        if char == ',' and paren_level == 0:
            result.append('或')
        else:
            result.append(char)

    return "".join(result)


def _parse_str_to_val(s, is_tsumo):
    """(内部使用) 将字符串部分解析为整数或元组。"""
    s = s.strip().replace('(', '').replace(')', '')
    if is_tsumo:
        # 允许使用连字符作为自摸点的分隔符，例如300-500
        s = s.replace('-', '/')
        try:
            if '/' in s:
                parts = s.split('/')
                return int(parts[0]), int(parts[1])
            elif ',' in s:
                parts = s.split(',')
                return int(parts[0]), int(parts[1])
            else:
                # 对于自摸题，不接受不完整的单个数字格式
                raise ValueError("Incomplete tsumo point format")
        except (ValueError, IndexError):
            raise ValueError("Invalid tsumo point format")
    else:  # is_ron
        return int(s)


def parse_answer_to_pts(answer_str, all_possible_pts):
    """解析答案字符串，返回一个包含所有对应点数的集合。"""
    if not all_possible_pts:
        return set()

    # 根据点数的数据类型判断问题是自摸还是荣和
    sample_pt = next(iter(all_possible_pts), None)
    is_tsumo = isinstance(sample_pt, tuple)

    # 处理 'all', 'ok' 等特殊答案
    norm_for_alias = answer_str.strip().lower().replace(" ", "")
    ok_aliases = ['all', '0', 'o', 'ok', '〇', '']
    if norm_for_alias in ok_aliases:
        return set(all_possible_pts)

    normalized_str = normalize_student_answer(answer_str)
    parts = normalized_str.split('或')
    final_pts = set()

    for part in parts:
        part = part.strip()
        if not part:
            continue

        try:
            # 处理比较符
            if part.startswith('>='):
                val = _parse_str_to_val(part[2:], is_tsumo)
                # 特殊处理：如果条件是 ">=" 理论最大值，则等同于只要求该最大值
                max_pt = max(all_possible_pts) if all_possible_pts else None
                if val == max_pt:
                    final_pts.add(val)
                else:
                    final_pts.update({p for p in all_possible_pts if p >= val})
                continue
            elif part.startswith('<='):
                val = _parse_str_to_val(part[2:], is_tsumo)
                final_pts.update({p for p in all_possible_pts if p <= val})
                continue
            elif part.startswith('>'):
                val = _parse_str_to_val(part[1:], is_tsumo)
                final_pts.update({p for p in all_possible_pts if p > val})
                continue
            elif part.startswith('<'):
                val = _parse_str_to_val(part[1:], is_tsumo)
                final_pts.update({p for p in all_possible_pts if p < val})
                continue

            # 处理范围
            if '~' in part:
                start_str, end_str = part.split('~')
                start_val = _parse_str_to_val(start_str, is_tsumo)
                end_val = _parse_str_to_val(end_str, is_tsumo)
                final_pts.update({p for p in all_possible_pts if start_val <= p <= end_val})
            
            # 处理单个值
            else:
                cleaned_part = part.lstrip('=')
                val = _parse_str_to_val(cleaned_part, is_tsumo)
                if val in all_possible_pts:
                    final_pts.add(val)

        except (ValueError, IndexError):
            # 忽略无法解析的部分，例如在自摸题中输入 '>=2000'
            continue

    return final_pts



def generate_random_scenario(forced_q_type=None):
    """生成一个随机的对局情景和问题"""

    # 1. 随机选择规则 (排除 "CUS")
    available_rules = {k: v for k, v in RULES.items() if k != "CUS"}
    ruleset_name = random.choice(list(available_rules.keys()))
    rules = available_rules[ruleset_name]

    # 2. 随机生成赛前积分
    # 通过模拟1-2场比赛来生成更真实的、与规则相关的赛前积分
    start_pts = [0.0] * 4
    num_games_to_simulate = random.randint(1, 2)

    for _ in range(num_games_to_simulate):
        # 为模拟的比赛生成随机的局内点数
        total_points_sim = rules["starting_pts"] * 4
        weights_sim = [random.random() for _ in range(4)]
        total_weight_sim = sum(weights_sim)
        unrounded_pts_sim = [(w / total_weight_sim) * total_points_sim for w in weights_sim]
        simulated_pts_in_hand = [round(p / 100) * 100 for p in unrounded_pts_sim]
        current_sum_sim = sum(simulated_pts_in_hand)
        diff_sim = total_points_sim - current_sum_sim
        simulated_pts_in_hand[random.randint(0, 3)] += diff_sim

        # 使用 game_pts 计算该场模拟比赛的得分
        game_scores = game_pts(
            simulated_pts_in_hand,
            rules["placement_pts"],
            rules["starting_pts"],
            tie_resolution=rules.get("tie_resolution", "split_points"),
            deposit_final_draw_recipient=rules.get("deposit_final_draw_recipient", "first_place")
        )

        # 累积得分
        for i in range(4):
            start_pts[i] += game_scores[i]

    # 将最终分数四舍五入到一位小数
    start_pts = [round(p, 1) for p in start_pts]

    # 3. 随机生成本场、场供等参数
    oya = 3
    # 根据概率减半的无上限分布生成本场和场供，并满足 deposit <= honba * 4 的约束

    # 1. 先生成本场数 (honba)
    honba = _generate_unbounded_geom()
    max_deposit = honba * 4

    # 2. 再根据约束生成有效的立直棒数 (deposit)
    deposit = _generate_geom_with_max(max_deposit)
    riichi_status = [0] * 4

    # 4. 根据场供生成局内点数
    # 总点数需要减去场供的点数
    total_points_in_hands = rules["starting_pts"] * 4 - (deposit * 1000)

    # 生成四个随机权重
    weights = [random.random() for _ in range(4)]
    total_weight = sum(weights)

    # 根据权重分配点数
    unrounded_pts = [(w / total_weight) * total_points_in_hands for w in weights]

    # 四舍五入到100的倍数，并处理总和偏差
    pts_in_hand = [round(p / 100) * 100 for p in unrounded_pts]
    current_sum = sum(pts_in_hand)
    diff = total_points_in_hands - current_sum

    # 将偏差加到随机一个选手身上
    pts_in_hand[random.randint(0, 3)] += diff

    # 5. 智能生成问题
    # 预计算当前排名以提出更有意义的问题
    other_players_pts = []
    all_pts = start_pts + other_players_pts

    # 生成 "先行有利" tiebreaker
    sorted_players_for_tiebreaker = sorted(enumerate(all_pts), key=lambda x: (-x[1], x[0]))
    tiebreaker = [0] * len(all_pts)
    for rank, (player_index, _) in enumerate(sorted_players_for_tiebreaker):
        tiebreaker[player_index] = rank

    # 计算当前总分和排名 (模拟当前瞬间流局来评估)
    current_game_pts = game_pts(
        pts_in_hand, rules['placement_pts'], rules['starting_pts'],
        deposit_final_draw_recipient="unclaimed"
    )
    current_total_pts = [round(u + v, 1) for u, v in zip(start_pts, current_game_pts)] + other_players_pts
    current_rank = final_ranking(current_total_pts, tiebreaker)

    # 选择一个非第一名的选手进行提问
    eligible_players = [i for i, rank in enumerate(current_rank) if rank > 1]
    if not eligible_players:  # 如果所有人都并列第一，则随机选一个
        eligible_players = list(range(4))
    player_to_ask = random.choice(eligible_players)
    player_rank = current_rank[player_to_ask]

    # 根据选手当前排名设定一个有挑战性的目标
    if player_rank == 2:
        goal_placement = 1
    elif player_rank == 3:
        goal_placement = random.choice([1, 2])
    else:  # player_rank == 4 or tied
        goal_placement = random.choice([2, 3])

    # 随机选择问题类型 (自摸或荣和)
    if forced_q_type:
        if forced_q_type == 'tsumo':
            question_type = 'tsumo'
        else:  # ron
            possible_ron_targets = [p for p in range(4) if p != player_to_ask]
            if not possible_ron_targets:  # Should not happen in a 4-player game
                question_type = 'tsumo'  # fallback
            else:
                target_player = random.choice(possible_ron_targets)
                question_type = f"ron_{target_player}"
    else:
        # Fallback to original logic if no type is forced
        possible_question_types = ["tsumo"]
        for i in range(4):
            if i != player_to_ask:
                possible_question_types.append(f"ron_{i}")
        question_type = random.choice(possible_question_types)

    return {
        "ruleset_name": ruleset_name,
        "rules": rules,
        "start_pts": start_pts,
        "pts_in_hand": pts_in_hand,
        "oya": oya,
        "honba": honba,
        "deposit": deposit,
        "riichi_status": riichi_status,
        "player_to_ask": player_to_ask,
        "goal_placement": goal_placement,
        "question_type": question_type,
        "other_players_pts": other_players_pts,
        "tiebreaker": tiebreaker,
        "current_total_pts": current_total_pts
    }


def _is_scenario_difficult(scenario, min_or_count):
    """检查一个场景是否满足指定的难度（'或'的数量）"""
    if min_or_count == 0:
        return True  # 普通模式总是满足

    try:
        temp_rules = scenario["rules"]
        temp_all_pts = scenario["start_pts"] + scenario["other_players_pts"]
        temp_tiebreaker = scenario["tiebreaker"]

        oya_tsumo, ko_tsumo, oya_ron, ko_ron = generate_all_possible_points(
            kiriage_mangan=temp_rules.get("kiriage_mangan", True),
            allow_double_yakuman=temp_rules.get("allow_double_yakuman", True),
            allow_composite_yakuman=temp_rules.get("allow_composite_yakuman", True)
        )

        q_type = scenario["question_type"]
        player = scenario["player_to_ask"]
        goal = scenario["goal_placement"]

        pts_list = []
        ok_end, ok_continue = [], []

        if q_type == "tsumo":
            pts_list = oya_tsumo if player == scenario["oya"] else ko_tsumo
            ok_end, ok_continue = calculation.calculate_tsumo_conditions(
                player, scenario["oya"], scenario["pts_in_hand"], scenario["honba"], scenario["deposit"],
                scenario["riichi_status"], temp_all_pts, goal, temp_tiebreaker, temp_rules, pts_list)

        elif q_type.startswith("ron_"):
            target_player = int(q_type.split('_')[1])
            pts_list = oya_ron if player == scenario["oya"] else ko_ron
            ok_end, ok_continue = calculation.calculate_ron_conditions(
                player, target_player, scenario["oya"], scenario["pts_in_hand"], scenario["honba"],
                scenario["deposit"], scenario["riichi_status"], temp_all_pts, goal, temp_tiebreaker, temp_rules,
                pts_list)

        merged_ok_pts = sorted(list(set(ok_end) | set(ok_continue)))
        merged_correct_str = format_as_intervals(merged_ok_pts, pts_list)

        return merged_correct_str.count("或") >= min_or_count
    except Exception:
        # 如果计算过程中出现任何错误，则认为不满足难度条件
        return False


# --- 用于并行生成题库的函数 ---

def _find_one_difficult_scenario_worker(q_type, min_or_count):
    """(内部使用) 为多进程设计的独立工作函数，用于寻找一个高难度场景。"""
    while True:
        scenario = generate_random_scenario(forced_q_type=q_type)
        if _is_scenario_difficult(scenario, min_or_count):
            return scenario

def _tsumo_worker_for_pool(_):
    """(内部使用) 多进程池的自摸场景生成器。"""
    return _find_one_difficult_scenario_worker('tsumo', MIN_OR_COUNT_FOR_DEVIL)

def _ron_worker_for_pool(_):
    """(内部使用) 多进程池的荣和场景生成器。"""
    return _find_one_difficult_scenario_worker('ron', MIN_OR_COUNT_FOR_DEVIL)


def load_qbank():
    """加载题库文件"""
    if os.path.exists(QBANK_FILE):
        with open(QBANK_FILE, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {"tsumo": [], "ron": []}
    return {"tsumo": [], "ron": []}


def save_qbank(qbank):
    """保存题库文件"""
    with open(QBANK_FILE, 'w', encoding='utf-8') as f:
        json.dump(qbank, f, ensure_ascii=False, indent=2)


def populate_initial_qbank():
    """生成初始的魔鬼模式题库 (使用并行处理)"""
    print("正在生成初始魔鬼模式题库，将使用多核并行处理以加快速度...")
    qbank = {"tsumo": [], "ron": []}

    try:
        # 留出一些核心以保持系统响应
        num_processes = max(1, multiprocessing.cpu_count() - 1)
    except NotImplementedError:
        num_processes = 4  # Fallback
    print(f"将使用 {num_processes} 个进程。")

    with multiprocessing.Pool(processes=num_processes) as pool:
        # --- 生成自摸问题 ---
        print("正在并行生成100个自摸难题...")
        tsumo_results = pool.imap_unordered(_tsumo_worker_for_pool, range(100))

        for i, scenario in enumerate(tsumo_results, 1):
            qbank["tsumo"].append(scenario)
            print(f"已找到 {i}/100 个自摸难题。", flush=True)

        # --- 生成荣和问题 ---
        print("正在并行生成100个荣和难题...")
        ron_results = pool.imap_unordered(_ron_worker_for_pool, range(100))

        for i, scenario in enumerate(ron_results, 1):
            qbank["ron"].append(scenario)
            print(f"已找到 {i}/100 个荣和难题。", flush=True)

    save_qbank(qbank)
    print(f"题库生成完毕，已保存至 {QBANK_FILE}")


def start_quiz(min_or_count=0):
    """开始一个练习会话"""

    # --- 1. 场景生成 ---
    q_type_choice = random.choice(['tsumo', 'ron'])
    scenario = None

    # 魔鬼模式使用题库
    if min_or_count >= MIN_OR_COUNT_FOR_DEVIL:
        print(f"正在生成魔鬼模式问题 (至少 {min_or_count} 个'或')...")
        qbank = load_qbank()
        found_new_scenario = False

        # 尝试生成一个新题目
        for _ in range(GENERATION_ATTEMPT_LIMIT):
            temp_scenario = generate_random_scenario(forced_q_type=q_type_choice)
            if _is_scenario_difficult(temp_scenario, min_or_count):
                scenario = temp_scenario
                found_new_scenario = True
                print("找到了一个新难题！已加入题库。")

                # 加入题库并保存
                bank_key = "tsumo" if q_type_choice == "tsumo" else "ron"
                if not "tsumo" in qbank: qbank["tsumo"] = []
                if not "ron" in qbank: qbank["ron"] = []
                qbank[bank_key].append(scenario)
                save_qbank(qbank)
                break

        # 如果没有找到新题，则从题库中选取
        if not found_new_scenario:
            bank_key = "tsumo" if q_type_choice == "tsumo" else "ron"
            if qbank.get(bank_key):
                print(f"未能在{GENERATION_ATTEMPT_LIMIT}次尝试中生成新难题，从题库中随机选择。")
                scenario = random.choice(qbank[bank_key])
            else:
                print(f"题库为空，且无法生成新难题。请先运行菜单选项99填充题库。")
                return

    # 困难或普通模式（无题库）
    else:
        if min_or_count > 0:
            mode_name = "困难"
            print(f"正在生成{mode_name}模式问题，请稍候...")
            while True:
                temp_scenario = generate_random_scenario(forced_q_type=q_type_choice)
                if _is_scenario_difficult(temp_scenario, min_or_count):
                    scenario = temp_scenario
                    break
        else:  # 普通模式
            scenario = generate_random_scenario(forced_q_type=q_type_choice)

    if scenario is None:
        print("无法生成或获取场景，练习中止。")
        return
    rules = scenario["rules"]
    all_pts = scenario["start_pts"] + scenario["other_players_pts"]
    tiebreaker = scenario["tiebreaker"]

    # --- 2. 打印场景和问题 ---
    WIND_MAP = {0: "东", 1: "南", 2: "西", 3: "北"}
    oya_wind = WIND_MAP[scenario["oya"]]

    print("\n--- 麻将终局条件计算练习 ---")
    print(f"规则: {rules['name']}")
    print(f"顺位马: {rules['placement_pts']}")
    print(f"赛前积分: {scenario['start_pts']}")

    # 计算并显示局内点差
    player_to_ask_idx = scenario['player_to_ask']
    pts_in_hand = scenario['pts_in_hand']
    player_to_ask_score = pts_in_hand[player_to_ask_idx]
    point_diffs = [p - player_to_ask_score for p in pts_in_hand]

    print(f"局内点数: {pts_in_hand}")
    print(f"局内点差: {point_diffs} (相对选手 {player_to_ask_idx})")

    # 显示当前总分
    print(f"当前总分: {scenario['current_total_pts']}")

    print(f"庄家: 选手 {scenario['oya']} ({oya_wind})")
    print(f"场供: {scenario['deposit'] * 1000} 点, 本场: {scenario['honba']}")
    print("-" * 30)

    # 提问之前给出速查表
    cheat_sheet = _get_high_fu_cheat_sheet(
        scenario["player_to_ask"], scenario["oya"], scenario["question_type"], rules
    )
    print(cheat_sheet)
    print("")

    # --- 3. 计算正确答案 ---
    oya_tsumo, ko_tsumo, oya_ron, ko_ron = generate_all_possible_points(
        kiriage_mangan=rules.get("kiriage_mangan", True),
        allow_double_yakuman=rules.get("allow_double_yakuman", True),
        allow_composite_yakuman=rules.get("allow_composite_yakuman", True)
    )

    question_str = ""
    correct_answer_str = "✕"

    q_type = scenario["question_type"]
    player = scenario["player_to_ask"]
    goal = scenario["goal_placement"]

    player_wind = WIND_MAP[player]
    if q_type == "tsumo":
        question_str = f"问题: 选手 {player} ({player_wind}) 为获得前 {goal} 名，其自摸条件是什么？"
        pts_list = oya_tsumo if player == scenario["oya"] else ko_tsumo
        ok_end, ok_continue = calculation.calculate_tsumo_conditions(
            player, scenario["oya"], scenario["pts_in_hand"], scenario["honba"], scenario["deposit"],
            scenario["riichi_status"], all_pts, goal, tiebreaker, rules, pts_list)

    elif q_type.startswith("ron_"):
        target_player = int(q_type.split('_')[1])
        target_wind = WIND_MAP[target_player]
        question_str = f"问题: 选手 {player} ({player_wind}) 为获得前 {goal} 名，其荣和选手 {target_player} ({target_wind}) 的条件是什么？"
        pts_list = oya_ron if player == scenario["oya"] else ko_ron
        ok_end, ok_continue = calculation.calculate_ron_conditions(
            player, target_player, scenario["oya"], scenario["pts_in_hand"], scenario["honba"],
            scenario["deposit"], scenario["riichi_status"], all_pts, goal, tiebreaker, rules, pts_list)

    else:
        print("错误：不支持的问题类型。")
        return

    # 格式化答案
    end_str = format_as_intervals(ok_end, pts_list)
    continue_str = format_as_intervals(ok_continue, pts_list)

    # 生成合并后的答案字符串，用于更灵活的评判
    merged_ok_pts = sorted(list(set(ok_end) | set(ok_continue)))
    merged_correct_str = format_as_intervals(merged_ok_pts, pts_list)

    # BUG FIX: 为亲家自摸的答案补上 " all"
    is_tsumo = scenario["question_type"] == "tsumo"
    is_oya = scenario["player_to_ask"] == scenario["oya"]
    if is_tsumo and is_oya:
        # 检查字符串是否是表示条件的格式 (e.g., ">= 1000")
        if end_str and end_str not in ["〇", "✕"]:
            end_str += " all"
        if continue_str and continue_str not in ["〇", "✕"]:
            continue_str += " all"
        if merged_correct_str and merged_correct_str not in ["〇", "✕"]:
            merged_correct_str += " all"

    is_end_valid = end_str and end_str != '✕'
    is_continue_valid = continue_str and continue_str != '✕'
    if not is_end_valid and not is_continue_valid:
        correct_answer_str = "✕"
    elif is_end_valid and not is_continue_valid:
        correct_answer_str = end_str
    elif not is_end_valid and is_continue_valid:
        correct_answer_str = f"{continue_str} (续行)"
    else:
        correct_answer_str = f"{end_str} (终局) / {continue_str} (续行)"

    # --- 6. 向学生提问并获取答案 ---
    print(question_str)
    print("(提示: 请按程序输出的格式作答。若答案为'〇'，可直接回车。)")
    print("(不连续条件请使用 逗号(,)、分号(;) 或 or 分隔，例如: <=2000, >=12000)")
    print("(点数闭区间请使用 波浪线(~) 分隔，例如: 3400~3900, >=24000)")
    start_time = time.time()
    student_answer = input("你的答案: ")
    end_time = time.time()

    # --- 5. 评判并给出反馈 ---
    print("-" * 30, flush=True)
    print(f"正确答案是: {correct_answer_str}", flush=True)

    # 将正确答案和学生答案都解析为点数集合进行比较
    correct_pts_set = parse_answer_to_pts(merged_correct_str, pts_list)
    student_pts_set = parse_answer_to_pts(student_answer, pts_list)

    # 特殊情况处理：〇 和 ✕
    student_input_norm = student_answer.strip().lower().replace(" ", "")
    ok_aliases = ['all', '0', 'o', 'ok', '〇', '']
    no_aliases = ['none', 'x', 'no', '✕']

    is_student_ok = student_input_norm in ok_aliases
    is_correct_ok = (merged_correct_str == "〇")

    is_student_no = student_input_norm in no_aliases
    is_correct_no = (merged_correct_str == "✕")

    is_correct = False
    if is_student_ok and is_correct_ok:
        is_correct = True
    elif is_student_no and is_correct_no:
        is_correct = True
    # 对非特殊情况, 比较分析得到的点数区间
    elif not is_correct_ok and not is_correct_no and correct_pts_set == student_pts_set:
        is_correct = True

    if is_correct:
        print("\n恭喜你，完全正确！你已经掌握了！", flush=True)
    else:
        print("\n答案不完全正确，请对照正确答案，检查你的计算过程。", flush=True)

    time_taken = end_time - start_time
    print(f"本题用时: {time_taken:.2f} 秒", flush=True)


def _display_exam_results(exam_results, seed, total_time):
    """显示考试结果的摘要。"""
    print("\n" + "="*40)
    print(" " * 15 + "考试结果")
    print("="*40)

    time_limit_seconds = 30 * 60  # 30分钟
    overtime = max(0, total_time - time_limit_seconds)
    penalty_points = math.ceil(overtime / 15) if overtime > 0 else 0

    total_minutes, total_seconds = divmod(int(total_time), 60)
    print(f"总用时: {total_minutes} 分 {total_seconds} 秒 (限时 30 分钟)")
    if overtime > 0:
        overtime_minutes, overtime_seconds = divmod(int(overtime), 60)
        print(f"超时: {overtime_minutes} 分 {overtime_seconds} 秒，扣分: -{penalty_points} 分")

    difficulty_points = {'simple': 10, 'hard': 20, 'devil': 30}

    simple_correct = sum(1 for r in exam_results if r['is_correct'] and r.get('difficulty') == 'simple')
    hard_correct = sum(1 for r in exam_results if r['is_correct'] and r.get('difficulty') == 'hard')
    devil_correct = sum(1 for r in exam_results if r['is_correct'] and r.get('difficulty') == 'devil')

    simple_score = simple_correct * difficulty_points['simple']
    hard_score = hard_correct * difficulty_points['hard']
    devil_score = devil_correct * difficulty_points['devil']
    base_score = simple_score + hard_score + devil_score
    final_score = base_score - penalty_points

    print("\n--- 得分报告 ---")
    print(f"普通题 (3题): {simple_correct}/3, 得分 {simple_score}/30")
    print(f"困难题 (2题): {hard_correct}/2, 得分 {hard_score}/40")
    print(f"魔鬼题 (1题): {devil_correct}/1, 得分 {devil_score}/30")
    print("-" * 20)
    print(f"基础得分: {base_score} / 100")
    if penalty_points > 0:
        print(f"超时扣分: -{penalty_points}")
    print(f"最终总分: {final_score} / 100")
    print(f"本次考试种子: {seed} (可在菜单使用此种子复盘)")
    print("="*40)

    if final_score < 100:
        print("\n--- 错题回顾 ---")
        for i, result in enumerate(exam_results):
            if not result['is_correct']:
                difficulty = result.get('difficulty', 'unknown')
                points = difficulty_points.get(difficulty, 0)
                time_taken_str = f"用时: {result['time_taken']:.2f}秒"
                print(f"\n--- 第 {i+1} 题 (错误, 本题 {points} 分, {time_taken_str}) ---")
                print(result['question_str'])
                print(f"你的答案: {result['student_answer_raw']}")
                print(f"正确答案: {result['correct_answer_str']}")


def start_exam_mode(seed=None):
    """开始一个包含6个问题的考试，最后统一评分。"""
    # --- 0. 设置种子 ---
    if seed is None:
        exam_seed = int(time.time())
        print(f"\n--- 考试模式启动 (新考试) ---", flush=True)
    else:
        exam_seed = seed
        print(f"\n--- 考试模式启动 (复盘模式, 种子: {exam_seed}) ---", flush=True)
    random.seed(exam_seed)

    # --- 1. 生成考卷计划 ---
    print("正在生成考卷...", flush=True)
    simple_q_plan = [{'difficulty': 0, 'level': 'simple', 'q_type': random.choice(['tsumo', 'ron'])} for _ in range(3)]
    
    if random.random() < 0.5:
        # 模式一: 2个困难(自摸) + 1个魔鬼(荣和)
        hard_q_plan = [{'difficulty': 1, 'level': 'hard', 'q_type': 'tsumo'} for _ in range(2)]
        devil_q_plan = [{'difficulty': MIN_OR_COUNT_FOR_DEVIL, 'level': 'devil', 'q_type': 'ron'}]
    else:
        # 模式二: 2个困难(荣和) + 1个魔鬼(自摸)
        hard_q_plan = [{'difficulty': 1, 'level': 'hard', 'q_type': 'ron'} for _ in range(2)]
        devil_q_plan = [{'difficulty': MIN_OR_COUNT_FOR_DEVIL, 'level': 'devil', 'q_type': 'tsumo'}]

    exam_plan = simple_q_plan + hard_q_plan + devil_q_plan
    random.shuffle(exam_plan)
    exam_results = []

    print(f"共 {len(exam_plan)} 题，请依次作答。答案将在最后统一公布。", flush=True)
    print("考试限时 30 分钟，超时将按每 15 秒扣 1 分的规则进行罚分。祝你好运！", flush=True)

    exam_start_time = time.time()

    for i, q_info in enumerate(exam_plan):
        print(f"\n--- 第 {i+1}/{len(exam_plan)} 题 ---", flush=True)
        min_or_count = q_info['difficulty']
        q_type_choice = q_info['q_type']

        # --- 2. 场景生成 (逻辑来自 start_quiz) ---
        scenario = None
        # 魔鬼模式使用题库
        if min_or_count >= MIN_OR_COUNT_FOR_DEVIL:
            qbank = load_qbank()
            bank_key = "tsumo" if q_type_choice == "tsumo" else "ron"
            if qbank.get(bank_key) and qbank[bank_key]:
                scenario = random.choice(qbank[bank_key])
            else:
                print(f"警告：魔鬼模式题库({bank_key})为空，降级为实时生成。", flush=True)
                # 降级，实时生成一个魔鬼题
                for _ in range(GENERATION_ATTEMPT_LIMIT):
                    temp_scenario = generate_random_scenario(forced_q_type=q_type_choice)
                    if _is_scenario_difficult(temp_scenario, min_or_count):
                        scenario = temp_scenario
                        break
        # 困难或普通模式
        else:
            while True:
                temp_scenario = generate_random_scenario(forced_q_type=q_type_choice)
                if _is_scenario_difficult(temp_scenario, min_or_count):
                    scenario = temp_scenario
                    break

        if scenario is None:
            print("无法生成或获取场景，跳过此题。", flush=True)
            continue
        
        rules = scenario["rules"]
        all_pts = scenario["start_pts"] + scenario["other_players_pts"]
        tiebreaker = scenario["tiebreaker"]

        # --- 3. 打印场景和问题 ---
        WIND_MAP = {0: "东", 1: "南", 2: "西", 3: "北"}
        oya_wind = WIND_MAP[scenario["oya"]]

        print(f"规则: {rules['name']}")
        print(f"顺位马: {rules['placement_pts']}")
        print(f"赛前积分: {scenario['start_pts']}")
        player_to_ask_idx = scenario['player_to_ask']
        pts_in_hand = scenario['pts_in_hand']
        player_to_ask_score = pts_in_hand[player_to_ask_idx]
        point_diffs = [p - player_to_ask_score for p in pts_in_hand]
        print(f"局内点数: {pts_in_hand}")
        print(f"局内点差: {point_diffs} (相对选手 {player_to_ask_idx})")
        print(f"当前总分: {scenario['current_total_pts']}")
        print(f"庄家: 选手 {scenario['oya']} ({oya_wind})")
        print(f"场供: {scenario['deposit'] * 1000} 点, 本场: {scenario['honba']}")
        print("-" * 30)
        cheat_sheet = _get_high_fu_cheat_sheet(scenario["player_to_ask"], scenario["oya"], scenario["question_type"], rules)
        print(cheat_sheet)
        print("")

        # --- 4. 计算正确答案 ---
        oya_tsumo, ko_tsumo, oya_ron, ko_ron = generate_all_possible_points(
            kiriage_mangan=rules.get("kiriage_mangan", True),
            allow_double_yakuman=rules.get("allow_double_yakuman", True),
            allow_composite_yakuman=rules.get("allow_composite_yakuman", True)
        )
        question_str = ""
        q_type = scenario["question_type"]
        player = scenario["player_to_ask"]
        goal = scenario["goal_placement"]
        player_wind = WIND_MAP[player]
        pts_list = []
        ok_end, ok_continue = [], []

        if q_type == "tsumo":
            question_str = f"问题: 选手 {player} ({player_wind}) 为获得前 {goal} 名，其自摸条件是什么？"
            pts_list = oya_tsumo if player == scenario["oya"] else ko_tsumo
            ok_end, ok_continue = calculation.calculate_tsumo_conditions(
                player, scenario["oya"], scenario["pts_in_hand"], scenario["honba"], scenario["deposit"],
                scenario["riichi_status"], all_pts, goal, tiebreaker, rules, pts_list)
        elif q_type.startswith("ron_"):
            target_player = int(q_type.split('_')[1])
            target_wind = WIND_MAP[target_player]
            question_str = f"问题: 选手 {player} ({player_wind}) 为获得前 {goal} 名，其荣和选手 {target_player} ({target_wind}) 的条件是什么？"
            pts_list = oya_ron if player == scenario["oya"] else ko_ron
            ok_end, ok_continue = calculation.calculate_ron_conditions(
                player, target_player, scenario["oya"], scenario["pts_in_hand"], scenario["honba"],
                scenario["deposit"], scenario["riichi_status"], all_pts, goal, tiebreaker, rules, pts_list)

        # 格式化答案
        end_str = format_as_intervals(ok_end, pts_list)
        continue_str = format_as_intervals(ok_continue, pts_list)
        merged_ok_pts = sorted(list(set(ok_end) | set(ok_continue)))
        merged_correct_str = format_as_intervals(merged_ok_pts, pts_list)

        # BUG FIX: 为亲家自摸的答案补上 " all"
        is_tsumo = scenario["question_type"] == "tsumo"
        is_oya = scenario["player_to_ask"] == scenario["oya"]
        if is_tsumo and is_oya:
            if end_str and end_str not in ["〇", "✕"]: end_str += " all"
            if continue_str and continue_str not in ["〇", "✕"]: continue_str += " all"
            if merged_correct_str and merged_correct_str not in ["〇", "✕"]: merged_correct_str += " all"

        is_end_valid = end_str and end_str != '✕'
        is_continue_valid = continue_str and continue_str != '✕'
        if not is_end_valid and not is_continue_valid:
            correct_answer_str_full = "✕"
        elif is_end_valid and not is_continue_valid:
            correct_answer_str_full = end_str
        elif not is_end_valid and is_continue_valid:
            correct_answer_str_full = f"{continue_str} (续行)"
        else:
            correct_answer_str_full = f"{end_str} (终局) / {continue_str} (续行)"

        # --- 5. 获取学生答案 ---
        question_start_time = time.time()
        print(question_str, flush=True)
        student_answer_raw = input("你的答案: ")
        question_end_time = time.time()
        time_taken = question_end_time - question_start_time

        print(f"本题用时: {time_taken:.2f} 秒", flush=True)

        # 显示剩余时间或超时
        time_limit_seconds = 30 * 60
        elapsed_time = question_end_time - exam_start_time
        remaining_time = time_limit_seconds - elapsed_time

        if remaining_time >= 0:
            rem_minutes, rem_seconds = divmod(int(remaining_time), 60)
            print(f"剩余时间: {rem_minutes} 分 {rem_seconds} 秒", flush=True)
        else:
            overtime_so_far = -remaining_time
            over_minutes, over_seconds = divmod(int(overtime_so_far), 60)
            print(f"已超时: {over_minutes} 分 {over_seconds} 秒", flush=True)

        # --- 6. 评估答案 ---
        correct_pts_set = parse_answer_to_pts(merged_correct_str, pts_list)
        student_pts_set = parse_answer_to_pts(student_answer_raw, pts_list)

        student_input_norm = student_answer_raw.strip().lower().replace(" ", "")
        ok_aliases = ['all', '0', 'o', 'ok', '〇', '']
        no_aliases = ['none', 'x', 'no', '✕']
        is_student_ok = student_input_norm in ok_aliases
        is_correct_ok = (merged_correct_str == "〇")
        is_student_no = student_input_norm in no_aliases
        is_correct_no = (merged_correct_str == "✕")

        is_correct = False
        if is_student_ok and is_correct_ok:
            is_correct = True
        elif is_student_no and is_correct_no:
            is_correct = True
        elif not is_correct_ok and not is_correct_no and correct_pts_set == student_pts_set:
            is_correct = True

        # --- 7. 保存结果 ---
        exam_results.append({
            "question_str": question_str,
            "student_answer_raw": student_answer_raw,
            "correct_answer_str": correct_answer_str_full,
            "is_correct": is_correct,
            "difficulty": q_info['level'],
            "time_taken": time_taken
        })

    # --- 8. 公布结果 ---
    total_exam_time = time.time() - exam_start_time
    _display_exam_results(exam_results, exam_seed, total_exam_time)


if __name__ == "__main__":
    while True:
        print("\n请选择练习模式:")
        print("1. 普通练习")
        print("2. 困难模式 (至少一个'或')")
        print("3. 魔鬼模式 (至少两个'或')")
        print("4. 考试模式 (预计用时 30 分钟)")
        print("5. 使用种子复盘考试")
        print("0. 退出")
        if not os.path.exists(QBANK_FILE):
            print("99. (首次使用)生成魔鬼模式题库")
        choice = input("你的选择: ")

        if choice == '0':
            break
        elif choice == '1':
            start_quiz(min_or_count=0)
        elif choice == '2':
            start_quiz(min_or_count=1)
        elif choice == '3':
            start_quiz(min_or_count=MIN_OR_COUNT_FOR_DEVIL)
        elif choice == '4':
            start_exam_mode()
        elif choice == '5':
            try:
                seed_input = int(input("请输入考试种子: ").strip())
                start_exam_mode(seed=seed_input)
            except ValueError:
                print("无效的种子，请输入一个数字。")
        elif choice == '99':
            populate_initial_qbank()
        else:
            print("无效输入，请重新选择。")
