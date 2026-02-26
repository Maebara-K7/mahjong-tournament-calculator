# practice.py
"""
麻将终局条件计算练习程序

这个程序会自动生成一个随机的麻将终局场景，
并要求你计算在某种情况下的和牌条件。
"""

import random
from cond_lib import *
from rulesets import RULES
import calculation


def _get_high_fu_cheat_sheet(player_to_ask, oya, question_type, rules):
    """Generates a cheat sheet for high-fu points (70, 90, 110)."""
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
    Normalizes the student's answer string to handle various separators for discontinuous conditions,
    while correctly ignoring commas within parentheses (e.g., in tsumo point formats).
    """
    # First, normalize unambiguous separators to the canonical '或'.
    # We handle ' or ' with spaces, and other common separators.
    norm_str = answer_str.strip().lower()
    norm_str = norm_str.replace(' or ', '或')
    norm_str = norm_str.replace('||', '或')
    norm_str = norm_str.replace('|', '或')
    norm_str = norm_str.replace(';', '或')

    # Remove all spaces for consistent comparison
    norm_str = norm_str.replace(' ', '')
    # Also replace 'or' without spaces, which might result from space removal
    norm_str = norm_str.replace('or', '或')

    # Manually iterate to replace commas only when they are outside of parentheses
    result = []
    paren_level = 0
    for char in norm_str:
        if char == '(':
            paren_level += 1
        elif char == ')':
            paren_level = max(0, paren_level - 1)  # Avoid going negative

        if char == ',' and paren_level == 0:
            result.append('或')
        else:
            result.append(char)

    return "".join(result)


def parse_answer_to_pts(answer_str, all_possible_pts):
    """解析答案字符串，返回一个包含所有对应点数的集合。"""
    if not all_possible_pts:
        return set()

    # Handle special 'all' or 'ok' answers
    norm_for_alias = answer_str.strip().lower().replace(" ", "")
    ok_aliases = ['all', '0', 'o', 'ok', '〇', '']
    if norm_for_alias in ok_aliases:
        return set(all_possible_pts)

    # 标准化并分割
    normalized_str = normalize_student_answer(answer_str)
    parts = normalized_str.split('或')

    final_pts = set()

    for part in parts:
        part = part.strip()
        if not part:
            continue

        # 处理 >=, >, <=, <
        try:
            if part.startswith('>='):
                val = int(part[2:])
                final_pts.update({p for p in all_possible_pts if p >= val})
                continue
            elif part.startswith('<='):
                val = int(part[2:])
                final_pts.update({p for p in all_possible_pts if p <= val})
                continue
            elif part.startswith('>'):
                val = int(part[1:])
                final_pts.update({p for p in all_possible_pts if p > val})
                continue
            elif part.startswith('<'):
                val = int(part[1:])
                final_pts.update({p for p in all_possible_pts if p < val})
                continue
        except (ValueError, IndexError):
            continue  # Ignore malformed comparisons

        # 处理范围 ～
        if '～' in part:
            try:
                start_val, end_val = map(int, part.split('～'))
                final_pts.update({p for p in all_possible_pts if start_val <= p <= end_val})
            except ValueError:
                continue  # 忽略格式错误的范围
        # 处理单个数字
        else:
            try:
                # 移除可能的前导'='
                cleaned_part = part.lstrip('=')
                val = int(cleaned_part)
                if val in all_possible_pts:
                    final_pts.add(val)
            except ValueError:
                continue  # 忽略无法解析的部分

    return final_pts


def generate_random_scenario():
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
    honba = random.randint(0, 4)
    deposit = random.randint(0, 4) if honba != 0 else 0
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

    # 将偏差加到随机一个玩家身上
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

    # 选择一个非第一名的玩家进行提问
    eligible_players = [i for i, rank in enumerate(current_rank) if rank > 1]
    if not eligible_players:  # 如果所有人都并列第一，则随机选一个
        eligible_players = list(range(4))
    player_to_ask = random.choice(eligible_players)
    player_rank = current_rank[player_to_ask]

    # 根据玩家当前排名设定一个有挑战性的目标
    if player_rank == 2:
        goal_placement = 1
    elif player_rank == 3:
        goal_placement = random.choice([1, 2])
    else:  # player_rank == 4 or tied
        goal_placement = random.choice([2, 3])

    # 随机选择问题类型 (自摸或荣和)
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
        "tiebreaker": tiebreaker
    }


def start_quiz():
    """开始一个练习会话"""

    # --- 1. 场景生成 ---
    scenario = generate_random_scenario()
    rules = scenario["rules"]
    all_pts = scenario["start_pts"] + scenario["other_players_pts"]
    tiebreaker = scenario["tiebreaker"]

    # --- 2. 打印场景和问题 ---
    WIND_MAP = {0: "东", 1: "南", 2: "西", 3: "北"}
    oya_wind = WIND_MAP[scenario["oya"]]

    print("--- 麻将终局条件计算练习 ---")
    print(f"规则: {rules['name']}")
    print(f"顺位马: {rules['placement_pts']}")
    print(f"赛前积分: {scenario['start_pts']}")
    print(f"当前局内点数: {scenario['pts_in_hand']}")
    print(f"庄家: 玩家 {scenario['oya']} ({oya_wind})")
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
        question_str = f"问题: 玩家 {player} ({player_wind}) 为获得前 {goal} 名，其自摸条件是什么？"
        pts_list = oya_tsumo if player == scenario["oya"] else ko_tsumo
        ok_end, ok_continue = calculation.calculate_tsumo_conditions(
            player, scenario["oya"], scenario["pts_in_hand"], scenario["honba"], scenario["deposit"],
            scenario["riichi_status"], all_pts, goal, tiebreaker, rules, pts_list)

    elif q_type.startswith("ron_"):
        target_player = int(q_type.split('_')[1])
        target_wind = WIND_MAP[target_player]
        question_str = f"问题: 玩家 {player} ({player_wind}) 为获得前 {goal} 名，其荣和玩家 {target_player} ({target_wind}) 的条件是什么？"
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
    student_answer = input("你的答案: ")

    # --- 5. 评判并给出反馈 ---
    print("-" * 30)
    print(f"正确答案是: {correct_answer_str}")

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
    # For non-special cases, compare the parsed point sets
    elif not is_correct_ok and not is_correct_no and correct_pts_set == student_pts_set:
        is_correct = True

    if is_correct:
        print("\n恭喜你，完全正确！你已经掌握了！")
    else:
        print("\n答案不完全正确，请对照正确答案，检查你的计算过程。")


def start_hard_quiz():
    """开始一个“真剣勝負”练习会话，专门寻找难题。"""
    print("\n--- 真剣勝負模式 ---")
    print("正在生成题目，请稍候...")

    found_scenario = False
    scenario_data = {}

    while not found_scenario:
        # 1. 场景生成
        scenario = generate_random_scenario()
        rules = scenario["rules"]
        all_pts = scenario["start_pts"] + scenario["other_players_pts"]
        tiebreaker = scenario["tiebreaker"]

        # 2. 计算答案，检查是否为不连续条件
        oya_tsumo, ko_tsumo, oya_ron, ko_ron = generate_all_possible_points(
            kiriage_mangan=rules.get("kiriage_mangan", True),
            allow_double_yakuman=rules.get("allow_double_yakuman", True),
            allow_composite_yakuman=rules.get("allow_composite_yakuman", True)
        )

        q_type = scenario["question_type"]
        player = scenario["player_to_ask"]
        goal = scenario["goal_placement"]
        ok_end, ok_continue, pts_list = [], [], []

        if q_type == "tsumo":
            pts_list = oya_tsumo if player == scenario["oya"] else ko_tsumo
            ok_end, ok_continue = calculation.calculate_tsumo_conditions(
                player, scenario["oya"], scenario["pts_in_hand"], scenario["honba"], scenario["deposit"],
                scenario["riichi_status"], all_pts, goal, tiebreaker, rules, pts_list)
        elif q_type.startswith("ron_"):
            target_player = int(q_type.split('_')[1])
            pts_list = oya_ron if player == scenario["oya"] else ko_ron
            ok_end, ok_continue = calculation.calculate_ron_conditions(
                player, target_player, scenario["oya"], scenario["pts_in_hand"], scenario["honba"],
                scenario["deposit"], scenario["riichi_status"], all_pts, goal, tiebreaker, rules, pts_list)

        merged_ok_pts = sorted(list(set(ok_end) | set(ok_continue)))
        formatted_merged_str = format_as_intervals(merged_ok_pts, pts_list)

        # 检查条件是否是不连续的 (包含 "或" 字)
        if "或" in formatted_merged_str:
            found_scenario = True
            scenario_data = {
                "scenario": scenario,
                "ok_end": ok_end,
                "ok_continue": ok_continue,
                "pts_list": pts_list
            }

    # --- 3. 提取找到的场景和答案 ---
    scenario = scenario_data["scenario"]
    rules = scenario["rules"]
    ok_end = scenario_data["ok_end"]
    ok_continue = scenario_data["ok_continue"]
    pts_list = scenario_data["pts_list"]
    player = scenario["player_to_ask"]
    goal = scenario["goal_placement"]
    q_type = scenario["question_type"]

    # --- 4. 打印场景和问题 ---
    WIND_MAP = {0: "东", 1: "南", 2: "西", 3: "北"}
    oya_wind = WIND_MAP[scenario["oya"]]

    print("--- 麻将终局条件计算练习 ---")
    print(f"规则: {rules['name']}")
    print(f"顺位马: {rules['placement_pts']}")
    print(f"赛前积分: {scenario['start_pts']}")
    print(f"当前局内点数: {scenario['pts_in_hand']}")
    print(f"庄家: 玩家 {scenario['oya']} ({oya_wind})")
    print(f"场供: {scenario['deposit'] * 1000} 点, 本场: {scenario['honba']}")
    print("-" * 30)

    cheat_sheet = _get_high_fu_cheat_sheet(
        scenario["player_to_ask"], scenario["oya"], scenario["question_type"], rules
    )
    print(cheat_sheet)
    print("")

    # 构造问题字符串
    player_wind = WIND_MAP[player]
    question_str = ""
    if q_type == "tsumo":
        question_str = f"问题: 玩家 {player} ({player_wind}) 为获得前 {goal} 名，其自摸条件是什么？"
    elif q_type.startswith("ron_"):
        target_player = int(q_type.split('_')[1])
        target_wind = WIND_MAP[target_player]
        question_str = f"问题: 玩家 {player} ({player_wind}) 为获得前 {goal} 名，其荣和玩家 {target_player} ({target_wind}) 的条件是什么？"

    # --- 5. 格式化正确答案 ---
    end_str = format_as_intervals(ok_end, pts_list)
    continue_str = format_as_intervals(ok_continue, pts_list)
    merged_correct_str = format_as_intervals(sorted(list(set(ok_end) | set(ok_continue))), pts_list)

    is_tsumo = scenario["question_type"] == "tsumo"
    is_oya = scenario["player_to_ask"] == scenario["oya"]
    if is_tsumo and is_oya:
        if end_str and end_str not in ["〇", "✕"]: end_str += " all"
        if continue_str and continue_str not in ["〇", "✕"]: continue_str += " all"
        if merged_correct_str and merged_correct_str not in ["〇", "✕"]: merged_correct_str += " all"

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
    print("(点数闭区间请使用 波浪线(～) 分隔，例如: 3400～3900, >=24000)")
    student_answer = input("你的答案: ")

    # --- 7. 评判并给出反馈 ---
    print("-" * 30)
    print(f"正确答案是: {correct_answer_str}")

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
    # For non-special cases, compare the parsed point sets
    elif not is_correct_ok and not is_correct_no and correct_pts_set == student_pts_set:
        is_correct = True

    if is_correct:
        print("\n恭喜你，完全正确！你已经掌握了！")
    else:
        print("\n答案不完全正确，请对照正确答案，检查你的计算过程。")


if __name__ == "__main__":
    while True:
        print("\n请选择练习模式:")
        print("1. 普通模式 (随机生成问题)")
        print("2. 真剣勝負模式 (专门寻找难题)")
        print("q. 退出")
        choice = input("请输入选项: ").strip()

        if choice == '1':
            start_quiz()
        elif choice == '2':
            start_hard_quiz()
        elif choice.lower() == 'q':
            break
        else:
            print("无效输入，请重新选择。")
