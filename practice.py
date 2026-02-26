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


def generate_random_scenario():
    """生成一个随机的对局情景和问题"""

    # 1. 随机选择规则 (排除 "CUS")
    available_rules = {k: v for k, v in RULES.items() if k != "CUS"}
    ruleset_name = random.choice(list(available_rules.keys()))
    rules = available_rules[ruleset_name]

    # 2. 随机生成赛前积分
    start_pts = [round(random.uniform(-60.0, 60.0), 1) for _ in range(4)]

    # 3. 随机生成局内点数 (使用更均匀的算法)
    total_points = rules["starting_pts"] * 4

    # 生成四个随机权重
    weights = [random.random() for _ in range(4)]
    total_weight = sum(weights)

    # 根据权重分配点数
    unrounded_pts = [(w / total_weight) * total_points for w in weights]

    # 四舍五入到100的倍数，并处理总和偏差
    pts_in_hand = [round(p / 100) * 100 for p in unrounded_pts]
    current_sum = sum(pts_in_hand)
    diff = total_points - current_sum

    # 将偏差加到随机一个玩家身上
    pts_in_hand[random.randint(0, 3)] += diff

    # 4. 随机生成其他参数
    oya = 3
    honba = random.randint(0, 4)
    deposit = random.randint(0, 4) if honba != 0 else 0
    riichi_status = [0] * 4

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
    print("--- 麻将终局条件计算练习 ---")
    print(f"规则: {rules['name']}")
    print(f"赛前积分: {scenario['start_pts']}")
    print(f"当前局内点数: {scenario['pts_in_hand']}")
    print(f"庄家: 玩家 {scenario['oya']}")
    print(f"场供: {scenario['deposit'] * 1000} 点, 本场: {scenario['honba']}, 立直棒: {sum(scenario['riichi_status'])}")
    print("-" * 30)

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

    if q_type == "tsumo":
        question_str = f"问题: 玩家 {player} 为获得前 {goal} 名，其自摸条件是什么？"
        pts_list = oya_tsumo if player == scenario["oya"] else ko_tsumo
        ok_end, ok_continue = calculation.calculate_tsumo_conditions(
            player, scenario["oya"], scenario["pts_in_hand"], scenario["honba"], scenario["deposit"],
            scenario["riichi_status"], all_pts, goal, tiebreaker, rules, pts_list)

    elif q_type.startswith("ron_"):
        target_player = int(q_type.split('_')[1])
        question_str = f"问题: 玩家 {player} 为获得前 {goal} 名，其荣和玩家 {target_player} 的条件是什么？"
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

    # --- 4. 向学生提问并获取答案 ---
    print(question_str)
    print("(提示: 请按程序输出的格式作答，例如 '>= 8000'。若答案为'〇'，可直接回车)")
    student_answer = input("你的答案: ")

    # --- 5. 评判并给出反馈 ---
    print("-" * 30)
    print(f"正确答案是: {correct_answer_str}")

    # 更智能的答案评判逻辑
    student_input = student_answer.strip().lower()
    correct_norm = correct_answer_str.strip().lower().replace(" ", "")
    merged_correct_norm = merged_correct_str.strip().lower().replace(" ", "")

    # 定义别名，将空输入也视作 "〇"
    ok_aliases = ['all', '0', 'o', 'ok', '〇', '']
    no_aliases = ['none', 'x', 'no', '✕']

    # 判断学生输入和正确答案是否属于“全部成功”或“全部失败”的类别
    is_student_ok = student_input in ok_aliases
    is_correct_ok = correct_norm in ok_aliases or merged_correct_norm in ok_aliases

    is_student_no = student_input in no_aliases
    is_correct_no = correct_norm in no_aliases or merged_correct_norm in no_aliases

    # 比较学生的答案和正确答案
    is_correct = False
    student_input_norm = student_input.replace(" ", "")

    # 1. 两者都属于“全部成功”的别名
    if is_student_ok and is_correct_ok:
        is_correct = True
    # 2. 两者都属于“全部失败”的别名
    elif is_student_no and is_correct_no:
        is_correct = True
    # 3. 学生的输入与详细答案或合并答案中的任意一个匹配
    elif student_input_norm == correct_norm or student_input_norm == merged_correct_norm:
        is_correct = True
    # 4. 特殊情况：当正确答案为"〇"时，检查学生是否输入了等价的 ">= 最低分"
    elif is_correct_ok:
        min_score = pts_list[0]
        is_tsumo = scenario["question_type"] == "tsumo"
        is_oya = scenario["player_to_ask"] == scenario["oya"]

        # 根据不同情况生成所有可能的正确答案格式
        expected_formats = []
        if is_tsumo:
            if is_oya:
                # 庄家自摸, e.g., 500 -> ">=500", ">=500all"
                expected_formats.append(f">={min_score}")
                expected_formats.append(f">={min_score}all")
            else:
                # 子家自摸, e.g., (300, 500) -> ">=300/500", ">=300-500"
                ko_pay, oya_pay = min_score
                expected_formats.append(f">={ko_pay}/{oya_pay}")
                expected_formats.append(f">={ko_pay}-{oya_pay}")
        else:  # 荣和
            # 荣和, e.g., 1000 -> ">=1000"
            expected_formats.append(f">={min_score}")

        if student_input_norm in expected_formats:
            is_correct = True

    if is_correct:
        print("\n恭喜你，完全正确！你已经掌握了！")
    else:
        print("\n答案不完全正确，请对照正确答案，检查你的计算过程。")


if __name__ == "__main__":
    start_quiz()