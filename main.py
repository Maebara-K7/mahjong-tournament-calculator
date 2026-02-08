from cond_lib import *
from rulesets import RULES
import calculation


def _print_conditions(base_str, end_str, continue_str):
    # 根据规则简化打印终局和续行条件。
    # 特殊规则：仅当流局条件的终局和续行都为'〇'时，合并它们。
    if "流局" in base_str and end_str == '〇' and continue_str == '〇':
        print(f"{base_str}: 〇")
        return

    is_end_valid = end_str and end_str != '✕'
    is_continue_valid = continue_str and continue_str != '✕'

    if not is_end_valid and not is_continue_valid:
        # 两个都无效，合并打印
        print(f"{base_str}: ✕")
    elif is_end_valid and not is_continue_valid:
        # 只有终局有效，只打印终局，并省略(终局)标识
        print(f"{base_str}: {end_str}")
    elif not is_end_valid and is_continue_valid:
        # 只有续行有效，只打印续行
        print(f"{base_str}[续行]: {continue_str}")
    else:
        # 两个都有效，分别打印
        print(f"{base_str}(终局): {end_str}")
        print(f"{base_str}[续行]: {continue_str}")


def play(start_pts, pts_in_hand, ruleset, deposit, riichi_status, goal_placement,
         oya=3, tiebreaker=None, other_players_pts=None):
    # --- 1. 初始化和输入验证 ---
    if other_players_pts is None:
        other_players_pts = []
    # 将场上四人分数与场外玩家分数合并，用于后续排名计算
    all_pts = start_pts + other_players_pts

    if tiebreaker is None:
        # 如果没有提供tiebreaker，则根据所有玩家的当前分数生成
        # "先行有利" tiebreaker: 分数高的排名靠前，同分则看初始顺序
        sorted_players = sorted(enumerate(all_pts), key=lambda x: (-x[1], x[0]))
        tiebreaker = [0] * len(all_pts)
        for rank, (player_index, _) in enumerate(sorted_players):
            tiebreaker[player_index] = rank

    if not ruleset.get("continue_on_negative_score", True):
        # 检查是否有玩家点数为负且规则不允许继续
        if any(p < 0 for p in pts_in_hand):
            raise ValueError("错误：有玩家点数为负，且规则不允许在此情况下继续。")

    # 点数检查逻辑
    starting = ruleset["starting_pts"]
    expected_total = 4 * starting
    actual_total = sum(pts_in_hand) + 1000 * (sum(riichi_status) + deposit)
    if expected_total != actual_total:
        difference = actual_total - expected_total
        num_riichi = sum(riichi_status)

        # 检查点数差异是否正好等于立直棒总数
        if num_riichi > 0 and difference == 1000 * num_riichi:
            if not ruleset.get("continue_on_negative_score", True):
                # 检查是否有玩家分数低于1000点但已经立直
                for i in range(4):
                    if riichi_status[i] == 1 and pts_in_hand[i] < 1000:
                        raise ValueError(f"错误: 玩家 {i} 已立直，但其手牌点数 ({pts_in_hand[i]}) 低于1000点。")

            prompt = (
                f"\n点数检查失败，总点数多了 {difference} 点。\n"
                f"这正好是 {num_riichi} 位玩家的立直棒点数。\n"
                f"是否自动从立直玩家的手牌点数中各扣除1000点并继续计算？ (y/n): "
            )
            choice = input(prompt).lower()
            if choice == 'y':
                print("已自动修正手牌点数。")
                corrected_pts_in_hand = pts_in_hand.copy()
                for i in range(len(corrected_pts_in_hand)):
                    if riichi_status[i] == 1:
                        corrected_pts_in_hand[i] -= 1000
                pts_in_hand = corrected_pts_in_hand
            else:
                error_message = (
                    f"点数检查失败！用户拒绝自动修正。\n"
                    f"期望总点数 (starting * 4): {expected_total}\n"
                    f"实际总点数 (手牌点数 + 供托 + 立直棒): {actual_total}\n"
                    f"差额: {difference}"
                )
                raise ValueError(error_message)
        else:
            error_message = (
                f"点数检查失败！\n"
                f"期望总点数 (starting * 4): {expected_total}\n"
                f"实际总点数 (手牌点数 + 供托 + 立直棒): {actual_total}\n"
                f"差额: {difference}"
            )
            raise ValueError(error_message)

    honba_input = input("请输入本场数: ")
    honba = int(honba_input) if honba_input else 0

    oya_tsumo, ko_tsumo, oya_ron, ko_ron = generate_all_possible_points(
        kiriage_mangan=ruleset.get("kiriage_mangan", True),
        allow_double_yakuman=ruleset.get("allow_double_yakuman", True),
        allow_composite_yakuman=ruleset.get("allow_composite_yakuman", True)
    )

    current_pts = [round(u + v, 1) for u, v in zip(start_pts, game_pts(
        pts_in_hand, ruleset['placement_pts'], ruleset['starting_pts'],
        deposit_final_draw_recipient="unclaimed"))] + other_players_pts
    print(f"当前得分: {current_pts}")
    current_rank = final_ranking(current_pts, tiebreaker)
    print(f"当前顺位: {current_rank}")
    print()

    # --- 2. 循环为每个玩家计算和打印 ---
    for player in range(4):
        print(f"----- 玩家 {player} 胜利条件 (目标: 前{goal_placement}) -----")

        # 自摸条件
        tsumo_pts = oya_tsumo if player == oya else ko_tsumo
        ok_tsumo_end, ok_tsumo_continue = calculation.calculate_tsumo_conditions(
            player, oya, pts_in_hand, honba, deposit, riichi_status,
            all_pts, goal_placement, tiebreaker, ruleset, tsumo_pts)

        tsumo_end_str = format_as_intervals(ok_tsumo_end, tsumo_pts)
        if player == oya and tsumo_end_str != "✕" and tsumo_end_str != "〇":
            tsumo_end_str = f"{tsumo_end_str} all"

        tsumo_continue_str = format_as_intervals(ok_tsumo_continue, tsumo_pts)
        if player == oya and tsumo_continue_str != "✕" and tsumo_continue_str != "〇":
            tsumo_continue_str = f"{tsumo_continue_str} all"

        _print_conditions(f"玩家{player}自摸条件", tsumo_end_str, tsumo_continue_str)

        # 荣和条件
        for player2 in range(4):
            if player == player2: continue
            ron_pts = oya_ron if player == oya else ko_ron
            ok_ron_end, ok_ron_continue = calculation.calculate_ron_conditions(
                player, player2, oya, pts_in_hand, honba, deposit,
                riichi_status, all_pts, goal_placement, tiebreaker, ruleset,
                ron_pts)

            ron_end_str = format_as_intervals(ok_ron_end, ron_pts)
            ron_continue_str = format_as_intervals(ok_ron_continue, ron_pts)

            _print_conditions(f"玩家{player}荣和玩家{player2}条件", ron_end_str, ron_continue_str)

        # 被动荣和（放铳）条件
        for player2 in range(4):
            if player == player2: continue
            lose_pts = oya_ron if player2 == oya else ko_ron
            ok_passive_ron = calculation.calculate_passive_ron_conditions(player, player2, oya, pts_in_hand, honba,
                                                                          deposit, riichi_status, all_pts,
                                                                          goal_placement, tiebreaker, ruleset, lose_pts)
            if ok_passive_ron or current_rank[player] <= goal_placement:
                print(f"玩家{player}被玩家{player2}荣和条件: {format_as_intervals(ok_passive_ron, lose_pts)}")

        # 被动自摸条件
        for player2 in range(4):
            if player == player2: continue
            lose_pts = oya_tsumo if player2 == oya else ko_tsumo
            ok_passive_tsumo = calculation.calculate_passive_tsumo_conditions(player, player2, oya, pts_in_hand, honba,
                                                                              deposit, riichi_status, all_pts,
                                                                              goal_placement, tiebreaker, ruleset,
                                                                              lose_pts)
            if ok_passive_tsumo:
                print(f"玩家{player}被玩家{player2}自摸条件: {format_as_intervals(ok_passive_tsumo, lose_pts)}")

        # 流局条件
        draw_results = calculation.calculate_draw_conditions(player, oya, riichi_status, pts_in_hand, honba, all_pts,
                                                             goal_placement, tiebreaker, ruleset)

        # 调用原始函数获取完整字符串
        full_draw_end_str = format_draw_condition_string(player, draw_results["tenpai_ok_end"],
                                                         draw_results["noten_ok_end"],
                                                         draw_results["all_cases_end"], "(终局)")
        # 从 "玩家X流局条件(终局): " 后提取核心描述
        draw_end_str = full_draw_end_str.split(': ', 1)[1] if full_draw_end_str else None

        # 调用原始函数获取完整字符串
        full_draw_continue_str = format_draw_condition_string(player, draw_results["tenpai_ok_continue"],
                                                              draw_results["noten_ok_continue"],
                                                              draw_results["all_cases_continue"], "[续行]")
        # 从 "玩家X流局条件[续行]: " 后提取核心描述
        draw_continue_str = full_draw_continue_str.split(': ', 1)[1] if full_draw_continue_str else None

        _print_conditions(f"玩家{player}流局条件", draw_end_str, draw_continue_str)

        # 新增：分析立直对流局条件的影响
        if riichi_status[player] == 0:
            riichi_impact_str = calculation.analyze_riichi_impact_on_draw(
                player, oya, riichi_status, pts_in_hand, honba, all_pts,
                goal_placement, tiebreaker, ruleset)
            if riichi_impact_str is not None:
                print(riichi_impact_str)

        if player < 3: print()


if __name__ == "__main__":
    # --- 输入模板 ---
    ruleset_name = "SK"
    rules = RULES[ruleset_name]
    start_pts = [-5.0, -27.0, -22.8, 54.8]
    other_players_pts = []
    pts_in_hand = [45000, 25500, 43300, 6200]
    tiebreaker = None
    deposit = 0
    riichi_status = [0, 0, 0, 0]
    goal_placement = 2
    oya = 3

    print(f"--- 使用规则: {rules['name']} ---")
    play(start_pts, pts_in_hand, rules,
         deposit=deposit, riichi_status=riichi_status, goal_placement=goal_placement,
         oya=oya, tiebreaker=tiebreaker, other_players_pts=other_players_pts)