from cond_lib import *


def calculate_tsumo_conditions(player, oya, pts_in_hand, honba, deposit, riichi_status, all_pts, goal_placement,
                               tiebreaker, ruleset, all_win_pts):
    """计算玩家自摸胜利的条件, 并区分终局和续行两种情况。"""
    # 分别记录两种情况下的胜利点数
    tsumo_wins_end = set()
    tsumo_wins_continue = set()
    is_oya = (player == oya)

    # 遍历所有可能的自摸得分组合
    for win_pts in all_win_pts:
        pts_copy = pts_in_hand.copy()
        # 根据是否为亲家，计算自摸后四位玩家的手牌点数变化
        if is_oya:
            x = win_pts
            for k in range(4):
                if k == player:
                    pts_copy[k] += 3 * x + 300 * honba + 1000 * (deposit + sum(riichi_status))
                else:
                    pts_copy[k] -= x + 100 * honba
        else:
            x, y = win_pts
            for k in range(4):
                if k == player:
                    pts_copy[k] += 2 * x + y + 300 * honba + 1000 * (deposit + sum(riichi_status))
                elif k != oya:
                    pts_copy[k] -= x + 100 * honba
                else:
                    pts_copy[k] -= y + 100 * honba

        # 计算包含顺位分的最终得分
        game = game_pts(pts_copy, ruleset["placement_pts"], ruleset["starting_pts"],
                        tie_resolution=ruleset.get("tie_resolution", "split_points"))

        # 将场上四人的最终得分与场外玩家的得分合并
        end_table = [u + v for u, v in zip(all_pts[:4], game)]
        end = end_table + all_pts[4:]

        # 根据所有人的最终得分计算排名
        final_rank = final_ranking(end, tiebreaker)

        # 检查玩家排名是否达到目标
        if final_rank[player] <= goal_placement:
            # 如果达到目标，再检查对局是否继续
            continues = check_continuation(pts_copy, ruleset, oya, oya_wins=(player == oya))
            if continues:
                tsumo_wins_continue.add(win_pts)
            else:
                tsumo_wins_end.add(win_pts)

    # --- 将离散的胜利点数集合格式化为区间列表 ---
    # 这个逻辑的作用是找到满足条件的最低分，并找出高于最低分但又不满足条件的"例外"点数
    # 这样 format_as_intervals 就能正确地将其显示为 "≥XXXX (不含 YYYY)" 的格式

    # 处理终局条件
    min_win_pts_end = None
    exception_pts_end = []
    for p in all_win_pts:
        if p in tsumo_wins_end:
            min_win_pts_end = p
            break
    if min_win_pts_end is not None:
        start_idx = all_win_pts.index(min_win_pts_end)
        for p in all_win_pts[start_idx:]:
            if p not in tsumo_wins_end:
                exception_pts_end.append(p)
    ok_tsumo_end = [p for p in all_win_pts[all_win_pts.index(min_win_pts_end):] if
                    p not in exception_pts_end] if min_win_pts_end is not None else []

    # 处理续行条件
    min_win_pts_continue = None
    exception_pts_continue = []
    for p in all_win_pts:
        if p in tsumo_wins_continue:
            min_win_pts_continue = p
            break
    if min_win_pts_continue is not None:
        start_idx = all_win_pts.index(min_win_pts_continue)
        for p in all_win_pts[start_idx:]:
            if p not in tsumo_wins_continue:
                exception_pts_continue.append(p)
    ok_tsumo_continue = [p for p in all_win_pts[all_win_pts.index(min_win_pts_continue):] if
                         p not in exception_pts_continue] if min_win_pts_continue is not None else []

    return ok_tsumo_end, ok_tsumo_continue


def calculate_ron_conditions(player, target_player, oya, pts_in_hand, honba, deposit, riichi_status, all_pts,
                             goal_placement, tiebreaker, ruleset, all_win_pts):
    """计算玩家荣和胜利的条件, 并区分终局和续行两种情况。"""
    ron_wins_end = set()
    ron_wins_continue = set()

    # 遍历所有可能的荣和得分
    for win_pts in all_win_pts:
        pts_copy = pts_in_hand.copy()
        # 计算荣和后, 和牌者与放铳者的手牌点数变化
        pts_copy[player] += win_pts + 300 * honba + 1000 * (deposit + sum(riichi_status))
        pts_copy[target_player] -= win_pts + 300 * honba

        # 计算包含顺位分的最终得分
        game = game_pts(pts_copy, ruleset["placement_pts"], ruleset["starting_pts"],
                        tie_resolution=ruleset.get("tie_resolution", "split_points"))

        # 将场上四人的最终得分与场外玩家的得分合并
        end_table = [u + v for u, v in zip(all_pts[:4], game)]
        end = end_table + all_pts[4:]

        # 根据所有人的最终得分计算排名
        final_rank = final_ranking(end, tiebreaker)

        # 检查玩家排名是否达到目标
        if final_rank[player] <= goal_placement:
            # 如果达到目标，再检查对局是否继续
            continues = check_continuation(pts_copy, ruleset, oya, oya_wins=(player == oya))
            if continues:
                ron_wins_continue.add(win_pts)
            else:
                ron_wins_end.add(win_pts)

    # --- 将离散的胜利点数集合格式化为区间列表 ---
    # 处理终局条件
    min_win_pts_end = None
    exception_pts_end = []
    for p in all_win_pts:
        if p in ron_wins_end:
            min_win_pts_end = p
            break
    if min_win_pts_end is not None:
        start_idx = all_win_pts.index(min_win_pts_end)
        for p in all_win_pts[start_idx:]:
            if p not in ron_wins_end:
                exception_pts_end.append(p)
    ok_ron_end = [p for p in all_win_pts[all_win_pts.index(min_win_pts_end):] if
                  p not in exception_pts_end] if min_win_pts_end is not None else []

    # 处理续行条件
    min_win_pts_continue = None
    exception_pts_continue = []
    for p in all_win_pts:
        if p in ron_wins_continue:
            min_win_pts_continue = p
            break
    if min_win_pts_continue is not None:
        start_idx = all_win_pts.index(min_win_pts_continue)
        for p in all_win_pts[start_idx:]:
            if p not in ron_wins_continue:
                exception_pts_continue.append(p)
    ok_ron_continue = [p for p in all_win_pts[all_win_pts.index(min_win_pts_continue):] if
                       p not in exception_pts_continue] if min_win_pts_continue is not None else []

    return ok_ron_end, ok_ron_continue


def calculate_passive_ron_conditions(player, dealer, oya, pts_in_hand, honba, deposit, riichi_status, all_pts,
                                     goal_placement, tiebreaker, ruleset, all_lose_pts):
    """计算玩家被荣和（放铳）的条件。"""
    max_lose_pts = None
    exception_pts = []

    for lose_pts in reversed(all_lose_pts):
        pts_copy = pts_in_hand.copy()
        pts_copy[dealer] += lose_pts + 300 * honba + 1000 * (deposit + sum(riichi_status))
        pts_copy[player] -= lose_pts + 300 * honba

        game = game_pts(pts_copy, ruleset["placement_pts"], ruleset["starting_pts"],
                        tie_resolution=ruleset.get("tie_resolution", "split_points"))
        end_table = [u + v for u, v in zip(all_pts[:4], game)]
        end = end_table + all_pts[4:]
        final_rank = final_ranking(end, tiebreaker)

        if final_rank[player] <= goal_placement:
            if max_lose_pts is None: max_lose_pts = lose_pts
        elif max_lose_pts is not None:
            exception_pts.append(lose_pts)

    if max_lose_pts is not None:
        end_idx = all_lose_pts.index(max_lose_pts)
        return [p for p in all_lose_pts[:end_idx + 1] if p not in exception_pts]
    return []


def calculate_passive_tsumo_conditions(player, dealer, oya, pts_in_hand, honba, deposit, riichi_status, all_pts,
                                       goal_placement, tiebreaker, ruleset, all_lose_pts):
    """计算玩家被自摸的条件。"""
    max_lose_pts = None
    exception_pts = []
    is_dealer_oya = (dealer == oya)

    for lose_pts in reversed(all_lose_pts):
        pts_copy = pts_in_hand.copy()
        if is_dealer_oya:
            x = lose_pts
            for k in range(4):
                if k == dealer:
                    pts_copy[k] += 3 * x + 300 * honba + 1000 * (deposit + sum(riichi_status))
                else:
                    pts_copy[k] -= x + 100 * honba
        else:
            x, y = lose_pts
            for k in range(4):
                if k == dealer:
                    pts_copy[k] += 2 * x + y + 300 * honba + 1000 * (deposit + sum(riichi_status))
                elif k != oya:
                    pts_copy[k] -= x + 100 * honba
                else:
                    pts_copy[k] -= y + 100 * honba

        game = game_pts(pts_copy, ruleset["placement_pts"], ruleset["starting_pts"],
                        tie_resolution=ruleset.get("tie_resolution", "split_points"))
        end_table = [u + v for u, v in zip(all_pts[:4], game)]
        end = end_table + all_pts[4:]
        final_rank = final_ranking(end, tiebreaker)

        if final_rank[player] <= goal_placement:
            if max_lose_pts is None: max_lose_pts = lose_pts
        elif max_lose_pts is not None:
            exception_pts.append(lose_pts)

    if max_lose_pts is not None:
        end_idx = all_lose_pts.index(max_lose_pts)
        return [p for p in all_lose_pts[:end_idx + 1] if p not in exception_pts]
    return []


def calculate_draw_conditions(player, oya, riichi_status, pts_in_hand, honba, all_pts, goal_placement, tiebreaker,
                              ruleset):
    """计算玩家在流局情况下的条件。"""
    results = {
        "tenpai_ok_end": [], "noten_ok_end": [], "all_cases_end": [],
        "tenpai_ok_continue": [], "noten_ok_continue": [], "all_cases_continue": []
    }

    all_possible_cases = list_all_possible_tenpai_cases(riichi_status)

    for case in all_possible_cases:
        pts_copy = pts_in_hand.copy()
        tenpai_players = [i for i, x in enumerate(case) if x == 1]
        noten_players = [i for i, x in enumerate(case) if x == 0]

        if 0 < len(tenpai_players) < 4:
            payment = 3000 // len(noten_players)
            receipt = 3000 // len(tenpai_players)
            for p in tenpai_players:
                pts_copy[p] += receipt
            for p in noten_players:
                pts_copy[p] -= payment

        is_oya_tenpai = (case[oya] == 1)
        continues = check_continuation(pts_copy, ruleset, oya, oya_wins=False, oya_tenpai_in_draw=is_oya_tenpai)

        if continues:
            results["all_cases_continue"].append(case)
        else:
            results["all_cases_end"].append(case)

        game = game_pts(pts_copy, ruleset["placement_pts"], ruleset["starting_pts"],
                        tie_resolution=ruleset.get("tie_resolution", "split_points"))
        end_table = [u + v for u, v in zip(all_pts[:4], game)]
        end = end_table + all_pts[4:]
        final_rank = final_ranking(end, tiebreaker)

        if final_rank[player] <= goal_placement:
            if continues:
                if case[player] == 1:
                    results["tenpai_ok_continue"].append(case)
                else:
                    results["noten_ok_continue"].append(case)
            else:
                if case[player] == 1:
                    results["tenpai_ok_end"].append(case)
                else:
                    results["noten_ok_end"].append(case)

    return results


def analyze_riichi_impact_on_draw(player, oya, riichi_status, pts_in_hand, honba, all_pts, goal_placement, tiebreaker,
                                  ruleset):
    """
    分析立直对玩家流局条件的影响。
    """
    # 1. 计算默听时的听牌成功组合和所有可能性
    damaten_results = calculate_draw_conditions(player, oya, riichi_status, pts_in_hand, honba, all_pts,
                                                goal_placement, tiebreaker, ruleset)
    damaten_ok_cases = set(map(tuple, damaten_results["tenpai_ok_end"] + damaten_results["tenpai_ok_continue"]))
    damaten_all_possible = damaten_results["all_cases_end"] + damaten_results["all_cases_continue"]

    # 2. 模拟立直情况
    riichi_pts_in_hand = pts_in_hand.copy()
    riichi_pts_in_hand[player] -= 1000

    riichi_riichi_status = riichi_status.copy()
    riichi_riichi_status[player] = 1

    # 3. 计算立直后的听牌成功组合和所有可能性
    riichi_results = calculate_draw_conditions(player, oya, riichi_riichi_status, riichi_pts_in_hand, honba, all_pts,
                                               goal_placement, tiebreaker, ruleset)
    riichi_ok_cases = set(map(tuple, riichi_results["tenpai_ok_end"] + riichi_results["tenpai_ok_continue"]))
    riichi_all_possible = riichi_results["all_cases_end"] + riichi_results["all_cases_continue"]

    # 4. 比较差异并格式化输出
    if damaten_ok_cases == riichi_ok_cases:
        return None

    removed_cases = damaten_ok_cases - riichi_ok_cases

    output_parts = ["立直后流局条件"]
    # 使用 format_tenpai_condition 格式化减少的组合
    formatted_removed = format_tenpai_condition(list(removed_cases), damaten_all_possible, player, True)
    # 去掉 "听牌: " 前缀
    summary = formatted_removed.split(': ', 1)[1]
    output_parts.append(f"减少: {summary}, ")

    # 使用 format_tenpai_condition 格式化剩余的组合
    formatted_all = format_tenpai_condition(list(riichi_ok_cases), riichi_all_possible, player, True)
    # 去掉 "听牌: " 前缀
    summary = formatted_all.split(': ', 1)[1]
    output_parts.append(f"剩余: {summary}")

    return "".join(output_parts)