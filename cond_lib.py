import math
from itertools import product


def ceil_to_hundred(n):
    return math.ceil(n / 100) * 100


def generate_all_possible_points(kiriage_mangan=True, allow_double_yakuman=False, allow_composite_yakuman=True):
    fan = [1, 2, 3, 4]

    fu = [20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110]
    exception = [(1, 20), (1, 25), (1, 110), (2, 25)]
    if kiriage_mangan:
        kiriage_exceptions = [(3, 60), (4, 30)]
        exception.extend(kiriage_exceptions)

    oya_tsumo = set()
    ko_tsumo = set()

    for i in fan:
        for j in fu:
            if (i, j) in exception:
                continue

            oya_pts = 2 * j * 2 ** (i + 2)
            if oya_pts >= 4000:
                break
            oya_pts = ceil_to_hundred(oya_pts)
            oya_tsumo.add(oya_pts)

            ko_pts = (j * 2 ** (i + 2), 2 * j * 2 ** (i + 2))
            ko_pts = (ceil_to_hundred(ko_pts[0]), ceil_to_hundred(ko_pts[1]))
            ko_tsumo.add(ko_pts)

    oya_tsumo = sorted(list(oya_tsumo))
    ko_tsumo = sorted(list(ko_tsumo))

    oya_tsumo.extend([4000, 6000, 8000, 12000])
    ko_tsumo.extend([(2000, 4000), (3000, 6000), (4000, 8000), (6000, 12000)])

    fu = [25, 30, 40, 50, 60, 70, 80, 90, 100, 110]
    exception = [(1, 25)]
    if kiriage_mangan:
        kiriage_exceptions = [(3, 60), (4, 30)]
        exception.extend(kiriage_exceptions)

    oya_ron = set()
    ko_ron = set()

    for i in fan:
        for j in fu:
            if (i, j) in exception:
                continue

            oya_pts = 6 * j * 2 ** (i + 2)
            if oya_pts >= 12000:
                break
            oya_pts = ceil_to_hundred(oya_pts)
            oya_ron.add(oya_pts)

            ko_pts = 4 * j * 2 ** (i + 2)
            ko_pts = ceil_to_hundred(ko_pts)
            ko_ron.add(ko_pts)

    oya_ron = sorted(list(oya_ron))
    ko_ron = sorted(list(ko_ron))
    oya_ron.extend([12000, 18000, 24000, 36000])
    ko_ron.extend([8000, 12000, 16000, 24000])

    # 役满点数
    max_yakuman_multiple = 1  # FF -> 1倍
    if allow_double_yakuman and allow_composite_yakuman:  # TT -> 6倍
        max_yakuman_multiple = 6
    elif allow_double_yakuman:  # TF -> 2倍
        max_yakuman_multiple = 2
    elif allow_composite_yakuman:  # FT -> 4倍
        max_yakuman_multiple = 4

    yakuman_multiples = range(1, max_yakuman_multiple + 1)

    for m in yakuman_multiples:
        # 亲家自摸: 16000点/倍
        oya_tsumo.append(16000 * m)
        # 子家自摸: 8000/16000点/倍
        ko_tsumo.append((8000 * m, 16000 * m))
        # 亲家荣和: 48000点/倍
        oya_ron.append(48000 * m)
        # 子家荣和: 32000点/倍
        ko_ron.append(32000 * m)

    # print(oya_tsumo)
    # print(ko_tsumo)
    # print(oya_ron)
    # print(ko_ron)

    return oya_tsumo, ko_tsumo, oya_ron, ko_ron


def game_pts(pts_in_hand, placement_pts, starting=25000, tie_resolution="split_points",
             deposit_final_draw_recipient="first_place"):
    # 生成玩家索引和对应点数的列表
    player_points = list(enumerate(pts_in_hand))
    # 按点数从高到低排序
    player_points_sorted = sorted(player_points, key=lambda x: x[1], reverse=True)

    rank_points = [0] * 4

    if tie_resolution == "seating":
        # 按座次决定顺位
        # 由于 sorted() 是稳定的，分数相同时，原始顺序（即座次）被保留。
        # 因此，直接遍历排序后的列表即可实现“同分按座次”的规则。
        for i in range(4):
            player_idx = player_points_sorted[i][0]
            rank_points[player_idx] = placement_pts[i]
    else:  # 默认为 "split_points"，使用您编写的原始逻辑
        # 分组，点数相同的玩家归为一组
        groups = []
        if player_points_sorted:
            current_group = [player_points_sorted[0]]
            for i in range(1, len(player_points_sorted)):
                if player_points_sorted[i][1] == player_points_sorted[i - 1][1]:
                    current_group.append(player_points_sorted[i])
                else:
                    groups.append(current_group)
                    current_group = [player_points_sorted[i]]
            groups.append(current_group)

        rank_index = 0
        for group in groups:
            group_size = len(group)
            total_points = sum(placement_pts[rank_index:rank_index + group_size])
            # 先计算每人基础分，向下取整到0.1的倍数
            base_point = int(total_points / group_size * 10) / 10
            # 计算剩余分数，单位为0.1
            remainder = int(round((total_points - base_point * group_size) * 10))
            # 按座次优先分配余数
            for idx, (player_idx, _) in enumerate(group):
                add_point = remainder * 0.1 if idx == 0 else 0.
                rank_points[player_idx] = base_point + add_point
            rank_index += group_size

    for i in range(4):
        rank_points[i] += (pts_in_hand[i] - starting) / 1000

    # 计算供托数量
    deposit = (4 * starting - sum(pts_in_hand)) // 1000

    # 初始化供托分配列表
    deposit_distribution = [0.0, 0.0, 0.0, 0.0]

    # 仅当有供托且规则为“分配给第一名”时，才执行分配逻辑
    if deposit > 0 and deposit_final_draw_recipient == "first_place":
        # 计算第一名玩家的索引列表
        max_pts = max(pts_in_hand)
        first_players = [i for i, p in enumerate(pts_in_hand) if p == max_pts]

        if len(first_players) == 1:
            # 单独第一，全部供托给该玩家
            deposit_distribution[first_players[0]] = deposit
        elif len(first_players) == 3:
            # 三人并列第一，按座次顺序分配0.4, 0.3, 0.3
            distribution = [0.4, 0.3, 0.3]
            for idx, player in enumerate(first_players):
                deposit_distribution[player] = deposit * distribution[idx]
        else:
            # 其他情况（2人或4人并列），平均分配
            share = deposit / len(first_players)
            for player in first_players:
                deposit_distribution[player] = share

    total_points = [u + v for u, v in zip(rank_points, deposit_distribution)]

    total_points = [round(p, 1) for p in total_points]
    return total_points


def final_ranking(pts, tiebreaker):
    """
    根据分数和tiebreaker数组计算最终玩家排名。
    tiebreaker用于解决分数相同的情况，值越小，排名越高。
    """
    # 将每个玩家的原始索引、分数和tiebreaker值配对。
    pts = [round(p, 1) for p in pts]
    player_data = list(zip(range(len(pts)), pts, tiebreaker))

    # 使用多级排序键对玩家进行排序。
    # 1. 主排序：按分数降序。
    # 2. 次排序：按tiebreaker升序（通过使用负的tiebreaker值并整体降序排列来实现）。
    # 例如，对于同分的玩家，tiebreaker为1（-1）会排在tiebreaker为2（-2）的前面。
    sorted_players = sorted(player_data, key=lambda x: (x[1], -x[2]), reverse=True)

    # 根据排序后的顺序分配排名。
    ranks = [0] * len(pts)
    for rank, (original_index, score, tb) in enumerate(sorted_players, 1):
        ranks[original_index] = rank

    return ranks


def list_possible_tenpai_cases(riichi_status, oya=3):
    # 亲家听牌不能流局，若亲家听牌则无流局情况
    # 立直玩家一定听牌
    # 立直状态为0的玩家听牌状态不确定，需枚举

    tenpai_options = []
    for i in range(4):
        if i == oya:
            # 亲家不听牌
            if riichi_status[i] == 1:
                # 立直玩家必听牌
                tenpai_options.append([])
            else:
                tenpai_options.append([0])
        elif riichi_status[i] == 1:
            # 立直玩家必听牌
            tenpai_options.append([1])
        else:
            # 其他玩家可能听牌也可能不听牌
            tenpai_options.append([0, 1])

    # 枚举所有可能的听牌组合
    all_cases = list(product(*tenpai_options))
    return all_cases


def list_possible_oya_tenpai_cases(riichi_status, oya=3):
    # 这次列举亲家听牌的情况
    # 立直玩家一定听牌
    # 立直状态为0的玩家听牌状态不确定，需枚举

    tenpai_options = []
    for i in range(4):
        if i == oya or riichi_status[i] == 1:
            # 亲家听牌，立直玩家必听牌
            tenpai_options.append([1])
        else:
            # 其他玩家可能听牌也可能不听牌
            tenpai_options.append([0, 1])

    # 枚举所有可能的听牌组合
    all_cases = list(product(*tenpai_options))
    return all_cases


def list_all_possible_tenpai_cases(riichi_status):
    # 立直玩家一定听牌
    # 立直状态为0的玩家听牌状态不确定，需枚举

    tenpai_options = []
    for i in range(4):
        if riichi_status[i] == 1:
            # 立直玩家必听牌
            tenpai_options.append([1])
        else:
            # 其他玩家可能听牌也可能不听牌
            tenpai_options.append([0, 1])

    # 枚举所有可能的听牌组合
    all_cases = list(product(*tenpai_options))
    return all_cases


def format_as_intervals(ok_pts, all_pts):
    """将满足条件的点数列表格式化为区间字符串。"""
    if not ok_pts:
        return '✕'

    if ok_pts == all_pts:
        return '〇'

    intervals = []
    # 为了效率，创建一个从点数到其在 all_pts 中索引的映射
    # 注意：这要求 all_pts 中的点是可哈希的（元组可以，列表不行）
    point_to_index = {point: i for i, point in enumerate(all_pts)}

    i = 0
    while i < len(ok_pts):
        start_pt = ok_pts[i]

        # 寻找连续区间的结尾
        j = i
        while j + 1 < len(ok_pts):
            current_pt_idx = point_to_index.get(ok_pts[j])
            next_pt_idx = point_to_index.get(ok_pts[j + 1])

            # 检查点是否在 all_pts 中，并且是否连续
            if current_pt_idx is not None and next_pt_idx is not None and next_pt_idx == current_pt_idx + 1:
                j += 1
            else:
                # 如果点不在 all_pts 中或不连续，则区间在此处断开
                break

        end_pt = ok_pts[j]

        # 检查区间的起止点是否是 all_pts 的“自然”边界
        is_first_block = (start_pt == all_pts[0])
        is_last_block = (end_pt == all_pts[-1])

        # 根据是否为边界来决定格式
        if is_first_block and start_pt != end_pt:
            intervals.append(f'<= {end_pt}')
        elif is_last_block and start_pt != end_pt:
            intervals.append(f'>= {start_pt}')
        elif start_pt == end_pt:
            # 如果区间的起点和终点相同，或者起点已经是理论最大值
            intervals.append(f"{start_pt}")
        else:
            intervals.append(f'{start_pt} ~ {end_pt}')

        i = j + 1

    return " 或 ".join(intervals)


def format_tenpai_condition(cases, all_possible_cases, current_player_index, is_tenpai, _is_exclusion_check=False):
    """
    将流局听牌情况的元组列表格式化为易于理解的自然语言描述。
    - cases: 满足条件的听牌组合列表 (例如 tenpai_ok)
    - all_possible_cases: 所有可能被考虑的听牌组合
    - current_player_index: 当前正在计算的玩家索引
    - is_tenpai: 布尔值，True表示当前在处理“听牌”条件，False为“没听”
    """
    self_status_str = "听牌" if is_tenpai else "没听"
    self_status_val = 1 if is_tenpai else 0

    if not cases:
        return f"{self_status_str}: ✕"

    cases_set = set(tuple(c) for c in cases)
    relevant_possible_cases = [c for c in all_possible_cases if c[current_player_index] == self_status_val]
    relevant_possible_cases_set = set(tuple(c) for c in relevant_possible_cases)

    if cases_set == relevant_possible_cases_set:
        return f"{self_status_str}: 〇"

    # 模式一：“必须听牌/没听”模式 (高优先级)
    must_be_tenpai = []
    must_be_noten = []
    other_players_indices = [i for i in range(4) if i != current_player_index]

    if cases:
        first_case = cases[0]
        initial_must_tenpai = {i for i in other_players_indices if first_case[i] == 1}
        initial_must_noten = {i for i in other_players_indices if first_case[i] == 0}

        for case in cases[1:]:
            initial_must_tenpai &= {i for i in other_players_indices if case[i] == 1}
            initial_must_noten &= {i for i in other_players_indices if case[i] == 0}

        must_be_tenpai = sorted(list(initial_must_tenpai))
        must_be_noten = sorted(list(initial_must_noten))

    desc_parts = []
    if must_be_tenpai:
        player_nums = '、'.join(map(str, must_be_tenpai))
        desc_parts.append(f"玩家{player_nums}听牌")
    if must_be_noten:
        player_nums = '、'.join(map(str, must_be_noten))
        desc_parts.append(f"玩家{player_nums}没听")

    if desc_parts:
        from itertools import product
        expected_cases_set = set()
        free_players = [i for i in other_players_indices if i not in must_be_tenpai and i not in must_be_noten]

        for statuses in product([0, 1], repeat=len(free_players)):
            new_case = [0] * 4
            new_case[current_player_index] = self_status_val
            for p_idx in must_be_tenpai:
                new_case[p_idx] = 1
            for p_idx in must_be_noten:
                new_case[p_idx] = 0

            free_player_map = dict(zip(free_players, statuses))
            for p_idx, status in free_player_map.items():
                new_case[p_idx] = status

            expected_cases_set.add(tuple(new_case))

        if cases_set == expected_cases_set:
            if _is_exclusion_check:
                return f"{self_status_str}: (除了{'且'.join(desc_parts)})"
            return f"{self_status_str}: ({'且'.join(desc_parts)})"

    # 模式二：“排除”模式
    # 仅在非递归调用时尝试此模式，以防止无限循环
    if not _is_exclusion_check:
        failed_cases = relevant_possible_cases_set - cases_set
        if failed_cases:
            # 递归调用本函数来总结“失败”的组合
            # 将 _is_exclusion_check 设为 True 来阻止下一层递归使用“排除模式”
            fail_summary = format_tenpai_condition(
                list(failed_cases),
                all_possible_cases,
                current_player_index,
                is_tenpai,
                _is_exclusion_check=True
            )

            if fail_summary:
                # 如果成功总结了失败条件，就输出, 否则直接输出所有组合
                return fail_summary
    else:
        return None

    # 模式三：基于点差变化的模式
    if (is_tenpai and len(cases_set) == 2) or (not is_tenpai and len(cases_set) == 6):
        for other_player_idx in range(4):
            if other_player_idx == current_player_index:
                continue

            if is_tenpai:
                gain_diff_pattern = {
                    c for c in all_possible_cases
                    if c[current_player_index] == 1
                       and c[other_player_idx] == 0
                       and sum(c) in [1, 3]
                }
                if cases_set == gain_diff_pattern:
                    return f"{self_status_str}: (与玩家{other_player_idx}拉开4000点差)"
            else:  # not is_tenpai
                loss_diff_pattern = {
                    c for c in all_possible_cases
                    if c[current_player_index] == 0
                       and c[other_player_idx] == 1
                       and sum(c) in [1, 3]
                }
                noten_universe = {c for c in all_possible_cases if c[current_player_index] == 0}
                avoid_loss_pattern = noten_universe - loss_diff_pattern
                if cases_set == avoid_loss_pattern:
                    return f"{self_status_str}: (不被玩家{other_player_idx}拉开4000点差)"

    return f"{self_status_str}: {cases}"


def format_draw_condition_string(player_index, tenpai_ok, noten_ok, all_possible_cases, condition_type_str):
    """
    生成完整的、可直接打印的流局条件描述字符串。
    - player_index: 当前玩家索引
    - tenpai_ok: 满足条件的听牌组合列表
    - noten_ok: 满足条件的没听组合列表
    - all_possible_cases: 所有可能的听牌组合
    - condition_type_str: 字符串，如 "(终局)" 或 "[续行]"
    """
    # 如果此场景从一开始就不可能发生（例如亲家立直时考虑终局流局），则直接返回
    if not all_possible_cases:
        return None

    # 1. 为“听牌”和“没听”两种情况分别生成描述片段
    tenpai_str = format_tenpai_condition(tenpai_ok, all_possible_cases, player_index, True)
    noten_str = format_tenpai_condition(noten_ok, all_possible_cases, player_index, False)

    # 2. 检查对于该玩家，“听牌”和“没听”是否都是可选状态
    can_be_tenpai = any(case[player_index] == 1 for case in all_possible_cases)
    can_be_noten = any(case[player_index] == 0 for case in all_possible_cases)

    # 3. 将需要显示的描述组合起来
    desc_parts = []
    if can_be_tenpai:
        desc_parts.append(tenpai_str)
    if can_be_noten:
        desc_parts.append(noten_str)

    # 如果没有任何可描述的部分，则不返回任何字符串
    if not desc_parts:
        return None

    # 4. 合并最终结果，恢复顶层逻辑
    # 如果所有部分都成功(〇)，则总结果为〇
    if all(p.endswith("〇") for p in desc_parts):
        final_desc = "〇"
    # 如果所有部分都失败(✕)，则总结果为✕
    elif all(p.endswith("✕") for p in desc_parts):
        final_desc = "✕"
    # 否则，正常拼接
    else:
        final_desc = ", ".join(desc_parts)

    # 5. 返回完整的、可直接打印的字符串
    return f"玩家{player_index}流局条件{condition_type_str}: {final_desc}"


def check_continuation(pts_in_hand, ruleset, oya, oya_wins=False, oya_tenpai_in_draw=False):
    """
    根据规则和游戏状态判断游戏是否继续。
    返回 False 代表终局, True 代表续行。
    """
    # 优先级 1: 强制终局 (击飞)
    if not ruleset.get("continue_on_negative_score", True):
        if any(p < 0 for p in pts_in_hand):
            return False

    # 优先级 2: 强制续行 (西入)
    if ruleset.get("has_west_round"):
        threshold = ruleset.get("west_round_entry_threshold", 30000)
        if all(p < threshold for p in pts_in_hand):
            return True

    # 优先级 3: 强制终局 (亲家一位终了)
    if ruleset.get("dealer_continuation_in_1st") is False:
        player_points = list(enumerate(pts_in_hand))
        player_points_sorted = sorted(player_points, key=lambda x: x[1], reverse=True)
        is_oya_sole_first = (len(player_points_sorted) > 1 and
                             player_points_sorted[0][0] == oya and
                             player_points_sorted[1][1] < player_points_sorted[0][1])

        if is_oya_sole_first:
            return False

    # 优先级 4: 默认续行 (亲家连庄)
    if oya_wins or oya_tenpai_in_draw:
        return True

    # 优先级 5: 默认终局
    return False


if __name__ == "__main__":
    pts = [26000, 26000, 26000, 21000]
    placement = [45, 5, -15, -35]
    starting = 25000
    print(game_pts(pts, placement, starting))

    # print(final_ranking([10, 20, 20, 10]))
    #
    # print(list_possible_tenpai_cases([0, 0, 0, 0]))    ko_tsumo = sorted(list(ko_tsumo))

    oya_tsumo.extend([4000, 6000, 8000, 12000])
    ko_tsumo.extend([(2000, 4000), (3000, 6000), (4000, 8000), (6000, 12000)])

    fu = [25, 30, 40, 50, 60, 70, 80, 90, 100, 110]
    exception = [(1, 25)]
    if kiriage_mangan:
        kiriage_exceptions = [(3, 60), (4, 30)]
        exception.extend(kiriage_exceptions)

    oya_ron = set()
    ko_ron = set()

    for i in fan:
        for j in fu:
            if (i, j) in exception:
                continue

            oya_pts = 6 * j * 2 ** (i + 2)
            if oya_pts >= 12000:
                break
            oya_pts = ceil_to_hundred(oya_pts)
            oya_ron.add(oya_pts)

            ko_pts = 4 * j * 2 ** (i + 2)
            ko_pts = ceil_to_hundred(ko_pts)
            ko_ron.add(ko_pts)

    oya_ron = sorted(list(oya_ron))
    ko_ron = sorted(list(ko_ron))
    oya_ron.extend([12000, 18000, 24000, 36000])
    ko_ron.extend([8000, 12000, 16000, 24000])

    # 役满点数
    max_yakuman_multiple = 1  # FF -> 1倍
    if allow_double_yakuman and allow_composite_yakuman:  # TT -> 6倍
        max_yakuman_multiple = 6
    elif allow_double_yakuman:  # TF -> 2倍
        max_yakuman_multiple = 2
    elif allow_composite_yakuman:  # FT -> 4倍
        max_yakuman_multiple = 4

    yakuman_multiples = range(1, max_yakuman_multiple + 1)

    for m in yakuman_multiples:
        # 亲家自摸: 16000点/倍
        oya_tsumo.append(16000 * m)
        # 子家自摸: 8000/16000点/倍
        ko_tsumo.append((8000 * m, 16000 * m))
        # 亲家荣和: 48000点/倍
        oya_ron.append(48000 * m)
        # 子家荣和: 32000点/倍
        ko_ron.append(32000 * m)

    # print(oya_tsumo)
    # print(ko_tsumo)
    # print(oya_ron)
    # print(ko_ron)

    return oya_tsumo, ko_tsumo, oya_ron, ko_ron


def game_pts(pts_in_hand, placement_pts, starting=25000, tie_resolution="split_points",
             deposit_final_draw_recipient="first_place"):
    # 生成玩家索引和对应点数的列表
    player_points = list(enumerate(pts_in_hand))
    # 按点数从高到低排序
    player_points_sorted = sorted(player_points, key=lambda x: x[1], reverse=True)

    rank_points = [0] * 4

    if tie_resolution == "seating":
        # 按座次决定顺位
        # 由于 sorted() 是稳定的，分数相同时，原始顺序（即座次）被保留。
        # 因此，直接遍历排序后的列表即可实现“同分按座次”的规则。
        for i in range(4):
            player_idx = player_points_sorted[i][0]
            rank_points[player_idx] = placement_pts[i]
    else:  # 默认为 "split_points"，使用您编写的原始逻辑
        # 分组，点数相同的玩家归为一组
        groups = []
        if player_points_sorted:
            current_group = [player_points_sorted[0]]
            for i in range(1, len(player_points_sorted)):
                if player_points_sorted[i][1] == player_points_sorted[i - 1][1]:
                    current_group.append(player_points_sorted[i])
                else:
                    groups.append(current_group)
                    current_group = [player_points_sorted[i]]
            groups.append(current_group)

        rank_index = 0
        for group in groups:
            group_size = len(group)
            total_points = sum(placement_pts[rank_index:rank_index + group_size])
            # 先计算每人基础分，向下取整到0.1的倍数
            base_point = int(total_points / group_size * 10) / 10
            # 计算剩余分数，单位为0.1
            remainder = int(round((total_points - base_point * group_size) * 10))
            # 按座次优先分配余数
            for idx, (player_idx, _) in enumerate(group):
                add_point = remainder * 0.1 if idx == 0 else 0.
                rank_points[player_idx] = base_point + add_point
            rank_index += group_size

    for i in range(4):
        rank_points[i] += (pts_in_hand[i] - starting) / 1000

    # 计算供托数量
    deposit = (4 * starting - sum(pts_in_hand)) // 1000

    # 初始化供托分配列表
    deposit_distribution = [0.0, 0.0, 0.0, 0.0]

    # 仅当有供托且规则为“分配给第一名”时，才执行分配逻辑
    if deposit > 0 and deposit_final_draw_recipient == "first_place":
        # 计算第一名玩家的索引列表
        max_pts = max(pts_in_hand)
        first_players = [i for i, p in enumerate(pts_in_hand) if p == max_pts]

        if len(first_players) == 1:
            # 单独第一，全部供托给该玩家
            deposit_distribution[first_players[0]] = deposit
        elif len(first_players) == 3:
            # 三人并列第一，按座次顺序分配0.4, 0.3, 0.3
            distribution = [0.4, 0.3, 0.3]
            for idx, player in enumerate(first_players):
                deposit_distribution[player] = deposit * distribution[idx]
        else:
            # 其他情况（2人或4人并列），平均分配
            share = deposit / len(first_players)
            for player in first_players:
                deposit_distribution[player] = share

    total_points = [u + v for u, v in zip(rank_points, deposit_distribution)]

    total_points = [round(p, 1) for p in total_points]
    return total_points


def final_ranking(pts, tiebreaker):
    """
    根据分数和tiebreaker数组计算最终玩家排名。
    tiebreaker用于解决分数相同的情况，值越小，排名越高。
    """
    # 将每个玩家的原始索引、分数和tiebreaker值配对。
    pts = [round(p, 1) for p in pts]
    player_data = list(zip(range(len(pts)), pts, tiebreaker))

    # 使用多级排序键对玩家进行排序。
    # 1. 主排序：按分数降序。
    # 2. 次排序：按tiebreaker升序（通过使用负的tiebreaker值并整体降序排列来实现）。
    # 例如，对于同分的玩家，tiebreaker为1（-1）会排在tiebreaker为2（-2）的前面。
    sorted_players = sorted(player_data, key=lambda x: (x[1], -x[2]), reverse=True)

    # 根据排序后的顺序分配排名。
    ranks = [0] * len(pts)
    for rank, (original_index, score, tb) in enumerate(sorted_players, 1):
        ranks[original_index] = rank

    return ranks


def list_possible_tenpai_cases(riichi_status, oya=3):
    # 亲家听牌不能流局，若亲家听牌则无流局情况
    # 立直玩家一定听牌
    # 立直状态为0的玩家听牌状态不确定，需枚举

    tenpai_options = []
    for i in range(4):
        if i == oya:
            # 亲家不听牌
            if riichi_status[i] == 1:
                # 立直玩家必听牌
                tenpai_options.append([])
            else:
                tenpai_options.append([0])
        elif riichi_status[i] == 1:
            # 立直玩家必听牌
            tenpai_options.append([1])
        else:
            # 其他玩家可能听牌也可能不听牌
            tenpai_options.append([0, 1])

    # 枚举所有可能的听牌组合
    all_cases = list(product(*tenpai_options))
    return all_cases


def list_possible_oya_tenpai_cases(riichi_status, oya=3):
    # 这次列举亲家听牌的情况
    # 立直玩家一定听牌
    # 立直状态为0的玩家听牌状态不确定，需枚举

    tenpai_options = []
    for i in range(4):
        if i == oya or riichi_status[i] == 1:
            # 亲家听牌，立直玩家必听牌
            tenpai_options.append([1])
        else:
            # 其他玩家可能听牌也可能不听牌
            tenpai_options.append([0, 1])

    # 枚举所有可能的听牌组合
    all_cases = list(product(*tenpai_options))
    return all_cases


def list_all_possible_tenpai_cases(riichi_status):
    # 立直玩家一定听牌
    # 立直状态为0的玩家听牌状态不确定，需枚举

    tenpai_options = []
    for i in range(4):
        if riichi_status[i] == 1:
            # 立直玩家必听牌
            tenpai_options.append([1])
        else:
            # 其他玩家可能听牌也可能不听牌
            tenpai_options.append([0, 1])

    # 枚举所有可能的听牌组合
    all_cases = list(product(*tenpai_options))
    return all_cases


def format_as_intervals(ok_pts, all_pts):
    """将满足条件的点数列表格式化为区间字符串。"""
    if not ok_pts:
        return '✕'

    if ok_pts == all_pts:
        return '〇'

    intervals = []
    # 为了效率，创建一个从点数到其在 all_pts 中索引的映射
    # 注意：这要求 all_pts 中的点是可哈希的（元组可以，列表不行）
    point_to_index = {point: i for i, point in enumerate(all_pts)}

    i = 0
    while i < len(ok_pts):
        start_pt = ok_pts[i]

        # 寻找连续区间的结尾
        j = i
        while j + 1 < len(ok_pts):
            current_pt_idx = point_to_index.get(ok_pts[j])
            next_pt_idx = point_to_index.get(ok_pts[j + 1])

            # 检查点是否在 all_pts 中，并且是否连续
            if current_pt_idx is not None and next_pt_idx is not None and next_pt_idx == current_pt_idx + 1:
                j += 1
            else:
                # 如果点不在 all_pts 中或不连续，则区间在此处断开
                break

        end_pt = ok_pts[j]

        # 检查区间的起止点是否是 all_pts 的“自然”边界
        is_first_block = (start_pt == all_pts[0])
        is_last_block = (end_pt == all_pts[-1])

        # 根据是否为边界来决定格式
        if is_first_block and start_pt != end_pt:
            intervals.append(f'<= {end_pt}')
        elif is_last_block and start_pt != end_pt:
            intervals.append(f'>= {start_pt}')
        elif start_pt == end_pt:
            intervals.append(f"== {start_pt}")
        else:
            intervals.append(f'{start_pt} ~ {end_pt}')

        i = j + 1

    return " 或 ".join(intervals)


def format_tenpai_condition(cases, all_possible_cases, current_player_index, is_tenpai, _is_exclusion_check=False):
    """
    将流局听牌情况的元组列表格式化为易于理解的自然语言描述。
    - cases: 满足条件的听牌组合列表 (例如 tenpai_ok)
    - all_possible_cases: 所有可能被考虑的听牌组合
    - current_player_index: 当前正在计算的玩家索引
    - is_tenpai: 布尔值，True表示当前在处理“听牌”条件，False为“没听”
    """
    self_status_str = "听牌" if is_tenpai else "没听"
    self_status_val = 1 if is_tenpai else 0

    if not cases:
        return f"{self_status_str}: ✕"

    cases_set = set(tuple(c) for c in cases)
    relevant_possible_cases = [c for c in all_possible_cases if c[current_player_index] == self_status_val]
    relevant_possible_cases_set = set(tuple(c) for c in relevant_possible_cases)

    if cases_set == relevant_possible_cases_set:
        return f"{self_status_str}: 〇"

    # 模式一：“必须听牌/没听”模式 (高优先级)
    must_be_tenpai = []
    must_be_noten = []
    other_players_indices = [i for i in range(4) if i != current_player_index]

    if cases:
        first_case = cases[0]
        initial_must_tenpai = {i for i in other_players_indices if first_case[i] == 1}
        initial_must_noten = {i for i in other_players_indices if first_case[i] == 0}

        for case in cases[1:]:
            initial_must_tenpai &= {i for i in other_players_indices if case[i] == 1}
            initial_must_noten &= {i for i in other_players_indices if case[i] == 0}

        must_be_tenpai = sorted(list(initial_must_tenpai))
        must_be_noten = sorted(list(initial_must_noten))

    desc_parts = []
    if must_be_tenpai:
        player_nums = '、'.join(map(str, must_be_tenpai))
        desc_parts.append(f"玩家{player_nums}听牌")
    if must_be_noten:
        player_nums = '、'.join(map(str, must_be_noten))
        desc_parts.append(f"玩家{player_nums}没听")

    if desc_parts:
        from itertools import product
        expected_cases_set = set()
        free_players = [i for i in other_players_indices if i not in must_be_tenpai and i not in must_be_noten]

        for statuses in product([0, 1], repeat=len(free_players)):
            new_case = [0] * 4
            new_case[current_player_index] = self_status_val
            for p_idx in must_be_tenpai:
                new_case[p_idx] = 1
            for p_idx in must_be_noten:
                new_case[p_idx] = 0

            free_player_map = dict(zip(free_players, statuses))
            for p_idx, status in free_player_map.items():
                new_case[p_idx] = status

            expected_cases_set.add(tuple(new_case))

        if cases_set == expected_cases_set:
            if _is_exclusion_check:
                return f"{self_status_str}: (除了{'且'.join(desc_parts)})"
            return f"{self_status_str}: ({'且'.join(desc_parts)})"

    # 模式二：“排除”模式
    # 仅在非递归调用时尝试此模式，以防止无限循环
    if not _is_exclusion_check:
        failed_cases = relevant_possible_cases_set - cases_set
        if failed_cases:
            # 递归调用本函数来总结“失败”的组合
            # 将 _is_exclusion_check 设为 True 来阻止下一层递归使用“排除模式”
            fail_summary = format_tenpai_condition(
                list(failed_cases),
                all_possible_cases,
                current_player_index,
                is_tenpai,
                _is_exclusion_check=True
            )

            if fail_summary:
                # 如果成功总结了失败条件，就输出, 否则直接输出所有组合
                return fail_summary
    else:
        return None

    # 模式三：基于点差变化的模式
    if (is_tenpai and len(cases_set) == 2) or (not is_tenpai and len(cases_set) == 6):
        for other_player_idx in range(4):
            if other_player_idx == current_player_index:
                continue

            if is_tenpai:
                gain_diff_pattern = {
                    c for c in all_possible_cases
                    if c[current_player_index] == 1
                       and c[other_player_idx] == 0
                       and sum(c) in [1, 3]
                }
                if cases_set == gain_diff_pattern:
                    return f"{self_status_str}: (与玩家{other_player_idx}拉开4000点差)"
            else:  # not is_tenpai
                loss_diff_pattern = {
                    c for c in all_possible_cases
                    if c[current_player_index] == 0
                       and c[other_player_idx] == 1
                       and sum(c) in [1, 3]
                }
                noten_universe = {c for c in all_possible_cases if c[current_player_index] == 0}
                avoid_loss_pattern = noten_universe - loss_diff_pattern
                if cases_set == avoid_loss_pattern:
                    return f"{self_status_str}: (不被玩家{other_player_idx}拉开4000点差)"

    return f"{self_status_str}: {cases}"


def format_draw_condition_string(player_index, tenpai_ok, noten_ok, all_possible_cases, condition_type_str):
    """
    生成完整的、可直接打印的流局条件描述字符串。
    - player_index: 当前玩家索引
    - tenpai_ok: 满足条件的听牌组合列表
    - noten_ok: 满足条件的没听组合列表
    - all_possible_cases: 所有可能的听牌组合
    - condition_type_str: 字符串，如 "(终局)" 或 "[续行]"
    """
    # 如果此场景从一开始就不可能发生（例如亲家立直时考虑终局流局），则直接返回
    if not all_possible_cases:
        return None

    # 1. 为“听牌”和“没听”两种情况分别生成描述片段
    tenpai_str = format_tenpai_condition(tenpai_ok, all_possible_cases, player_index, True)
    noten_str = format_tenpai_condition(noten_ok, all_possible_cases, player_index, False)

    # 2. 检查对于该玩家，“听牌”和“没听”是否都是可选状态
    can_be_tenpai = any(case[player_index] == 1 for case in all_possible_cases)
    can_be_noten = any(case[player_index] == 0 for case in all_possible_cases)

    # 3. 将需要显示的描述组合起来
    desc_parts = []
    if can_be_tenpai:
        desc_parts.append(tenpai_str)
    if can_be_noten:
        desc_parts.append(noten_str)

    # 如果没有任何可描述的部分，则不返回任何字符串
    if not desc_parts:
        return None

    # 4. 合并最终结果，恢复顶层逻辑
    # 如果所有部分都成功(〇)，则总结果为〇
    if all(p.endswith("〇") for p in desc_parts):
        final_desc = "〇"
    # 如果所有部分都失败(✕)，则总结果为✕
    elif all(p.endswith("✕") for p in desc_parts):
        final_desc = "✕"
    # 否则，正常拼接
    else:
        final_desc = ", ".join(desc_parts)

    # 5. 返回完整的、可直接打印的字符串
    return f"玩家{player_index}流局条件{condition_type_str}: {final_desc}"


def check_continuation(pts_in_hand, ruleset, oya, oya_wins=False, oya_tenpai_in_draw=False):
    """
    根据规则和游戏状态判断游戏是否继续。
    返回 False 代表终局, True 代表续行。
    """
    # 优先级 1: 强制终局 (击飞)
    if not ruleset.get("continue_on_negative_score", True):
        if any(p < 0 for p in pts_in_hand):
            return False

    # 优先级 2: 强制续行 (西入)
    if ruleset.get("has_west_round"):
        threshold = ruleset.get("west_round_entry_threshold", 30000)
        if all(p < threshold for p in pts_in_hand):
            return True

    # 优先级 3: 强制终局 (亲家一位终了)
    if ruleset.get("dealer_continuation_in_1st") is False:
        player_points = list(enumerate(pts_in_hand))
        player_points_sorted = sorted(player_points, key=lambda x: x[1], reverse=True)
        is_oya_sole_first = (len(player_points_sorted) > 1 and
                             player_points_sorted[0][0] == oya and
                             player_points_sorted[1][1] < player_points_sorted[0][1])

        if is_oya_sole_first:
            return False

    # 优先级 4: 默认续行 (亲家连庄)
    if oya_wins or oya_tenpai_in_draw:
        return True

    # 优先级 5: 默认终局
    return False


if __name__ == "__main__":
    pts = [26000, 26000, 26000, 21000]
    placement = [45, 5, -15, -35]
    starting = 25000
    print(game_pts(pts, placement, starting))

    # print(final_ranking([10, 20, 20, 10]))
    #
    # print(list_possible_tenpai_cases([0, 0, 0, 0]))
