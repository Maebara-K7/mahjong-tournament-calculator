# rulesets.py
# 这个文件用于存储各种预设的麻将规则集

RULES = {
    "ML": {
        "name": "M-League 规则",
        "placement_pts": [45, 5, -15, -35],
        "starting_pts": 25000,
        "kiriage_mangan": True,
        "allow_double_yakuman": False,
        "allow_composite_yakuman": True,
        "tie_resolution": "split_points",
        "has_west_round": False,
        "continue_on_negative_score": True,
        "dealer_continuation_in_1st": True,
        "deposit_final_draw_recipient": "first_place"
    },

    "TH": {
        "name": "天凤比赛场 规则",
        "placement_pts": [50, 20, 0, -70],
        "starting_pts": 25000,
        "kiriage_mangan": False,
        "allow_double_yakuman": False,
        "allow_composite_yakuman": True,
        "tie_resolution": "seating",
        "has_west_round": True,
        "west_round_entry_threshold": 30000,
        "continue_on_negative_score": False,
        "dealer_continuation_in_1st": False,
        "deposit_final_draw_recipient": "first_place"
    },

    "JT": {
        "name": "雀魂比赛场 规则",
        "placement_pts": [15, 5, -5, -15],
        "starting_pts": 25000,
        "kiriage_mangan": False,
        "allow_double_yakuman": True,
        "allow_composite_yakuman": True,
        "tie_resolution": "seating",
        "has_west_round": True,
        "west_round_entry_threshold": 30000,
        "continue_on_negative_score": False,
        "dealer_continuation_in_1st": False,
        "deposit_final_draw_recipient": "first_place"
    },

    "SK": {
        "name": "最高位战 规则",
        "placement_pts": [30, 10, -10, -30],
        "starting_pts": 30000,
        "kiriage_mangan": True,
        "allow_double_yakuman": False,
        "allow_composite_yakuman": True,
        "tie_resolution": "split_points",
        "has_west_round": False,
        "continue_on_negative_score": True,
        "dealer_continuation_in_1st": True,
        "deposit_final_draw_recipient": "unclaimed"
    },

    "CUS": {
        "name": "雀魂比赛场规则",
        "placement_pts": [30, 10, -10, -30],
        "starting_pts": 25000,
        "kiriage_mangan": False,
        "allow_double_yakuman": False,
        "allow_composite_yakuman": True,
        "tie_resolution": "seating",
        "has_west_round": True,
        "west_round_entry_threshold": 30000,
        "continue_on_negative_score": False,
        "dealer_continuation_in_1st": False,
        "deposit_final_draw_recipient": "first_place"
    },
}
