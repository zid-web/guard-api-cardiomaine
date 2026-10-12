"""Test d'intégration : app_rules est accepté, accusé de réception, nuits non consécutives."""
from solver import GenerateWeekRequest, generate_week

def _req(app_rules=None, prev=None):
    meds = [{"id": i, "statut": "permanent"} for i in "ABGZHSUPKRT"] + \
           [{"id": i, "statut": "astreinte_coro"} for i in "MOW"] + \
           [{"id": "FV", "statut": "fv"}, {"id": "D", "statut": "d"}, {"id": "DAAS", "statut": "daas"}, {"id": "CH", "statut": "ch"}]
    return GenerateWeekRequest(week_start_date="2026-10-26", week_type=2, medecins=meds,
                               previous_sunday_guard_doctor=prev, app_rules=app_rules)

def test_sans_app_rules_inchange():
    r = generate_week(_req())
    assert r.applied_rules is None and r.rules_version is None

def test_app_rules_ack_et_nuits_non_consecutives():
    rules = {"version": 1, "night_guard": {"no_consecutive_nights": True, "friday_pool": ["B", "G", "A", "P", "Z", "H", "S"],
             "never_friday": ["O", "W", "M"], "previous_sunday_guard_doctor": "U"}}
    r = generate_week(_req(rules, prev="U"))
    assert r.rules_version == 1 and r.applied_rules["no_consecutive_nights"]
    nights = {}
    for a in r.assignments:
        if a.activity == "GARDE" and a.slot == "nuit":
            nights[a.day_name] = a.doctor
    order = ["LUNDI", "MARDI", "MERCREDI", "JEUDI", "VENDREDI"]
    for d1, d2 in zip(order, order[1:]):
        assert not (d1 in nights and d2 in nights and nights[d1] == nights[d2]), nights
    assert nights.get("LUNDI") != "U"
    if "VENDREDI" in nights:
        assert nights["VENDREDI"] in rules["night_guard"]["friday_pool"]

def test_creneau_ferme():
    rules = {"version": 1, "closed_slots": [{"row": "Garde Nuit", "day": "MERCREDI", "reason": "test"}]}
    r = generate_week(_req(rules))
    assert not any(a.activity == "GARDE" and a.slot == "nuit" and a.day_name == "MERCREDI" for a in r.assignments)

if __name__ == "__main__":
    test_sans_app_rules_inchange(); test_app_rules_ack_et_nuits_non_consecutives(); test_creneau_ferme(); print("OK")

def test_eligibilites_et_hors_site():
    rules = {"version": 1, "eligibility": {"ett_forbidden": ["O"], "ee_solo_doctors": ["O", "V", "DAAS"]},
             "hors_site_usual_days": {"Hors site - CDL": ["MARDI"], "Hors site - IRM": ["LUNDI", "VENDREDI"]},
             "priorities": {"coro_first": ["M", "O", "W"], "echo_group": ["H", "Z", "G", "B", "S"]},
             "fixed_assignments": [{"row": "Garde Nuit", "day": "MARDI", "doctors": ["A"]}],
             "cleared_cells": [{"row": "Garde Nuit", "day": "JEUDI"}]}
    r = generate_week(_req(rules))
    assert r.applied_rules["ee_rules"] and r.applied_rules["priorities"]
    assert not any(a.doctor == "O" and "ETT" in a.activity for a in r.assignments)
    assert not any(a.activity == "GARDE" and a.slot == "nuit" and a.day_name == "JEUDI" for a in r.assignments)
    for a in r.assignments:
        if a.activity == "CDL":
            assert a.day_name == "MARDI"

if __name__ == "__main__":
    test_eligibilites_et_hors_site(); print("OK2")

def test_entrees_pss_binome():
    r = generate_week(_req())
    garde = {a.day_name: a.doctor for a in r.assignments if a.activity == "GARDE" and a.slot == "am"}
    pss = {}
    for a in r.assignments:
        if "ntr" in a.activity and "PSS" in a.activity:
            pss.setdefault(a.day_name, []).append(a.doctor)
    for d in ("MERCREDI", "JEUDI", "VENDREDI"):
        if d in pss:
            assert pss[d] == [garde[d]], (d, pss[d], garde.get(d))
    for d in ("LUNDI", "MARDI"):
        if d in pss and d in garde:
            assert garde[d] in pss[d] and len(pss[d]) <= 2, (d, pss[d], garde[d])
    print("pss", pss, "garde", garde)

if __name__ == "__main__":
    test_entrees_pss_binome(); print("OK3")

def test_coro_un_chacun():
    rules = {"version": 1, "coro_one_each_per_day": {"doctors": ["M", "O", "W"], "penalty": 80}}
    r = generate_week(_req(rules))
    assert r.applied_rules["coro_one_each_per_day"]
    load = {}
    for a in r.assignments:
        if a.doctor in "MOW" and ((a.activity == "CORO") or (a.slot == "nuit" and a.activity in ("ASTREINTE", "GARDE"))) and a.day_name in ("LUNDI","MARDI","MERCREDI","VENDREDI"):
            load[(a.day_name, a.doctor)] = load.get((a.day_name, a.doctor), 0) + 1
    print("charges M/O/W par jour:", load)
    for day in ("LUNDI", "MARDI", "MERCREDI", "VENDREDI"):
        assert max([load.get((day, d), 0) for d in "MOW"]) <= 2

if __name__ == "__main__":
    test_coro_un_chacun(); print("OK4")
def test_mardi_pool():
    rules = {"version": 1, "night_guard": {"day_preferences": {"MARDI": {"prefer": ["G", "B", "M", "W"]}}}}
    for ws in ("2026-10-19", "2026-11-02", "2026-11-16"):
        r = _req(rules); r.week_start_date = ws; r.week_type = 1
        out = generate_week(r)
        mardi = [a.doctor for a in out.assignments if a.activity == "GARDE" and a.slot == "nuit" and a.day_name == "MARDI"]
        assert mardi and mardi[0] in ("G", "B", "M", "W"), (ws, mardi)

if __name__ == "__main__":
    test_mardi_pool(); print("OK5")

def test_vendredi_egalite():
    pool = ["B", "G", "A", "P", "Z", "H", "S", "U"]
    counts = {d: 5 for d in pool}; counts["H"] = 0
    rules = {"version": 1, "night_guard": {"friday_pool": pool, "friday_counts": counts, "never_friday": ["O", "W", "M"]}}
    r = generate_week(_req(rules))
    ven = [a.doctor for a in r.assignments if a.activity == "GARDE" and a.slot == "nuit" and a.day_name == "VENDREDI"]
    assert r.applied_rules["friday_equality"]
    assert ven == ["H"], ven

if __name__ == "__main__":
    test_vendredi_egalite(); print("OK6")

def test_mercredi_s_p_alternent():
    prefs = {"MERCREDI": {"prefer": ["S", "P"]}}
    r = _req({"version": 1, "night_guard": {"day_preferences": prefs, "day_counts": {"MERCREDI": {"S": 0, "P": 4}}}})
    r.week_start_date = "2026-11-16"; r.week_type = 1
    out = generate_week(r)
    mer = [a.doctor for a in out.assignments if a.activity == "GARDE" and a.slot == "nuit" and a.day_name == "MERCREDI"]
    assert mer == ["S"], mer
    ett = [a.doctor for a in out.assignments if a.day_name == "MERCREDI" and a.slot == "am" and "ETT salle 1" in a.activity]
    assert ett == ["S"], ett  # ETT ped conservée avec la garde

if __name__ == "__main__":
    test_mercredi_s_p_alternent(); print("OK7")
