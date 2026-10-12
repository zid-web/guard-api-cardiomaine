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
