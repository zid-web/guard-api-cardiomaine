"""
Chargement des règles métier (listes d'éligibilité, demi-journées off, calendrier NCT fixe...)
depuis un fichier JSON plutôt que codées en dur dans solver.py.

But : permettre de changer une règle (ex: un médecin change de statut, une nouvelle
exclusion) sans avoir à modifier le code Python ni redéployer le service.

Deux niveaux de configuration, fusionnés à chaque appel :
1. Défaut : rules_config.json, embarqué avec le service (sert de filet de sécurité).
2. Surcharge : `rules_override` optionnel dans la requête /generate-week, que le
   front peut renseigner depuis une table Supabase modifiable par un administrateur,
   sans jamais toucher au code de ce service.

Limite actuelle : le fichier JSON par défaut nécessite quand même un redéploiement
pour changer. La vraie souplesse vient de `rules_override` envoyé par le front -
à brancher sur une table Supabase dédiée (ex: `rules_config`) côté front.
"""
import json
import os
from typing import Any, Dict, Optional

_CONFIG_PATH = os.environ.get(
    "GUARD_RULES_CONFIG_PATH",
    os.path.join(os.path.dirname(__file__), "rules_config.json"),
)

_cached_default_rules: Optional[Dict[str, Any]] = None


def load_default_rules() -> Dict[str, Any]:
    """Charge (et met en cache) les règles par défaut depuis rules_config.json."""
    global _cached_default_rules
    if _cached_default_rules is None:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            _cached_default_rules = json.load(f)
    return _cached_default_rules


def merge_rules(default: Dict[str, Any], override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Fusionne superficiellement les règles par défaut avec une surcharge envoyée
    dans la requête. Une clé présente dans `override` remplace entièrement la
    valeur par défaut correspondante (pas de fusion profonde champ par champ).
    """
    if not override:
        return default
    merged = dict(default)
    for key, value in override.items():
        if key in merged:
            merged[key] = value
    return merged


def half_days_off_as_dict(rules: Dict[str, Any]) -> Dict[Any, set]:
    """Convertit la liste JSON `half_days_off` en dict {(jour, créneau): {médecins}}."""
    result = {}
    for entry in rules.get("half_days_off", []):
        result[(entry["day"], entry["slot"])] = set(entry["doctors"])
    return result


def fixed_exclusions_as_dict(rules: Dict[str, Any]) -> Dict[str, set]:
    """Convertit `fixed_exclusions` JSON en dict {médecin: {indices de jour}}."""
    return {doc: set(days) for doc, days in rules.get("fixed_exclusions", {}).items()}
