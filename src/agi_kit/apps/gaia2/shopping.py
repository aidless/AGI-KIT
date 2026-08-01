# Shopping app - GAIA2 app simulation.
from __future__ import annotations
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).parent.parent.parent))
from agi_kit.apps.gaia2 import load_state, save_state


_CATALOG = [
    {"product_id": "p_001", "name": "Notebook", "price": 4.99},
    {"product_id": "p_002", "name": "Pen set", "price": 12.50},
    {"product_id": "p_003", "name": "USB-C cable", "price": 9.75},
    {"product_id": "p_004", "name": "Coffee mug", "price": 8.00},
    {"product_id": "p_005", "name": "Wireless mouse", "price": 24.99},
]


def get_product_information(scenario_id, args):
    pid = args.get("product_id", "")
    for p in _CATALOG:
        if p["product_id"] == pid:
            return {"product": p}
    return {"product": None}


def display_all_products(scenario_id, args):
    return {"products": _CATALOG}


def purchase_item(scenario_id, args):
    state = load_state(scenario_id, "shopping")
    pid = args.get("product_id", "")
    for p in _CATALOG:
        if p["product_id"] == pid:
            state.setdefault("purchases", []).append({"product": p, "qty": args.get("quantity", 1)})
            save_state(scenario_id, "shopping", state)
            return {"purchased": p, "qty": args.get("quantity", 1)}
    return {"error": "no such product"}


def get_purchase_history(scenario_id, args):
    state = load_state(scenario_id, "shopping")
    return {"history": state.get("purchases", [])}


FUNCTIONS = {
    "get_product_information": get_product_information,
    "display_all_products": display_all_products,
    "purchase_item": purchase_item,
    "get_purchase_history": get_purchase_history,
}


def call(scenario_id, function, args):
    if function not in FUNCTIONS:
        return {"error": "unknown function"}
    return FUNCTIONS[function](scenario_id, args)

