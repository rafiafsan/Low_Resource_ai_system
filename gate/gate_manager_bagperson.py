import json
from config.settings import GATE_CONFIG_FILE_PERSON_BAG

class GateManagerBag:
    def __init__(self):
        self.gate_points = []
        self.inside_reference = None
        self.load_config()

    def load_config(self):
        try:
            with open(GATE_CONFIG_FILE_PERSON_BAG, 'r') as f:
                data = json.load(f)
                points = data.get("gate_points", [])
                if points and len(points) == 2:
                    self.gate_points = [tuple(p) for p in points]
                ref = data.get("inside_reference", None)
                if ref:
                    self.inside_reference = tuple(ref)
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    def save_config(self):
        with open(GATE_CONFIG_FILE_PERSON_BAG, 'w') as f:
            json.dump({
                "gate_points": self.gate_points,
                "inside_reference": self.inside_reference
            }, f, indent=4)
