"""LabelTable: names and colours for label numbers (1 = 'Ceramic', ...)."""
PALETTE = [(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48),
           (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60)]


class LabelTable:
    def __init__(self):
        self.entries = {}          # id -> {"name":..., "color":[r,g,b], "visible":True}

    def add(self, name, color=None, label_id=None):
        if label_id is None:
            label_id = max(self.entries, default=0) + 1
        if color is None:
            color = PALETTE[(label_id - 1) % len(PALETTE)]
        self.entries[label_id] = {"name": name, "color": list(color), "visible": True}
        return label_id

    def remove(self, label_id):
        self.entries.pop(label_id, None)

    def to_dict(self):
        return {str(k): v for k, v in self.entries.items()}

    @classmethod
    def from_dict(cls, d):
        t = cls()
        t.entries = {int(k): v for k, v in d.items()}
        return t
