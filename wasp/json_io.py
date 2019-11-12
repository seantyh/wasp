import json
import dataclasses

class MultiChoiceJsonEncoder(json.JSONEncoder):
    #pylint: disable=method-hidden
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)