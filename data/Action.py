class Action:
    _value: int
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value