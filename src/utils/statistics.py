class Average:
    def __init__(self):
        self.reset()

    def reset(self):
        self._count = 0
        self._value = None

    def update(self, data):
        if self._count == 0:
            self._value = data
        else:
            cumulative_value = self._value * self._count
            self._value = (cumulative_value + data) / (self._count + 1)
        self._count += 1

    @property
    def value(self):
        return self._value
