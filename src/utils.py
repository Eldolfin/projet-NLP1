class Prediction:
    def __init__(self, method: str, scenario: str, intent: str, proba: float):
        self.method = method
        self.scenario = scenario
        self.intent = intent
        self.proba = proba

    def __str__(self):
        return f"|{self.method}|{self.scenario}|{self.intent}|{self.proba}"
