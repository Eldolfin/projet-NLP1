import time


class Prediction:
    def __init__(
        self,
        method: str,
        scenario: str,
        intent: str,
        proba: float,
        before: float,
    ):
        self.method = method
        self.scenario = scenario
        self.intent = intent
        self.proba = proba
        self.time_taken = time.process_time() - before

    def __str__(self):
        return f"|{self.method}|{self.scenario}|{self.intent}|{self.proba}|{round(self.time_taken,2)}s"
