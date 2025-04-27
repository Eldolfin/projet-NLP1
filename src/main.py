from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.reactive import reactive
from textual.widgets import (
    LoadingIndicator,
    Input,
    Static,
    Header,
    Footer,
    DataTable,
    ProgressBar,
)
from textual.containers import Container, Horizontal
from enum import Enum
import asyncio
from utils import Prediction
from typing import List
import time
from phrases import SCENARIOS
from dataclasses import dataclass

# Disable training models and loading dataset for faster ui iteration
DISABLE_MODELS = False

if DISABLE_MODELS:
    import random as rand
else:
    import basic
    import ngrams
    import word2vec


@dataclass
class TrainingStep:
    """Class for keeping track of an item in inventory."""

    name: str
    started_at: float
    ended_at: float
    # TODO: add score?

    def time_taken_secs(self) -> float:
        return round(self.ended_at - self.started_at, 2)


class AppState(Enum):
    LOADING_DATASET = 1
    TRAINING_MODELS = 2
    INFERENCE_INPUT = 3


class DemoNlpApp(App):
    input_text = reactive("")
    state = reactive(AppState.LOADING_DATASET)
    training_steps: reactive[List[TrainingStep]] = reactive([])
    BINDINGS = [
        Binding(key="^q", action="quit", description="Quit the app"),
    ]
    CSS = """
        #content {
            height: 100%;
            margin: 4 8;
            background: $panel;
            color: $text;
            border: tall $background;
            padding: 1 2;
            content-align: center middle;
        }

        #loading-state {
            text-style: bold;
            content-align: center middle;
            height: 20%
        }
        .training-step-indicator {
            width: 10%;
            margin: 0 1;
        }
    """

    def content(self) -> ComposeResult:
        match self.state:
            case AppState.LOADING_DATASET:
                yield Static("1. Loading dataset", id="loading-state")
                yield LoadingIndicator(name="Loading dataset")
            case AppState.TRAINING_MODELS:
                yield Static("2. Training models", id="loading-state")
                for i, step in enumerate(self.training_steps):
                    if i + 1 == len(self.training_steps):
                        yield Horizontal(
                            Container(
                                ProgressBar(),
                                classes="training-step-indicator",
                            ),
                            Container(
                                Static(
                                    f"[bold italic yellow]2.{i}. {step.name}",
                                    expand=True,
                                )
                            ),
                        )
                    else:
                        yield Static(
                            f"✅ [bold italic lightgreen]2.{i}. {step.name} (took {step.time_taken_secs()}s)",
                            expand=True,
                            classes="training-step",
                        )
            case AppState.INFERENCE_INPUT:
                yield Input(placeholder="Type here...")
                yield DataTable()

    def compose(self) -> ComposeResult:
        yield Header()
        yield Footer()
        yield Container(
            *list(self.content()),
            id="content",
        )

    async def on_mount(self) -> None:
        self.title = "NLP project demo"
        self.sub_title = "Alexa task classifier 🤖 🔥"
        self.prepare()

    @work(exclusive=True)
    async def prepare(self):
        await self.load_dataset()
        self.change_state(AppState.TRAINING_MODELS)
        await self.train_models()
        self.change_state(AppState.INFERENCE_INPUT)

    async def load_dataset(self):
        from datasets import load_dataset  # type: ignore[import-untyped]

        if DISABLE_MODELS:
            await asyncio.sleep(1)
        else:
            self.ds = await asyncio.to_thread(
                load_dataset, "AmazonScience/massive", "fr-FR"
            )

    async def train_models(self):
        if DISABLE_MODELS:
            self.add_training_step("Training fake model 1")
            await asyncio.sleep(1)
            self.add_training_step("Training fake model 2")
            await asyncio.sleep(1)
            self.add_training_step("Training fake model 3")
            await asyncio.sleep(1)
        else:
            await asyncio.to_thread(self.train_models_blocking)

    def add_training_step(self, name: str):
        now = time.process_time()
        if self.training_steps != []:
            self.training_steps[-1].ended_at = now
        self.training_steps.append(
            TrainingStep(name=name, started_at=now, ended_at=0)
        )
        self.refresh(recompose=True)

    def train_models_blocking(self):
        import basic
        import ngrams
        import word2vec

        # from transformer import tf_train, tf_classify
        from sklearn.feature_extraction.text import (
            CountVectorizer,
            TfidfVectorizer as Tfv,
        )
        from sklearn.linear_model import LogisticRegression as Lr
        from sklearn.naive_bayes import MultinomialNB as Mnb  # type: ignore[import-untyped]

        X_train = self.ds["train"]["utt"]
        y_train = self.ds["train"]["scenario"]
        X_test = self.ds["test"]["utt"]
        y_test = self.ds["test"]["scenario"]

        self.add_training_step("Training bag of words")
        self.bow_vectorizer, self.bow_clf, self.bow_intent_models = (
            basic.basic_train(
                self.ds, X_train, y_train, X_test, y_test, CountVectorizer, Mnb
            )
        )

        self.add_training_step("Training ngrams")
        self.scenario_grams, self.intent_grams = ngrams.train_ngrams(
            self.ds, X_train, y_train, X_test, y_test
        )

        self.add_training_step("Training idf")
        self.idf_vectorizer, self.idf_clf, self.idf_intent_models = (
            basic.basic_train(
                self.ds, X_train, y_train, X_test, y_test, Tfv, Lr
            )
        )

        # self.add_training_step("Training word2vec")
        # self.w2v_model, self.w2v_clf, self.w2v_intent_models = (
        #     word2vec.w2v_train(self.ds, X_train, y_train, X_test, y_test)
        # )

        # self.add_training_step("Training neural network")
        # tf_model, tf_clf, tf_intent_models = tf_train(self.ds)

    def change_state(self, new_state: AppState):
        self.state = new_state
        self.refresh(recompose=True)

    def on_input_changed(self, event: Input.Changed):
        self.input_text = event.value
        predictions = self.predict_class(self.input_text)
        table = self.query_one(DataTable)
        table.clear(columns=True)
        table.add_column("Model", key="model")
        table.add_column("Scenario", key="scenario")
        table.add_column("Intent", key="intent")
        table.add_column("Proba", key="proba")
        table.add_column("Inference time", key="time")
        table.zebra_stripes = True
        for prediction in predictions:
            table.add_row(
                prediction.method.capitalize(),
                prediction.scenario,
                prediction.intent,
                str(round(prediction.proba, 2)),
                str(round(prediction.time_taken, 3)),
            )
        table.sort("proba", reverse=True)

    def predict_class(self, text) -> List[Prediction]:

        if DISABLE_MODELS:
            return [
                Prediction(
                    "random model",
                    rand.choice(SCENARIOS),
                    rand.choice(SCENARIOS) + "/" + rand.choice(SCENARIOS),
                    rand.random() * 100,
                    before=time.process_time() - 0.03,
                ),
                Prediction(
                    "clown model 🤡",
                    rand.choice(SCENARIOS[:2]),
                    SCENARIOS[0] + "/" + rand.choice(SCENARIOS[:2]),
                    rand.random() * 10,
                    before=time.process_time() - 0.01,
                ),
            ]
        return [
            basic.basic_classify(
                self.ds,
                self.bow_vectorizer,
                self.bow_clf,
                self.bow_intent_models,
                self.input_text,
                "bow",
            ),
            ngrams.ngrams_classify(
                self.ds, self.scenario_grams, self.intent_grams, self.input_text
            ),
            basic.basic_classify(
                self.ds,
                self.idf_vectorizer,
                self.idf_clf,
                self.idf_intent_models,
                self.input_text,
                "idf",
            ),
            # word2vec.w2v_classify(
            #     self.ds,
            #     self.w2v_model,
            #     self.w2v_clf,
            #     self.w2v_intent_models,
            #     self.input_text,
            #     "word2vec",
            # ),
            # tf_classify(
            #     ds, tf_model, tf_clf, tf_intent_models, user_input, "word2vec"
            # )
        ]


if __name__ == "__main__":
    DemoNlpApp().run()
