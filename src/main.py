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
from typing import List, Self
import time
from phrases import SCENARIOS
from dataclasses import dataclass
import pickle
from typing import Union, Dict
import os
from textual.suggester import Suggester

# Disable training models and loading dataset for faster ui iteration
DISABLE_MODELS = False
MODEL_SAVE_PATH = "./.cache/models.pkl"
countries = ["England", "Scotland", "Portugal", "Spain", "France"]

if DISABLE_MODELS:
    import random as rand
else:
    import basic
    import ngrams
    import word2vec
    import transformer


@dataclass
class TrainingStep:
    """Class for keeping track of an item in inventory."""

    name: str
    started_at: float
    ended_at: float
    # TODO: add score?

    def time_taken_secs(self) -> float:
        return round(self.ended_at - self.started_at, 2)


@dataclass
class Models:
    # Bag Of Words
    bow_vectorizer = None
    bow_clf = None
    bow_intent_models: Dict | None = None
    input_text = None
    # Ngrams
    scenario_grams: ngrams.NgramList | None = None
    intent_grams = None
    # IDF
    idf_vectorizer = None
    idf_clf = None
    idf_intent_models = None
    # W2V
    w2v_model = None
    w2v_clf = None
    w2v_intent_models = None
    # Neural Network
    tf_model = None
    tf_clf = None
    tf_intent_models = None

    def save(self) -> None:
        with open(MODEL_SAVE_PATH, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load() -> Self | None:  # type: ignore[misc]
        if not os.path.isfile(MODEL_SAVE_PATH):
            return None
        with open(MODEL_SAVE_PATH, "rb") as f:
            return pickle.load(f)


class AppState(Enum):
    LOADING_DATASET = 1
    TRAINING_MODELS = 2
    INFERENCE_INPUT = 3


class DemoNlpApp(App):
    input_text = reactive("")
    state = reactive(AppState.LOADING_DATASET)
    training_steps: reactive[List[TrainingStep]] = reactive([])
    models: Models | None = None
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
                yield Input(
                    placeholder="Demandez une question à Alexa (en français)...",
                    suggester=SmartSuggester(self, case_sensitive=False),
                )
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
        await asyncio.to_thread(self.train_models_blocking)

    def add_training_step(self, name: str):
        now = time.process_time()
        if self.training_steps != []:
            self.training_steps[-1].ended_at = now
        self.training_steps.append(TrainingStep(name=name, started_at=now, ended_at=0))
        self.refresh(recompose=True)

    def train_models_blocking(self):
        if DISABLE_MODELS:
            self.add_training_step("Training fake model 1")
            time.sleep(1)
            self.add_training_step("Training fake model 2")
            time.sleep(1)
            self.add_training_step("Training fake model 3")
            time.sleep(1)
            return

        self.add_training_step(f"Trying to load models from [i]{MODEL_SAVE_PATH}[/i]")
        self.models = Models.load()
        if self.models is not None:
            return
        self.models = Models()
        import basic
        import ngrams
        import word2vec
        import transformer
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
        (
            self.models.bow_vectorizer,
            self.models.bow_clf,
            self.models.bow_intent_models,
        ) = basic.basic_train(
            self.ds, X_train, y_train, X_test, y_test, CountVectorizer, Mnb
        )

        self.add_training_step("Training ngrams")
        self.models.scenario_grams, self.models.intent_grams = ngrams.train_ngrams(
            self.ds, X_train, y_train, X_test, y_test
        )

        # self.add_training_step("Training idf")
        # (
        #     self.models.idf_vectorizer,
        #     self.models.idf_clf,
        #     self.models.idf_intent_models,
        # ) = basic.basic_train(self.ds, X_train, y_train, X_test, y_test, Tfv, Lr)

        # self.add_training_step("Training word2vec")
        # (
        #     self.models.w2v_model,
        #     self.models.w2v_clf,
        #     self.models.w2v_intent_models,
        # ) = word2vec.w2v_train(self.ds, X_train, y_train, X_test, y_test)

        # self.add_training_step("Training neural network")
        # self.tf_model, self.tf_clf, self.tf_intent_models = (
        #     transformer.tf_train(self.ds)
        # )

        # self.add_training_step("Training word2vec")
        # (
        #     self.models.w2v_model,
        #     self.models.w2v_clf,
        #     self.models.w2v_intent_models,
        # ) = word2vec.w2v_train(self.ds, X_train, y_train, X_test, y_test)

        # self.add_training_step("Training neural network")
        # self.tf_model, self.tf_clf, self.tf_intent_models = (
        #     transformer.tf_train(self.ds)
        # )

        self.models.save()

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
                self.models.bow_vectorizer,
                self.models.bow_clf,
                self.models.bow_intent_models,
                self.input_text,
                "bow",
            ),
            ngrams.ngrams_classify(
                self.ds,
                self.models.scenario_grams,
                self.models.intent_grams,
                self.input_text,
            ),
            # basic.basic_classify(
            #     self.ds,
            #     self.models.idf_vectorizer,
            #     self.models.idf_clf,
            #     self.models.idf_intent_models,
            #     self.input_text,
            #     "idf",
            # ),
            # word2vec.w2v_classify(
            #     self.ds,
            #     self.models.w2v_model,
            #     self.models.w2v_clf,
            #     self.models.w2v_intent_models,
            #     self.input_text,
            #     "word2vec",
            # ),
            # transformer.tf_classify(
            #     self.ds,
            #     self.models.tf_model,
            #     self.models.tf_clf,
            #     self.models.tf_intent_models,
            #     self.input_text,
            #     "word2vec",
            # ),
        ]


class SmartSuggester(Suggester):
    """Give completion suggestions based on a fixed list of options.

    Example:
        ```py
        class MyApp(App[None]):
            def compose(self) -> ComposeResult:
                yield Input(suggester=SmartSuggester(case_sensitive=False))
        ```
    """

    def __init__(
        self,
        app: DemoNlpApp,
        case_sensitive=False,
    ) -> None:
        super().__init__(case_sensitive=case_sensitive)
        self.app = app

    async def get_suggestion(self, value: str) -> str | None:
        """Gets a completion from the given possibilities.

        Args:
            value: The current value.

        Returns:
            A valid completion suggestion or `None`.
        """
        if DISABLE_MODELS:
            return "patate" if value.startswith("patate") else None

        assert self.app.models != None
        assert self.app.models.scenario_grams != None

        return ngrams.ngrams_generate(
            value.split(),
            self.app.models.scenario_grams,
        )


if __name__ == "__main__":
    DemoNlpApp().run()
