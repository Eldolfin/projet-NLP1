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
)
from textual.containers import Container, Horizontal
from enum import Enum
import asyncio
from utils import Prediction
from typing import List
import time
import random as rand
from phrases import SCENARIOS
from term_widgets import SpinnerWidget
import basic
import ngrams
import word2vec
from dataclasses import dataclass

# Disable training models and loading dataset for faster ui iteration
DISABLE_MODELS = False


@dataclass
class TrainingStep:
    """Class for keeping track of an item in inventory."""

    name: str
    started_at: float
    ended_at: float

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
            width: 6%;
        }
    """
    time_taken_sec = reactive(0.0)

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
                                SpinnerWidget(),
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
                yield Static(f"Took {self.time_taken_sec}s")
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

        self.add_training_step("Training ngrams")
        self.scenario_grams, self.intent_grams = ngrams.train_ngrams(
            self.ds, X_train, y_train, X_test, y_test
        )
        self.add_training_step("Training bag of words")
        self.bow_vectorizer, self.bow_clf, self.bow_intent_models = (
            basic.basic_train(
                self.ds, X_train, y_train, X_test, y_test, CountVectorizer, Mnb
            )
        )
        self.add_training_step("Training idf")
        self.idf_vectorizer, self.idf_clf, self.idf_intent_models = (
            basic.basic_train(
                self.ds, X_train, y_train, X_test, y_test, Tfv, Lr
            )
        )
        self.add_training_step("Training word2vec")
        self.w2v_model, self.w2v_clf, self.w2v_intent_models = (
            word2vec.w2v_train(self.ds, X_train, y_train, X_test, y_test)
        )
        # self.add_training_step("Training neural network")
        # tf_model, tf_clf, tf_intent_models = tf_train(self.ds)

    def change_state(self, new_state: AppState):
        self.state = new_state
        self.refresh(recompose=True)

    def on_input_changed(self, event: Input.Changed):

        self.input_text = event.value
        before = time.process_time()
        predictions = self.predict_class(self.input_text)
        after = time.process_time()
        self.time_taken_sec = after - before
        table = self.query_one(DataTable)
        table.clear(columns=True)
        table.add_columns("Model", "Scenario", "Intent", "Proba")
        table.zebra_stripes = True
        for prediction in predictions:
            table.add_row(
                prediction.method.capitalize(),
                prediction.scenario,
                prediction.intent,
                str(round(prediction.proba, 2)),
            )

    def predict_class(self, text) -> List[Prediction]:

        if DISABLE_MODELS:
            return [
                Prediction(
                    "random model",
                    rand.choice(SCENARIOS),
                    rand.choice(SCENARIOS) + "/" + rand.choice(SCENARIOS),
                    rand.random() * 100,
                ),
                Prediction(
                    "clown model 🤡",
                    rand.choice(SCENARIOS[:2]),
                    SCENARIOS[0] + "/" + rand.choice(SCENARIOS[:2]),
                    rand.random() * 10,
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
            basic.basic_classify(
                self.ds,
                self.idf_vectorizer,
                self.idf_clf,
                self.idf_intent_models,
                self.input_text,
                "idf",
            ),
            ngrams.ngrams_classify(
                self.ds, self.scenario_grams, self.intent_grams, self.input_text
            ),
            word2vec.w2v_classify(
                self.ds,
                self.w2v_model,
                self.w2v_clf,
                self.w2v_intent_models,
                self.input_text,
                "word2vec",
            ),
            # tf_classify(
            #     ds, tf_model, tf_clf, tf_intent_models, user_input, "word2vec"
            # )
        ]


def main():
    DemoNlpApp().run()
    # not needed?
    # nltk.download("punkt_tab", quiet=True)
    # nltk.download("stopwords", quiet=True)

    # print(
    #     "\nUwU~ je suis Awexa, ton assistante pweferé!!! Toujours wà pour discuter avec twa nya~ (✿˵◕ ‿ ◕˵) \n\n"
    # )

    # for i in range(10):
    # print(ngrams_generate("quelle", scenario_grams, intent_grams, 15))

    # while True:
    #     user_input = input(
    #         "\nEntwe une fwhase à cwassifier, s'il te pwait, nya~ 💖\n> "
    #     ).lower()

    #     if user_input == "quit":
    #         return

    # print("")
    # print(basic_classify(ds, bow_vectorizer, bow_clf, bow_intent_models, user_input, "bow"))
    # print(basic_classify(ds, idf_vectorizer, idf_clf, idf_intent_models, user_input, "idf"))
    # print(w2v_classify(ds, w2v_model, w2v_clf, w2v_intent_models, user_input, "word2vec"))
    # print(
    #     tf_classify(
    #         ds, tf_model, tf_clf, tf_intent_models, user_input, "word2vec"
    #     )
    # )
    # print(ngrams_classify(ds, scenario_grams, intent_grams, user_input))

    # TODO: print this with the consensus (majority of votes between models)
    # f"Sugoi no kawaine!!\n D'apwes la method {self.method}, we pense que tu weux pawler de {self.scenario} et que tu weux plus pwecisement {self.intent} awec une pwoba de {self.proba} (≧◡≦) \n"

    # TODO: add train_nn


if __name__ == "__main__":
    main()
