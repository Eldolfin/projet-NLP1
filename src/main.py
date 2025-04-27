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
from textual.widget import Widget
from textual.containers import Container
from enum import Enum
import asyncio
import basic
from utils import Prediction
from typing import List
import ngrams

# Disable training models and loading dataset for faster ui iteration
DISABLE_MODELS = False


class AppState(Enum):
    LOADING_DATASET = 1
    TRAINING_MODELS = 2
    INFERENCE_INPUT = 3


class DemoNlpApp(App):
    input_text = reactive("")
    state = reactive(AppState.LOADING_DATASET)
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
    """

    def compose(self) -> ComposeResult:
        yield Header()
        yield Footer()
        match self.state:
            case AppState.LOADING_DATASET:
                yield Container(
                    Static("1. Loading dataset", id="loading-state"),
                    LoadingIndicator(name="Loading dataset"),
                    id="content",
                )
            case AppState.TRAINING_MODELS:
                yield Container(
                    Static("2. Training models", id="loading-state"),
                    LoadingIndicator(name="Training models"),
                    id="content",
                )
            case AppState.INFERENCE_INPUT:
                yield Container(
                    Input(placeholder="Type here..."),
                    DataTable(),
                    id="content",
                )

    async def on_mount(self) -> None:
        self.title = "NLP project demo"
        self.sub_title = "Alexa task classifier 🤖 🔥"
        self.load_dataset()

    @work(exclusive=True)
    async def load_dataset(self):
        from datasets import load_dataset

        if DISABLE_MODELS:
            await asyncio.sleep(0.5)
        else:
            self.ds = await asyncio.to_thread(
                load_dataset, "AmazonScience/massive", "fr-FR"
            )
        self.change_state(AppState.TRAINING_MODELS)
        self.train_models()

    @work(exclusive=True)
    async def train_models(self):
        if DISABLE_MODELS:
            await asyncio.sleep(0.5)
        else:
            await asyncio.to_thread(self.train_models_blocking)
        self.change_state(AppState.INFERENCE_INPUT)

    def train_models_blocking(self):
        from word2vec import w2v_train, w2v_classify
        from transformer import tf_train, tf_classify
        from sklearn.feature_extraction.text import (
            CountVectorizer,
            TfidfVectorizer as Tfv,
        )
        from sklearn.linear_model import LogisticRegression as Lr
        from sklearn.naive_bayes import MultinomialNB as Mnb

        X_train = self.ds["train"]["utt"]
        y_train = self.ds["train"]["scenario"]
        X_test = self.ds["test"]["utt"]
        y_test = self.ds["test"]["scenario"]

        # Train models
        self.scenario_grams, self.intent_grams = ngrams.train_ngrams(
            self.ds, X_train, y_train, X_test, y_test
        )
        self.bow_vectorizer, self.bow_clf, self.bow_intent_models = (
            basic.basic_train(
                self.ds, X_train, y_train, X_test, y_test, CountVectorizer, Mnb
            )
        )
        self.idf_vectorizer, self.idf_clf, self.idf_intent_models = (
            basic.basic_train(
                self.ds, X_train, y_train, X_test, y_test, Tfv, Lr
            )
        )
        # w2v_model, w2v_clf, w2v_intent_models = w2v_train(
        #     self.ds, X_train, y_train, X_test, y_test
        # )
        # tf_model, tf_clf, tf_intent_models = tf_train(self.ds)

    def change_state(self, new_state: AppState):
        self.state = new_state
        self.refresh(recompose=True)

    def on_input_changed(self, event: Input.Changed):

        self.input_text = event.value
        predictions = self.predict_class(self.input_text)
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
            # w2v_classify(
            #     ds, w2v_model, w2v_clf, w2v_intent_models, user_input, "word2vec"
            # )
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
