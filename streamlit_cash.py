import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
from datetime import date
import joblib
import torch
from sentence_transformers import util
import numpy as np

# Set up page layout
st.set_page_config(layout="wide")


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


local_css("styles/style.css")

col1, col2 = st.columns([1, 0.5])
with col1:
    st.image("images/MSBA.png", width=300)
with col2:
    st.image("images/osb logo.png", width=300)

# Display the title
st.markdown(
    "<h1 style='text-align: center;'>Influence of Consumer Behavior on Food and Beverage Brand Stocks: Media's Role in Boycotts During the Palestine Conflict</h1>",
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_scaler_model(file_path):
    return joblib.load(file_path)


@st.cache_data(show_spinner=False)
def load_lstm_model(file_path):
    with open(file_path, "rb") as file:
        return joblib.load(file)


@st.cache_data(show_spinner=False)
def load_sa_models():
    return (
        joblib.load("models/bert_sentiment_model.joblib"),
        joblib.load("models/bert_tokenizer.joblib"),
        joblib.load("models/sentiment_analysis_pipeline.joblib"),
    )


@st.cache_data(show_spinner=False)
def load_se_models():
    return (
        joblib.load("models/sentence_transformer_model.joblib"),
        joblib.load("models/queries.joblib"),
        joblib.load("models/query_embeddings.joblib"),
        joblib.load("models/queries_detailed.joblib"),
        joblib.load("models/query_embeddings_detailed.joblib"),
    )


# Load models and scalers
scaler_sbux = load_scaler_model("models/scaler_lstm_sbux.pkl")
model_sbux = load_lstm_model("models/model_lstm_sbux.pkl")

scaler_mcd = load_scaler_model("models/scaler_lstm_mcd.pkl")
model_mcd = load_lstm_model("models/model_lstm_mcd.pkl")

scaler_ko = load_scaler_model("models/scaler_lstm_ko.pkl")
model_ko = load_lstm_model("models/model_lstm_ko.pkl")

scaler_pep = load_scaler_model("models/scaler_lstm_pep.pkl")
model_pep = load_lstm_model("models/model_lstm_pep.pkl")

SA_model, SA_tokenizer, SA_sentiment_analysis = load_sa_models()
(
    SE_model,
    SE_queries,
    SE_query_embeddings,
    SE_queries_detailed,
    SE_query_embeddings_detailed,
) = load_se_models()

# Ensure the embeddings are on CPU
SE_query_embeddings = {
    k: torch.tensor(v, device="cpu") for k, v in SE_query_embeddings.items()
}
SE_query_embeddings_detailed = {
    k: torch.tensor(v, device="cpu") for k, v in SE_query_embeddings_detailed.items()
}


def initialize_session_state(tab_name):
    keys = [
        f"{tab_name}_sentiment",
        f"{tab_name}_semantic_scores",
        f"{tab_name}_stock_prediction",
        f"{tab_name}_prediction_date",
        f"{tab_name}_sentiment_result_visible",
        f"{tab_name}_semantic_result_visible",
        f"{tab_name}_result_visible",  # Add this line
        f"{tab_name}_sentiment_message",
    ]
    for key in keys:
        if key not in st.session_state:
            if "scores" in key:
                st.session_state[key] = {}
            elif "visible" in key:
                st.session_state[key] = False
            else:
                st.session_state[key] = None


def create_inputs(tab_name):
    col1, col2 = st.columns([6, 1])
    with col1:
        date_input = st.date_input("Select a date", date.today())
    with col2:
        st.write("")  # Add some vertical space to align with text area

    col1, col2 = st.columns([6, 1])
    with col1:
        text_input1 = st.text_area(
            "Enter your tweet", height=100, key=f"tweet_input_{tab_name}"
        )
    with col2:
        st.write("")  # Add some vertical space to align with text area
        submit_button = st.button("↑", key=f"submit_tweet_{tab_name}")

    sentiment_result_container = st.container()

    col3, col4 = st.columns([6, 1])
    with col3:
        text_input2 = st.text_area(
            "Enter your article", height=100, key=f"article_input_{tab_name}"
        )
    with col4:
        st.write("")  # Add some vertical space to align with text area
        submit_button2 = st.button("↑", key=f"submit_article_{tab_name}")

    semantic_result_container = st.container()

    stock_button = st.button(
        "Predict Stock",
        key=f"predict_stock_{tab_name}",
        help="Click to predict the stock price",
    )
    stock_pred_container = st.container()

    return (
        date_input,
        text_input1,
        text_input2,
        submit_button,
        submit_button2,
        stock_button,
        sentiment_result_container,
        semantic_result_container,
        stock_pred_container,
    )


def prepare_input_data(data, scaler, seq_length=60):
    if data.ndim == 1:
        data = data.reshape(1, -1)

    scaled_data = scaler.transform(data)
    scaled_data = np.repeat(scaled_data, seq_length, axis=0)
    input_data = np.expand_dims(scaled_data, axis=0)

    return input_data


def truncate_text(text, tokenizer, max_length=512):
    tokens = tokenizer.tokenize(text)
    if len(tokens) > max_length - 2:
        tokens = tokens[: max_length - 2]
    return tokenizer.convert_tokens_to_string(tokens)


@st.cache_data(show_spinner=False)
def load_stock_data(ticker_symbol, period="2y"):
    ticker = yf.Ticker(ticker_symbol)
    data = ticker.history(period=period).reset_index()
    data["Date"] = pd.to_datetime(data["Date"], utc=True)
    return data


def plot_stock_price(data, split_date_str):
    split_date = pd.to_datetime(split_date_str, utc=True)
    data_before = data[data["Date"] <= split_date]
    data_after = data[data["Date"] > split_date]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data_before["Date"],
            y=data_before["Close"],
            mode="lines",
            name="Before",
            line=dict(color="blue"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=data_after["Date"],
            y=data_after["Close"],
            mode="lines",
            name="After",
            line=dict(color="orange"),
        )
    )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        hovermode="x",
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
    )
    st.plotly_chart(fig)


def tab1():
    tab_name = "Starbucks"
    initialize_session_state(tab_name)
    st.title("Starbucks Stock Price (Last 2 Years)")

    data = load_stock_data("SBUX")
    plot_stock_price(data, "2023-10-07")

    st.title("Starbucks Stock Price Prediction")
    (
        date_input,
        text_input1,
        text_input2,
        submit_button,
        submit_button2,
        stock_button,
        sentiment_result_container,
        semantic_result_container,
        stock_pred_container,
    ) = create_inputs(tab_name)

    if submit_button:
        if text_input1:
            truncated_text = truncate_text(text_input1, SA_tokenizer)
            result = SA_sentiment_analysis(truncated_text)

            if result:
                st.session_state[f"{tab_name}_sentiment"] = result[0]["label"]
                if st.session_state[f"{tab_name}_sentiment"] == "LABEL_1":
                    st.session_state[f"{tab_name}_sentiment_message"] = (
                        "This text has a positive sentiment, which might indicate a positive outlook for Starbucks stock."
                    )
                else:
                    st.session_state[f"{tab_name}_sentiment_message"] = (
                        "This text has a negative sentiment, which might indicate a negative outlook for Starbucks stock."
                    )
            else:
                st.session_state[f"{tab_name}_sentiment_message"] = (
                    "Unable to determine sentiment. Please try again with a different text."
                )
            st.session_state[f"{tab_name}_sentiment_result_visible"] = True
        else:
            st.session_state[f"{tab_name}_sentiment_message"] = (
                "Please enter some text before submitting."
            )
            st.session_state[f"{tab_name}_sentiment_result_visible"] = True

    if submit_button2:
        if text_input2:
            text_embedding = SE_model.encode(text_input2, convert_to_tensor=True).cpu()

            semantic_scores = {}
            for key, query_embedding in SE_query_embeddings.items():
                cosine_scores = util.pytorch_cos_sim(text_embedding, query_embedding)[0]
                semantic_scores[key] = cosine_scores.numpy()

            for key, query_embedding in SE_query_embeddings_detailed.items():
                cosine_scores = util.pytorch_cos_sim(text_embedding, query_embedding)[0]
                semantic_scores[key] = cosine_scores.numpy()

            st.session_state[f"{tab_name}_semantic_scores"] = semantic_scores
            st.session_state[f"{tab_name}_semantic_result_visible"] = True
        else:
            st.session_state[f"{tab_name}_semantic_result_visible"] = False

    if stock_button:
        date_input = pd.to_datetime(date_input).tz_localize("UTC").normalize()
        next_day = date_input + pd.Timedelta(days=1)

        if date_input.weekday() >= 5:
            date_input += pd.Timedelta(days=(7 - date_input.weekday()))
        if next_day.weekday() >= 5:
            next_day += pd.Timedelta(days=(7 - next_day.weekday()))

        close_sbux = data[data["Date"].dt.normalize() == date_input]["Close"].values
        if len(close_sbux) == 0:
            st.error("The selected date is not available in the data.")
            return
        close_sbux = close_sbux[-1]

        media_influence = st.session_state[f"{tab_name}_semantic_scores"].get(
            "detailed_media_influence", [0]
        )[0]
        economic_impact = st.session_state[f"{tab_name}_semantic_scores"].get(
            "detailed_economic_impact", [0]
        )[0]
        political_context = st.session_state[f"{tab_name}_semantic_scores"].get(
            "detailed_political_context", [0]
        )[0]

        sentiment_bert = (
            1
            if st.session_state[f"{tab_name}_sentiment"] == "LABEL_1"
            else 0 if st.session_state[f"{tab_name}_sentiment"] == "LABEL_0" else -1
        )

        input_features = np.array(
            [
                close_sbux,
                media_influence,
                economic_impact,
                political_context,
                sentiment_bert,
            ]
        ).reshape(1, -1)
        input_data = prepare_input_data(input_features, scaler_sbux)
        prediction = model_sbux.predict(input_data)

        inverse_prediction = np.zeros((prediction.shape[0], 5))
        inverse_prediction[:, 0] = prediction.flatten()
        inverse_prediction = scaler_sbux.inverse_transform(inverse_prediction)

        st.session_state[f"{tab_name}_stock_prediction"] = inverse_prediction[0][0]
        st.session_state[f"{tab_name}_prediction_date"] = next_day.date()
        st.session_state[f"{tab_name}_result_visible"] = True

    if st.session_state[f"{tab_name}_sentiment_result_visible"]:
        with sentiment_result_container:
            st.write(st.session_state[f"{tab_name}_sentiment_message"])

    if st.session_state[f"{tab_name}_semantic_result_visible"]:
        with semantic_result_container:
            st.write("Semantic similarity scores:")
            scores = list(st.session_state[f"{tab_name}_semantic_scores"].items())
            for key, score in scores:
                st.write(f"{key}: {score}")

    if st.session_state[f"{tab_name}_result_visible"]:
        with stock_pred_container:
            if st.session_state[f"{tab_name}_stock_prediction"] is not None:
                st.write(
                    f"Predicted Close SBUX for {st.session_state[f'{tab_name}_prediction_date']}: {st.session_state[f'{tab_name}_stock_prediction']:.2f} +- 4.5"
                )


def tab2():
    tab_name = "McDonalds"
    initialize_session_state(tab_name)
    st.title("McDonald's Stock Price (Last 2 Years)")

    data = load_stock_data("MCD")
    plot_stock_price(data, "2023-10-07")

    st.title("McDonald's Stock Price Prediction")
    (
        date_input,
        text_input1,
        text_input2,
        submit_button,
        submit_button2,
        stock_button,
        sentiment_result_container,
        semantic_result_container,
        stock_pred_container,
    ) = create_inputs(tab_name)

    if submit_button:
        if text_input1:
            truncated_text = truncate_text(text_input1, SA_tokenizer)
            result = SA_sentiment_analysis(truncated_text)

            if result:
                st.session_state[f"{tab_name}_sentiment"] = result[0]["label"]
                if st.session_state[f"{tab_name}_sentiment"] == "LABEL_1":
                    st.session_state[f"{tab_name}_sentiment_message"] = (
                        "This text has a positive sentiment, which might indicate a positive outlook for McDonald's stock."
                    )
                else:
                    st.session_state[f"{tab_name}_sentiment_message"] = (
                        "This text has a negative sentiment, which might indicate a negative outlook for McDonald's stock."
                    )
            else:
                st.session_state[f"{tab_name}_sentiment_message"] = (
                    "Unable to determine sentiment. Please try again with a different text."
                )
            st.session_state[f"{tab_name}_sentiment_result_visible"] = True
        else:
            st.session_state[f"{tab_name}_sentiment_message"] = (
                "Please enter some text before submitting."
            )
            st.session_state[f"{tab_name}_sentiment_result_visible"] = True

    if submit_button2:
        if text_input2:
            text_embedding = SE_model.encode(text_input2, convert_to_tensor=True).cpu()

            semantic_scores = {}
            for key, query_embedding in SE_query_embeddings.items():
                cosine_scores = util.pytorch_cos_sim(text_embedding, query_embedding)[0]
                semantic_scores[key] = cosine_scores.numpy()

            for key, query_embedding in SE_query_embeddings_detailed.items():
                cosine_scores = util.pytorch_cos_sim(text_embedding, query_embedding)[0]
                semantic_scores[key] = cosine_scores.numpy()

            st.session_state[f"{tab_name}_semantic_scores"] = semantic_scores
            st.session_state[f"{tab_name}_semantic_result_visible"] = True
        else:
            st.session_state[f"{tab_name}_semantic_result_visible"] = False

    if stock_button:
        date_input = pd.to_datetime(date_input).tz_localize("UTC").normalize()
        next_day = date_input + pd.Timedelta(days=1)

        if date_input.weekday() >= 5:
            date_input += pd.Timedelta(days=(7 - date_input.weekday()))
        if next_day.weekday() >= 5:
            next_day += pd.Timedelta(days=(7 - next_day.weekday()))

        close_mcd = data[data["Date"].dt.normalize() == date_input]["Close"].values
        if len(close_mcd) == 0:
            st.error("The selected date is not available in the data.")
            return
        close_mcd = close_mcd[-1]

        media_influence = st.session_state[f"{tab_name}_semantic_scores"].get(
            "detailed_media_influence", [0]
        )[0]
        economic_impact = st.session_state[f"{tab_name}_semantic_scores"].get(
            "detailed_economic_impact", [0]
        )[0]
        political_context = st.session_state[f"{tab_name}_semantic_scores"].get(
            "detailed_political_context", [0]
        )[0]

        sentiment_bert = (
            1
            if st.session_state[f"{tab_name}_sentiment"] == "LABEL_1"
            else 0 if st.session_state[f"{tab_name}_sentiment"] == "LABEL_0" else -1
        )

        input_features = np.array(
            [
                close_mcd,
                media_influence,
                economic_impact,
                political_context,
                sentiment_bert,
            ]
        ).reshape(1, -1)
        input_data = prepare_input_data(input_features, scaler_mcd)
        prediction = model_mcd.predict(input_data)

        inverse_prediction = np.zeros((prediction.shape[0], 5))
        inverse_prediction[:, 0] = prediction.flatten()
        inverse_prediction = scaler_mcd.inverse_transform(inverse_prediction)

        st.session_state[f"{tab_name}_stock_prediction"] = inverse_prediction[0][0]
        st.session_state[f"{tab_name}_prediction_date"] = next_day.date()
        st.session_state[f"{tab_name}_result_visible"] = True

    if st.session_state[f"{tab_name}_sentiment_result_visible"]:
        with sentiment_result_container:
            st.write(st.session_state[f"{tab_name}_sentiment_message"])

    if st.session_state[f"{tab_name}_semantic_result_visible"]:
        with semantic_result_container:
            st.write("Semantic similarity scores:")
            scores = list(st.session_state[f"{tab_name}_semantic_scores"].items())
            for key, score in scores:
                st.write(f"{key}: {score}")

    if st.session_state[f"{tab_name}_result_visible"]:
        with stock_pred_container:
            if st.session_state[f"{tab_name}_stock_prediction"] is not None:
                st.write(
                    f"Predicted Close MCD for {st.session_state[f'{tab_name}_prediction_date']}: {st.session_state[f'{tab_name}_stock_prediction']:.2f} +- 20.3"
                )


def tab3():

    tab_name = "Pepsi"
    initialize_session_state(tab_name)
    st.title("Pepsi Stock Price (Last 2 Years)")

    data = load_stock_data("PEP")
    plot_stock_price(data, "2023-10-07")

    st.title("Pepsi Stock Price Prediction")
    (
        date_input,
        text_input1,
        text_input2,
        submit_button,
        submit_button2,
        stock_button,
        sentiment_result_container,
        semantic_result_container,
        stock_pred_container,
    ) = create_inputs(tab_name)

    if submit_button:
        if text_input1:
            truncated_text = truncate_text(text_input1, SA_tokenizer)
            result = SA_sentiment_analysis(truncated_text)

            if result:
                st.session_state[f"{tab_name}_sentiment"] = result[0]["label"]
                if st.session_state[f"{tab_name}_sentiment"] == "LABEL_1":
                    st.session_state[f"{tab_name}_sentiment_message"] = (
                        "This text has a positive sentiment, which might indicate a positive outlook for McDonald's stock."
                    )
                else:
                    st.session_state[f"{tab_name}_sentiment_message"] = (
                        "This text has a negative sentiment, which might indicate a negative outlook for McDonald's stock."
                    )
            else:
                st.session_state[f"{tab_name}_sentiment_message"] = (
                    "Unable to determine sentiment. Please try again with a different text."
                )
            st.session_state[f"{tab_name}_sentiment_result_visible"] = True
        else:
            st.session_state[f"{tab_name}_sentiment_message"] = (
                "Please enter some text before submitting."
            )
            st.session_state[f"{tab_name}_sentiment_result_visible"] = True

    if submit_button2:
        if text_input2:
            text_embedding = SE_model.encode(text_input2, convert_to_tensor=True).cpu()

            semantic_scores = {}
            for key, query_embedding in SE_query_embeddings.items():
                cosine_scores = util.pytorch_cos_sim(text_embedding, query_embedding)[0]
                semantic_scores[key] = cosine_scores.numpy()

            for key, query_embedding in SE_query_embeddings_detailed.items():
                cosine_scores = util.pytorch_cos_sim(text_embedding, query_embedding)[0]
                semantic_scores[key] = cosine_scores.numpy()

            st.session_state[f"{tab_name}_semantic_scores"] = semantic_scores
            st.session_state[f"{tab_name}_semantic_result_visible"] = True
        else:
            st.session_state[f"{tab_name}_semantic_result_visible"] = False

    if stock_button:
        date_input = pd.to_datetime(date_input).tz_localize("UTC").normalize()
        next_day = date_input + pd.Timedelta(days=1)

        if date_input.weekday() >= 5:
            date_input += pd.Timedelta(days=(7 - date_input.weekday()))
        if next_day.weekday() >= 5:
            next_day += pd.Timedelta(days=(7 - next_day.weekday()))

        close_pep = data[data["Date"].dt.normalize() == date_input]["Close"].values
        if len(close_pep) == 0:
            st.error("The selected date is not available in the data.")
            return
        close_pep = close_pep[-1]

        media_influence = st.session_state[f"{tab_name}_semantic_scores"].get(
            "detailed_media_influence", [0]
        )[0]
        economic_impact = st.session_state[f"{tab_name}_semantic_scores"].get(
            "detailed_economic_impact", [0]
        )[0]
        political_context = st.session_state[f"{tab_name}_semantic_scores"].get(
            "detailed_political_context", [0]
        )[0]

        sentiment_bert = (
            1
            if st.session_state[f"{tab_name}_sentiment"] == "LABEL_1"
            else 0 if st.session_state[f"{tab_name}_sentiment"] == "LABEL_0" else -1
        )

        input_features = np.array(
            [
                close_pep,
                media_influence,
                economic_impact,
                political_context,
                sentiment_bert,
            ]
        ).reshape(1, -1)
        input_data = prepare_input_data(input_features, scaler_pep)
        prediction = model_pep.predict(input_data)

        inverse_prediction = np.zeros((prediction.shape[0], 5))
        inverse_prediction[:, 0] = prediction.flatten()
        inverse_prediction = scaler_pep.inverse_transform(inverse_prediction)

        st.session_state[f"{tab_name}_stock_prediction"] = inverse_prediction[0][0]
        st.session_state[f"{tab_name}_prediction_date"] = next_day.date()
        st.session_state[f"{tab_name}_result_visible"] = True

    if st.session_state[f"{tab_name}_sentiment_result_visible"]:
        with sentiment_result_container:
            st.write(st.session_state[f"{tab_name}_sentiment_message"])

    if st.session_state[f"{tab_name}_semantic_result_visible"]:
        with semantic_result_container:
            st.write("Semantic similarity scores:")
            scores = list(st.session_state[f"{tab_name}_semantic_scores"].items())
            for key, score in scores:
                st.write(f"{key}: {score}")

    if st.session_state[f"{tab_name}_result_visible"]:
        with stock_pred_container:
            if st.session_state[f"{tab_name}_stock_prediction"] is not None:
                st.write(
                    f"Predicted Close PEP for {st.session_state[f'{tab_name}_prediction_date']}: {st.session_state[f'{tab_name}_stock_prediction']:.2f} +- 5.9"
                )


def tab4():
    tab_name = "CocaCola"
    initialize_session_state(tab_name)
    st.title("CocaCola Stock Price (Last 2 Years)")

    data = load_stock_data("KO")
    plot_stock_price(data, "2023-10-07")

    st.title("CocaCola Stock Price Prediction")
    (
        date_input,
        text_input1,
        text_input2,
        submit_button,
        submit_button2,
        stock_button,
        sentiment_result_container,
        semantic_result_container,
        stock_pred_container,
    ) = create_inputs(tab_name)

    if submit_button:
        if text_input1:
            truncated_text = truncate_text(text_input1, SA_tokenizer)
            result = SA_sentiment_analysis(truncated_text)

            if result:
                st.session_state[f"{tab_name}_sentiment"] = result[0]["label"]
                if st.session_state[f"{tab_name}_sentiment"] == "LABEL_1":
                    st.session_state[f"{tab_name}_sentiment_message"] = (
                        "This text has a positive sentiment, which might indicate a positive outlook for McDonald's stock."
                    )
                else:
                    st.session_state[f"{tab_name}_sentiment_message"] = (
                        "This text has a negative sentiment, which might indicate a negative outlook for McDonald's stock."
                    )
            else:
                st.session_state[f"{tab_name}_sentiment_message"] = (
                    "Unable to determine sentiment. Please try again with a different text."
                )
            st.session_state[f"{tab_name}_sentiment_result_visible"] = True
        else:
            st.session_state[f"{tab_name}_sentiment_message"] = (
                "Please enter some text before submitting."
            )
            st.session_state[f"{tab_name}_sentiment_result_visible"] = True

    if submit_button2:
        if text_input2:
            text_embedding = SE_model.encode(text_input2, convert_to_tensor=True).cpu()

            semantic_scores = {}
            for key, query_embedding in SE_query_embeddings.items():
                cosine_scores = util.pytorch_cos_sim(text_embedding, query_embedding)[0]
                semantic_scores[key] = cosine_scores.numpy()

            for key, query_embedding in SE_query_embeddings_detailed.items():
                cosine_scores = util.pytorch_cos_sim(text_embedding, query_embedding)[0]
                semantic_scores[key] = cosine_scores.numpy()

            st.session_state[f"{tab_name}_semantic_scores"] = semantic_scores
            st.session_state[f"{tab_name}_semantic_result_visible"] = True
        else:
            st.session_state[f"{tab_name}_semantic_result_visible"] = False

    if stock_button:
        date_input = pd.to_datetime(date_input).tz_localize("UTC").normalize()
        next_day = date_input + pd.Timedelta(days=1)

        if date_input.weekday() >= 5:
            date_input += pd.Timedelta(days=(7 - date_input.weekday()))
        if next_day.weekday() >= 5:
            next_day += pd.Timedelta(days=(7 - next_day.weekday()))

        close_ko = data[data["Date"].dt.normalize() == date_input]["Close"].values
        if len(close_ko) == 0:
            st.error("The selected date is not available in the data.")
            return
        close_ko = close_ko[-1]

        media_influence = st.session_state[f"{tab_name}_semantic_scores"].get(
            "detailed_media_influence", [0]
        )[0]
        economic_impact = st.session_state[f"{tab_name}_semantic_scores"].get(
            "detailed_economic_impact", [0]
        )[0]
        political_context = st.session_state[f"{tab_name}_semantic_scores"].get(
            "detailed_political_context", [0]
        )[0]

        sentiment_bert = (
            1
            if st.session_state[f"{tab_name}_sentiment"] == "LABEL_1"
            else 0 if st.session_state[f"{tab_name}_sentiment"] == "LABEL_0" else -1
        )

        input_features = np.array(
            [
                close_ko,
                media_influence,
                economic_impact,
                political_context,
                sentiment_bert,
            ]
        ).reshape(1, -1)
        input_data = prepare_input_data(input_features, scaler_ko)
        prediction = model_ko.predict(input_data)

        inverse_prediction = np.zeros((prediction.shape[0], 5))
        inverse_prediction[:, 0] = prediction.flatten()
        inverse_prediction = scaler_ko.inverse_transform(inverse_prediction)

        st.session_state[f"{tab_name}_stock_prediction"] = inverse_prediction[0][0]
        st.session_state[f"{tab_name}_prediction_date"] = next_day.date()
        st.session_state[f"{tab_name}_result_visible"] = True

    if st.session_state[f"{tab_name}_sentiment_result_visible"]:
        with sentiment_result_container:
            st.write(st.session_state[f"{tab_name}_sentiment_message"])

    if st.session_state[f"{tab_name}_semantic_result_visible"]:
        with semantic_result_container:
            st.write("Semantic similarity scores:")
            scores = list(st.session_state[f"{tab_name}_semantic_scores"].items())
            for key, score in scores:
                st.write(f"{key}: {score}")

    if st.session_state[f"{tab_name}_result_visible"]:
        with stock_pred_container:
            if st.session_state[f"{tab_name}_stock_prediction"] is not None:
                st.write(
                    f"Predicted Close PEP for {st.session_state[f'{tab_name}_prediction_date']}: {st.session_state[f'{tab_name}_stock_prediction']:.2f} +- 1.2"
                )


# Create a top menu
tabs = option_menu(
    None,
    ["Starbucks", "McDonalds", "Pepsi", "CocaCola"],
    icons=["cup-hot", "shop", "cup", "cup-straw"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#fafafa"},
        "nav-link": {
            "font-size": "16px",
            "text-align": "left",
            "margin": "0px",
            "--hover-color": "#eee",
        },
        "nav-link-selected": {"background-color": "#840434"},
    },
)

# Render the content of the selected tab
if tabs == "Starbucks":
    tab1()
elif tabs == "McDonalds":
    tab2()
elif tabs == "Pepsi":
    tab3()
elif tabs == "CocaCola":
    tab4()
