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

st.set_page_config(layout="wide")


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


local_css("styles/style.css")

# st.image("aub logo2.png", width=200)
# st.image("osb logo.png", width=200)

col1, col2 = st.columns([1, 0.5])
with col1:
    st.image("images/MSBA.png", width=300)
with col2:
    st.image("images/osb logo.png", width=300)


st.markdown(
    "<h1 style='text-align: center;'>Influence of Consumer Behavior on Food and Beverage Brand Stocks: Media's Role in Boycotts During the Palestine Conflict</h1>",
    unsafe_allow_html=True,
)


def initialize_session_state(tab_name):
    if f"{tab_name}_sentiment" not in st.session_state:
        st.session_state[f"{tab_name}_sentiment"] = None
    if f"{tab_name}_semantic_scores" not in st.session_state:
        st.session_state[f"{tab_name}_semantic_scores"] = {}
    if f"{tab_name}_stock_prediction" not in st.session_state:
        st.session_state[f"{tab_name}_stock_prediction"] = None
    if f"{tab_name}_prediction_date" not in st.session_state:
        st.session_state[f"{tab_name}_prediction_date"] = None


def create_inputs(tab_name):
    col1a, col2a = st.columns([6, 1])
    with col1a:
        date_input = st.date_input(f"Select a date ({tab_name})", date.today())
    with col2a:
        st.write("")

    col1, col2 = st.columns([6, 1])
    with col1:
        text_input1 = st.text_area(
            f"Enter your tweet ({tab_name})", height=100, key=f"{tab_name}_tweet_input"
        )
    with col2:
        st.write("")  # Add some vertical space to align with text area
        submit_button = st.button("↑", key=f"{tab_name}_submit_tweet")

    # Create a placeholder for the sentiment analysis result
    sentiment_result = st.empty()

    col3, col4 = st.columns([6, 1])
    with col3:
        text_input2 = st.text_area(
            f"Enter your article ({tab_name})",
            height=100,
            key=f"{tab_name}_article_input",
        )
    with col4:
        st.write("")  # Add some vertical space to align with text area
        submit_button2 = st.button("↑", key=f"{tab_name}_submit_article")

    semantic_result_1 = st.empty()
    semantic_result_2 = st.empty()
    semantic_result_3 = st.empty()
    semantic_result_4 = st.empty()
    stock_button = st.button(
        "Predict Stock",
        key=f"{tab_name}_predict_stock",
        help="Click to predict the stock price",
    )
    stock_pred = st.empty()

    return (
        date_input,
        text_input1,
        text_input2,
        submit_button,
        submit_button2,
        sentiment_result,
        semantic_result_1,
        semantic_result_2,
        semantic_result_3,
        semantic_result_4,
        stock_button,
        stock_pred,
    )


# Load models and scalers
scaler_sbux = joblib.load("models/scaler_lstm_sbux.pkl")
with open("models/model_lstm_sbux.pkl", "rb") as file:
    model_sbux = joblib.load(file)

scaler_mcd = joblib.load("models/scaler_lstm_mcd.pkl")
with open("models/model_lstm_mcd.pkl", "rb") as file:
    model_mcd = joblib.load(file)

scaler_ko = joblib.load("models/scaler_lstm_ko.pkl")
with open("models/model_lstm_ko.pkl", "rb") as file:
    model_ko = joblib.load(file)

scaler_pep = joblib.load("models/scaler_lstm_pep.pkl")
with open("models/model_lstm_pep.pkl", "rb") as file:
    model_pep = joblib.load(file)

# SA model
SA_model = joblib.load("models/bert_sentiment_model.joblib")
SA_tokenizer = joblib.load("models/bert_tokenizer.joblib")
SA_sentiment_analysis = joblib.load("models/sentiment_analysis_pipeline.joblib")

# SE model
SE_model = joblib.load("models/sentence_transformer_model.joblib")
SE_queries = joblib.load("models/queries.joblib")
SE_query_embeddings = joblib.load("models/query_embeddings.joblib")
SE_queries_detailed = joblib.load("models/queries_detailed.joblib")
SE_query_embeddings_detailed = joblib.load("models/query_embeddings_detailed.joblib")

# Ensure the embeddings are on CPU
SE_query_embeddings = {
    k: torch.tensor(v, device="cpu") for k, v in SE_query_embeddings.items()
}
SE_query_embeddings_detailed = {
    k: torch.tensor(v, device="cpu") for k, v in SE_query_embeddings_detailed.items()
}


def prepare_input_data(data, scaler, seq_length=60):
    # Ensure data is 2D
    if data.ndim == 1:
        data = data.reshape(1, -1)

    # Normalize the data
    scaled_data = scaler.transform(data)

    # Repeat the single row to create a sequence
    scaled_data = np.repeat(scaled_data, seq_length, axis=0)

    # Reshape to the format (1, seq_length, number_of_features)
    input_data = np.expand_dims(scaled_data, axis=0)

    return input_data


def truncate_text(text, tokenizer, max_length=512):
    tokens = tokenizer.tokenize(text)
    # Account for [CLS] and [SEP] tokens
    if len(tokens) > max_length - 2:
        tokens = tokens[: max_length - 2]
    return tokenizer.convert_tokens_to_string(tokens)


def predict_stock(tab_name, ticker_symbol, model, scaler, margin_of_error):
    st.title(f"{ticker_symbol} Stock Price (Last 2 Years)")
    ticker = yf.Ticker(ticker_symbol)
    # get data of the most recent date
    todays_data = ticker.history(period="2y").reset_index()

    todays_data["Date"] = pd.to_datetime(todays_data["Date"], utc=True)

    split_date = pd.to_datetime("2023-10-07", utc=True)
    data_before = todays_data[todays_data["Date"] <= split_date]
    data_after = todays_data[todays_data["Date"] > split_date]

    # Create a time plot using Plotly
    fig = go.Figure()

    # Add trace for data before Oct 7, 2023
    fig.add_trace(
        go.Scatter(
            x=data_before["Date"],
            y=data_before["Close"],
            mode="lines",
            name="Before",
            line=dict(color="blue"),
        )
    )

    # Add trace for data after Oct 7, 2023
    fig.add_trace(
        go.Scatter(
            x=data_after["Date"],
            y=data_after["Close"],
            mode="lines",
            name="After",
            line=dict(color="orange"),
        )
    )

    # Customize the layout
    fig.update_layout(
        # title=f"{ticker_symbol} Stock Price (Last 2 Years)",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        hovermode="x",
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
    )

    # Display the plot
    st.plotly_chart(fig)

    st.title(f"{ticker_symbol} Stock Price Prediction")
    (
        date_input,
        text_input1,
        text_input2,
        submit_button,
        submit_button2,
        sentiment_result,
        semantic_result_1,
        semantic_result_2,
        semantic_result_3,
        semantic_result_4,
        stock_button,
        stock_pred,
    ) = create_inputs(tab_name)

    if submit_button:
        if text_input1:
            # Process the input text
            truncated_text = truncate_text(text_input1, SA_tokenizer)

            # Use the SA sentiment analysis pipeline
            result = SA_sentiment_analysis(truncated_text)

            if result:
                st.session_state[f"{tab_name}_sentiment"] = result[0]["label"]

                # Interpret the result
                if st.session_state[f"{tab_name}_sentiment"] == "LABEL_1":
                    sentiment_result.write(
                        "This text has a positive sentiment, which might indicate a positive outlook for the stock.",
                    )
                else:
                    sentiment_result.write(
                        "This text has a negative sentiment, which might indicate a negative outlook for the stock.",
                    )
            else:
                sentiment_result.write(
                    "Unable to determine sentiment. Please try again with a different text."
                )
        else:
            sentiment_result.write("Please enter some text before submitting.")

    if submit_button2:
        if text_input2:
            # Embed the input text using the model
            text_embedding = SE_model.encode(text_input2, convert_to_tensor=True).cpu()

            # Compute cosine similarity between the input text embedding and the saved query embeddings
            semantic_scores = {}
            for key, query_embedding in SE_query_embeddings.items():
                cosine_scores = util.pytorch_cos_sim(text_embedding, query_embedding)[0]
                semantic_scores[key] = cosine_scores.numpy()

            for key, query_embedding in SE_query_embeddings_detailed.items():
                cosine_scores = util.pytorch_cos_sim(text_embedding, query_embedding)[0]
                semantic_scores[key] = cosine_scores.numpy()

            st.session_state[f"{tab_name}_semantic_scores"] = semantic_scores

            # Display the semantic similarity results
            semantic_result_1.write("Semantic similarity scores:")
            scores = list(semantic_scores.items())
            if len(scores) > 0:
                for key, score in scores[: len(scores) // 4]:
                    semantic_result_1.write(f"{key}: {score}")
                for key, score in scores[len(scores) // 4 : len(scores) // 2]:
                    semantic_result_2.write(f"{key}: {score}")
                for key, score in scores[len(scores) // 2 : 3 * len(scores) // 4]:
                    semantic_result_3.write(f"{key}: {score}")
                for key, score in scores[3 * len(scores) // 4 :]:
                    semantic_result_4.write(f"{key}: {score}")

    if stock_button:
        date_input = pd.to_datetime(date_input).tz_localize("UTC").normalize()

        # Find the next day after the provided date
        next_day = date_input + pd.Timedelta(days=1)

        # Check if the input date or the next day is a weekend
        if date_input.weekday() >= 5:  # If date_input is Saturday (5) or Sunday (6)
            date_input += pd.Timedelta(days=(7 - date_input.weekday()))
        if next_day.weekday() >= 5:  # If next_day is Saturday (5) or Sunday (6)
            next_day += pd.Timedelta(days=(7 - next_day.weekday()))

        # Get the closing price of the selected date
        close_price = todays_data[todays_data["Date"].dt.normalize() == date_input][
            "Close"
        ].values
        if len(close_price) == 0:
            st.error("The selected date is not available in the data.")
            return
        close_price = close_price[-1]

        media_influence = st.session_state[f"{tab_name}_semantic_scores"].get(
            "detailed_media_influence", [0]
        )[0]
        economic_impact = st.session_state[f"{tab_name}_semantic_scores"].get(
            "detailed_economic_impact", [0]
        )[0]
        political_context = st.session_state[f"{tab_name}_semantic_scores"].get(
            "detailed_political_context", [0]
        )[0]

        # Convert sentiment to numerical value
        if st.session_state[f"{tab_name}_sentiment"] == "LABEL_1":
            sentiment_bert = 1
        elif st.session_state[f"{tab_name}_sentiment"] == "LABEL_0":
            sentiment_bert = 0
        else:
            sentiment_bert = -1  # Default value if sentiment is not set

        input_features = np.array(
            [
                close_price,
                media_influence,
                economic_impact,
                political_context,
                sentiment_bert,
            ]
        ).reshape(1, -1)

        # Prepare the input data
        input_data = prepare_input_data(input_features, scaler)

        # Make the prediction
        prediction = model.predict(input_data)

        # Inverse transform the prediction to original scale
        inverse_prediction = np.zeros((prediction.shape[0], 5))
        inverse_prediction[:, 0] = prediction.flatten()
        inverse_prediction = scaler.inverse_transform(inverse_prediction)

        # Store the prediction and the date in the session state
        st.session_state[f"{tab_name}_stock_prediction"] = inverse_prediction[0][0]
        st.session_state[f"{tab_name}_prediction_date"] = next_day.date()

    # Display the stored prediction
    if st.session_state[f"{tab_name}_stock_prediction"] is not None:
        stock_pred.write(
            f"Predicted Close {ticker_symbol} for {st.session_state[f'{tab_name}_prediction_date']}: {st.session_state[f'{tab_name}_stock_prediction']:.2f} +- {margin_of_error}"
        )


def tab1():
    initialize_session_state("sbux")
    predict_stock("sbux", "SBUX", model_sbux, scaler_sbux, 16.39)


def tab2():
    initialize_session_state("mcd")
    predict_stock("mcd", "MCD", model_mcd, scaler_mcd, 46.43)


def tab3():
    initialize_session_state("pep")
    predict_stock("pep", "PEP", model_pep, scaler_pep, 12.44)


def tab4():
    initialize_session_state("ko")
    predict_stock("ko", "KO", model_ko, scaler_ko, 0.46)


# Create a top menu
with st.container():
    st.markdown("<style>.custom-option-menu {}</style>", unsafe_allow_html=True)
    selected = option_menu(
        menu_title=None,
        options=["Starbucks", "McDonalds", "Pepsi", "CocaCola"],
        icons=["cup-hot", "shop", "cup-straw", "cup-straw"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "#fafafa"},
            "icon": {"color": "#840434", "font-size": "25px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "0px",
                "--hover-color": "#eee",
            },
            "nav-link-selected": {"background-color": "#840434"},
        },
    )
    st.markdown("</div>", unsafe_allow_html=True)

# Render the content of the selected tab
if selected == "Starbucks":
    tab1()
elif selected == "McDonalds":
    tab2()
elif selected == "Pepsi":
    tab3()
elif selected == "CocaCola":
    tab4()
