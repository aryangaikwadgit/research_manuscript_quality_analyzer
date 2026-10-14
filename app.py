import streamlit as st

from src.predict_distilbert import DistilBertPredictor

# configure the page
st.set_page_config(
    page_title="Research Manuscript Quality Analyzer", page_icon="📝", layout="centered"
)


# set the final evaluated model path
MODEL_PATH = r"models\distilbert_final\run_20261013_180354"


# load the model once and reuse it
@st.cache_resource
def load_predictor():
    return DistilBertPredictor(MODEL_PATH)


# display the application header
st.title("📝 Research Manuscript Quality Analyzer")

st.write(
    "Analyze the linguistic acceptability of an English sentence "
    "using a fine-tuned DistilBERT model."
)

st.info(
    "This prototype evaluates linguistic acceptability only. "
    "It does not verify research facts, methodology, citations, "
    "plagiarism, or overall manuscript quality."
)


# load the model and handle loading errors
try:
    predictor = load_predictor()
except Exception as error:
    st.error(f"Could not load the model: {error}")
    st.stop()


# accept sentence input
sentence = st.text_area(
    "Enter a sentence",
    placeholder=("Example: The scientist conducted the experiment carefully."),
    height=130,
    max_chars=10000,
)


# run prediction when the button is clicked
if st.button("Analyze Sentence", type="primary"):

    if not sentence.strip():
        st.warning("Please enter a sentence first.")

    else:
        with st.spinner("Analyzing sentence..."):
            try:
                result = predictor.predict_one(sentence)

                st.subheader("Prediction")

                if result["label"] == 1:
                    st.success("Acceptable")
                else:
                    st.error("Unacceptable")

                st.caption(
                    "The scores below are model softmax outputs, "
                    "not calibrated confidence estimates."
                )

                st.subheader("Class scores")

                unacceptable = result["probabilities"]["Unacceptable"]
                acceptable = result["probabilities"]["Acceptable"]

                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Unacceptable", f"{unacceptable:.2%}")
                    st.progress(unacceptable)

                with col2:
                    st.metric("Acceptable", f"{acceptable:.2%}")
                    st.progress(acceptable)

            except Exception as error:
                st.error(f"Prediction failed: {error}")


# display project information
st.divider()

st.caption(
    "Model: DistilBERT base uncased | "
    "Dataset: CoLA | "
    "Task: Binary linguistic acceptability classification"
)
