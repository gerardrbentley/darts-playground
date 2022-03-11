import ast
from datetime import datetime
from inspect import isclass, signature
from pathlib import Path
from typing import Optional

import darts.datasets as ds
import darts.models as models
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import streamlit as st
from darts import TimeSeries
from darts.utils.utils import ModelMode, SeasonalityMode, TrendMode

st.set_page_config(page_title="Darts API Playground", page_icon=":dart:")

seasonality_modes = [*SeasonalityMode]
model_modes = [*ModelMode]
trend_modes = [*TrendMode]
modes = [*model_modes, *seasonality_modes, *trend_modes]

ALL_MODELS = pd.read_csv("darts_models.csv", index_col="Model")
# ALL_MODELS = {x: models.__getattribute__(x) for x in dir(models) if isclass(models.__getattribute__(x))}

ALL_DATASETS = {
    x: ds.__getattribute__(x)
    for x in dir(ds)
    if isclass(ds.__getattribute__(x))
    and x not in ("DatasetLoaderMetadata", "DatasetLoaderCSV")
}
toast = st.empty()


@st.experimental_memo
def load_darts_dataset(dataset_name):
    dataset = ALL_DATASETS.get(dataset_name)()
    timeseries = dataset.load()
    df = timeseries.pd_dataframe()
    return df, timeseries


with st.expander("What is this?"):
    st.markdown(Path("README.md").read_text())

with st.expander("More info on Darts Datasets"):
    for name, dataset in ALL_DATASETS.items():
        st.write(f"#### {name}\n\n{dataset.__doc__}")

with st.expander("More info on Darts Models Compatibility"):
    st.write(ALL_MODELS)

st.sidebar.subheader("Choose a Dataset")
use_example = st.sidebar.checkbox(
    "Use example dataset",
    True,
    help="If checked, will use Darts example dataset. Else requires uploading a CSV",
)
options = {
    "Monthly": ("M", 12),
    "Weekly": ("W", 52),
    "Yearly": ("A", 1),
    "Daily": ("D", 365),
    "Hourly": ("H", 365 * 24),
    "Quarterly": ("Q", 8),
}
if use_example:
    dataset_choice = st.sidebar.selectbox("Dataset", ALL_DATASETS, index=0)
    with st.spinner("Fetching Dataset"):
        toast.info(f"Loading {dataset_choice}")
        df, timeseries = load_darts_dataset(dataset_choice)
        toast.success(f"Loaded {dataset_choice}")
else:
    timeseries = None
    csv_data = st.sidebar.file_uploader("Upload a CSV with Time Series Data")
    delimiter = st.sidebar.text_input(
        "CSV Delimiter",
        value=",",
        max_chars=1,
        help="How your CSV values are separated",
    )

    if csv_data is None:
        st.warning("Upload a CSV to analyze")
        st.stop()

    df = pd.read_csv(csv_data, sep=delimiter)
    with st.expander("Show Raw CSV Data"):
        st.dataframe(df)

    time_col = st.sidebar.selectbox(
        "Time Column",
        df.columns,
        help="Name of the column in your csv with time period data",
    )
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.set_index(time_col, drop=True)

columns = list(df.columns)
if len(columns) == 1:
    value_cols = columns[0]
    st.sidebar.info(f"Univariate Dataset, setting value column to {value_cols}")
else:
    value_cols = st.sidebar.multiselect(
        "Values Column(s)",
        columns,
        columns[0],
        help="Name of column(s) with values to sample and forecast",
    )

if timeseries is None:
    sampling_period = st.sidebar.selectbox(
        "Time Series Period",
        options,
        help="How to define samples. Pandas will sum entries between periods to create a well-formed Time Series",
    )
    freq_string, periods_per_year = options[sampling_period]
    df = df.resample(freq_string).sum()
if len(value_cols) == 0:
    toast.error(f"Must select at least one Value Column to forecast")
    st.stop()
timeseries = TimeSeries.from_dataframe(df, value_cols=value_cols)

st.sidebar.subheader("Choose a Model")
model_choice = st.sidebar.selectbox(
    "Model Selection", ALL_MODELS.index, index=len(ALL_MODELS) - 1
)
model_cls = models.__getattribute__(model_choice)
toast.success(f"Loaded {model_choice}")


if len(timeseries.columns) == 1 and not ALL_MODELS.at[model_choice, "Univariate"]:
    toast.error(f"Model {model_choice} is not capable of handling a Univariate dataset")
    st.stop()
if len(timeseries.columns) > 1 and not ALL_MODELS.at[model_choice, "Multivariate"]:
    toast.error(
        f"Model {model_choice} is not capable of handling a Multivariate dataset. Remove all but one option from Value Columns"
    )
    st.stop()

st.sidebar.subheader("Hand Tune Model Parameters")
model_kwargs = {}
model_args = []
for name, parameter in signature(model_cls.__init__).parameters.items():
    if name != "self":
        if parameter.annotation == int:
            value = st.sidebar.number_input(
                name, 0, 10000, parameter.default, key=name, help=str(parameter)
            )
            model_kwargs[name] = value
        elif parameter.annotation == Optional[ModelMode]:
            value = st.sidebar.selectbox(
                name,
                modes,
                modes.index(parameter.default),
                key=name,
                help=str(parameter),
            )
            model_kwargs[name] = value
        elif parameter.annotation == SeasonalityMode:
            value = st.sidebar.selectbox(
                name,
                seasonality_modes,
                seasonality_modes.index(parameter.default),
                key=name,
                help=str(parameter),
            )
            model_kwargs[name] = value
        elif parameter.annotation == ModelMode:
            value = st.sidebar.selectbox(
                name,
                model_modes,
                model_modes.index(parameter.default),
                key=name,
                help=str(parameter),
            )
            model_kwargs[name] = value
        elif parameter.annotation == TrendMode:
            value = st.sidebar.selectbox(
                name,
                trend_modes,
                trend_modes.index(parameter.default),
                key=name,
                help=str(parameter),
            )
            model_kwargs[name] = value
        elif parameter.annotation == Optional[bool]:
            default_index = 2 if parameter.default is None else int(parameter.default)
            value = st.sidebar.selectbox(
                name, [False, True, None], default_index, key=name, help=str(parameter)
            )
            model_kwargs[name] = value
        elif "kwargs" in name:
            raw_value = st.sidebar.text_area(name, "{}", key=name, help=str(parameter))
            parsed_kwargs = ast.literal_eval(raw_value)
            model_kwargs.update(parsed_kwargs)
        elif name == "autoarima_args":
            raw_value = st.sidebar.text_area(name, "[]", key=name, help=str(parameter))
            model_args = ast.literal_eval(raw_value)
        else:
            raw_value = st.sidebar.text_input(
                name, parameter.default, key=name, help=str(parameter)
            )
            parsed_value = ast.literal_eval(raw_value)
            model_kwargs[name] = parsed_value

with st.expander("Current Model Details"):
    st.write(model_kwargs)
    st.write(model_cls.__init__.__doc__)

model = model_cls(*model_args, **model_kwargs)

st.sidebar.subheader("Customize Training")
num_periods = st.sidebar.slider(
    "Number of validation periods",
    key="cust_period",
    min_value=2,
    max_value=len(timeseries),
    value=36,
    help="How many periods worth of datapoints to exclude from training",
)
if not ALL_MODELS.at[model_choice, "Probabilistic"]:
    st.sidebar.info("Not probabilistic, setting num_samples to 1")
    num_samples = 1
else:
    num_samples = st.sidebar.number_input(
        "Number of prediction samples",
        key="cust_sample",
        min_value=1,
        max_value=10000,
        value=1000,
        help="Number of times a prediction is sampled for a probabilistic model",
    )

st.sidebar.subheader("Customize Plotting")
low_quantile = st.sidebar.slider(
    "Lower Percentile",
    key="cust_low",
    min_value=0.01,
    max_value=0.99,
    value=0.05,
    help="The quantile to use for the lower bound of the plotted confidence interval.",
)
high_quantile = st.sidebar.slider(
    "High Percentile",
    key="cust_high",
    min_value=0.01,
    max_value=0.99,
    value=0.95,
    help="The quantile to use for the upper bound of the plotted confidence interval.",
)


train, val = timeseries[:-num_periods], timeseries[-num_periods:]

toast.info("Training Model")
model.fit(train)
toast.success("Trained Model")

num_predictions = len(val)
toast.info(f"Forecasting {num_predictions} periods")
prediction = model.predict(num_predictions, num_samples=num_samples)
toast.success(f"Forecasted data")

st.subheader("Forecast Plot")

if prediction.is_deterministic:
    prediction_df = prediction.pd_dataframe()
else:
    prediction_df = prediction.quantile_df()

display_data = timeseries.pd_dataframe().rename(lambda c: f"observation_{c}", axis=1).join(prediction_df.rename(lambda c: f"prediction_{c}", axis=1))
st.plotly_chart(px.line(display_data))

custom_fig = plt.figure()
timeseries.plot()

prediction.plot(
    label="forecast", low_quantile=low_quantile, high_quantile=high_quantile
)

plt.legend()
st.pyplot(custom_fig)


with st.expander("Raw Training Data"):
    st.dataframe(train.pd_dataframe())

with st.expander("Forecasted Data"):
    st.dataframe(prediction_df)


@st.experimental_memo
def convert_df(df):
    return df.to_csv().encode("utf-8")


csv = convert_df(prediction_df)
st.download_button(
    label="Download data as CSV",
    data=csv,
    file_name=f"predictions_{model_choice}_{datetime.now().strftime('%Y_%m_%d')}.csv",
    mime="text/csv",
)
