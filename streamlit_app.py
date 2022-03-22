import ast
import pickle
from datetime import datetime
from inspect import isclass, signature
from functools import partial
from pathlib import Path
from typing import Any, List, Optional

from darts.utils.statistics import plot_hist
import numpy as np
import darts.datasets as ds
import darts.models as models
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import streamlit as st
from darts import TimeSeries
from darts.utils.utils import ModelMode, SeasonalityMode, TrendMode
import darts.utils.statistics as statistics
import darts.metrics as metrics

st.set_page_config(page_title="Darts API Playground", page_icon=":dart:")
toast = st.empty()
for key, values in st.experimental_get_query_params().items():
    value = values[0] if len(values) == 1 else values
    if key not in st.session_state:
        st.session_state[key] = value


def update_state(key: str, value: Optional[List[Any]] = None):
    if value is not None:
        st.session_state[key] = value
    new_args = {**st.experimental_get_query_params(), key: st.session_state[key]}
    st.experimental_set_query_params(**new_args)


NON_DTW_METRICS = {
    x: metrics.__getattribute__(x)
    for x in dir(metrics)
    if not x.startswith("__") and x not in ("metrics", "dtw_metric")
}
DTW_METRICS = {
    f"dtw_{key}": partial(metrics.dtw_metric, metric=fn)
    for key, fn in NON_DTW_METRICS.items()
    if key != "mase"  # no time warp
}

ALL_METRICS = {**NON_DTW_METRICS, **DTW_METRICS}
SEASONALITY_MODES = [*SeasonalityMode]
MODEL_MODES = [*ModelMode]
TREND_MODES = [*TrendMode]
ALL_MODES = [*MODEL_MODES, *SEASONALITY_MODES, *TREND_MODES]

ALL_MODELS = pd.read_csv("darts_models.csv", index_col="Model")

ALL_DATASETS = {
    name: attr
    for name, attr in vars(ds).items()
    if isclass(attr) and name not in ("DatasetLoaderMetadata", "DatasetLoaderCSV")
}
DS_NAMES = ["Custom"] + [
    name
    for name, attr in vars(ds).items()
    if isclass(attr) and name not in ("DatasetLoaderMetadata", "DatasetLoaderCSV")
]

MODIFY_STATISTICS = [
    "fill_missing_values",
    #   "remove_from_series",  # requires some fidgeting
    "remove_seasonality",
    "remove_trend",
]
RESULT_STATISTICS = [
    "check_seasonality",
    "extract_trend_and_seasonality",
    #   "granger_causality_tests",  # requires some fidgeting
    "stationarity_test_adf",
    "stationarity_test_kpss"
    #   "stationarity_tests"  # both other tests
]
PLOT_STATISTICS = [
    "plot_acf",
    "plot_hist",
    "plot_pacf",
    #   "plot_residuals_analysis",  # Can't display easily
]


@st.experimental_memo
def load_darts_dataset(dataset_name):
    dataset_cls = getattr(ds, dataset_name)
    timeseries = dataset_cls().load()
    df = timeseries.pd_dataframe()
    return df, timeseries


with st.expander("What is this?"):
    st.markdown(Path("README.md").read_text())

with st.expander("More info on Darts Datasets"):
    ds_name = st.selectbox("See Docs for Dataset:", DS_NAMES[1:], key="ds_doc")
    ds_for_doc = getattr(ds, ds_name)
    st.write(f"#### {ds_name}")
    st.text(ds_for_doc)
    st.text(ds_for_doc.__doc__)

with st.expander("More info on Darts Models"):
    st.write(ALL_MODELS)
    model_name = st.selectbox("See Docs for Model:", ALL_MODELS.index, key="model_doc")
    model_for_doc = getattr(models, model_name)
    st.write(f"#### {model_name}")
    st.text(model_for_doc)
    st.text(model_for_doc.__init__.__doc__)

with st.expander("More info on Darts Metrics"):
    metric_name = st.selectbox("See Docs for Metric:", ALL_METRICS, key="metric_doc")
    if "dtw" in metric_name:
        st.info(
            f"Dynamic Time Warp applies to all metrics. Select {metric_name.replace('dtw_', '')} to see docs on it"
        )
        metric_name = "dtw_metric"
    metric_for_doc = getattr(metrics, metric_name)
    st.write(f"#### {metric_name}")
    st.text(metric_for_doc)
    st.text(metric_for_doc.__doc__)


@st.experimental_memo
def load_csv_data(csv_data, delimiter):
    return pd.read_csv(csv_data, sep=delimiter)


st.sidebar.subheader("Choose a Dataset")
if "dataset" not in st.session_state:
    toast.info(f"Setting Default Dataset")
    update_state("dataset", DS_NAMES[1])


def update_dataset():
    if "value_column" in st.session_state:
        new_args = st.experimental_get_query_params()
        del st.session_state["value_column"]
        del new_args["value_column"]
        st.experimental_set_query_params(**new_args)
    update_state("dataset")


st.sidebar.selectbox(
    "Dataset", DS_NAMES, index=0, key="dataset", on_change=update_dataset
)
if st.session_state.dataset != "Custom":
    with st.spinner("Fetching Dataset"):
        toast.info(f"Loading {st.session_state.dataset}")
        df, timeseries = load_darts_dataset(st.session_state.dataset)
        toast.success(f"Loaded {st.session_state.dataset}")
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

    df = load_csv_data(csv_data, delimiter)
    with st.expander("Show Raw CSV Data"):
        st.dataframe(df)

    time_col = st.sidebar.selectbox(
        "Time Column",
        df.columns,
        help="Name of the column in your csv with time step values",
    )
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.set_index(time_col, drop=True)


columns = list(df.columns)
if "value_column" not in st.session_state:
    update_state("value_column", columns[0])
if len(columns) == 1:
    value_cols = columns[0]
    st.sidebar.info(f"Univariate Dataset, setting value column to {value_cols}")
else:
    value_cols = st.sidebar.multiselect(
        "Values Column(s)",
        columns,
        columns[0],
        key="value_column",
        help="Name of column(s) with values to sample and forecast",
        on_change=update_state,
        args=("value_column",),
    )

if timeseries is None:
    options = {
        "Monthly": ("M", 12),
        "Weekly": ("W", 52),
        "Yearly": ("A", 1),
        "Daily": ("D", 365),
        "Hourly": ("H", 365 * 24),
        "Quarterly": ("Q", 8),
    }
    sampling_period = st.sidebar.selectbox(
        "Time Series Period",
        options,
        help="How to define samples. Pandas will sum entries between time steps to create a well-formed Time Series",
    )
    freq_string, periods_per_year = options[sampling_period]
    df = df.resample(freq_string).sum()
if len(value_cols) == 0:
    toast.error(f"Must select at least one Value Column to forecast")
    st.stop()
timeseries = TimeSeries.from_dataframe(df, value_cols=value_cols)

st.sidebar.subheader("Choose a Model")
model_choice = st.sidebar.selectbox(
    "Model Selection",
    ALL_MODELS.index,
    2,
    help="Open 'Current Model Details' for documentation on parameters",
    key="model",
    on_change=update_state,
    args=("model",),
)
model_cls = getattr(models, model_choice)
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
            if name in st.session_state:
                st.session_state[name] = int(st.session_state[name])
            value = st.sidebar.number_input(
                name,
                0,
                value=parameter.default,
                key=name,
                help=str(parameter),
                on_change=update_state,
                args=(name,),
            )
            model_kwargs[name] = value
        elif parameter.annotation == Optional[ModelMode]:
            value = st.sidebar.selectbox(
                name,
                ALL_MODES,
                ALL_MODES.index(parameter.default),
                key=name,
                help=str(parameter),
            )
            model_kwargs[name] = value
        elif parameter.annotation == SeasonalityMode:
            value = st.sidebar.selectbox(
                name,
                SEASONALITY_MODES,
                SEASONALITY_MODES.index(parameter.default),
                key=name,
                help=str(parameter),
            )
            model_kwargs[name] = value
        elif parameter.annotation == ModelMode:
            value = st.sidebar.selectbox(
                name,
                MODEL_MODES,
                MODEL_MODES.index(parameter.default),
                key=name,
                help=str(parameter),
            )
            model_kwargs[name] = value
        elif parameter.annotation == TrendMode:
            value = st.sidebar.selectbox(
                name,
                TREND_MODES,
                TREND_MODES.index(parameter.default),
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
                name,
                parameter.default,
                key=name,
                help=str(parameter),
                on_change=update_state,
                args=(name,),
            )
            parsed_value = ast.literal_eval(raw_value)
            model_kwargs[name] = parsed_value

with st.expander("Explore Current Dataset"):
    analysis_choices = st.multiselect(
        "Analysis Methods",
        RESULT_STATISTICS,
        ["check_seasonality"],
        key="analysis_choices",
    )
    if len(analysis_choices) and not timeseries.is_univariate:
        st.warning(
            f"Can't run analysis on Multivariate Time Series. Please choose only one Value column to use this."
        )
    else:
        for analysis in analysis_choices:
            try:
                analysis_fn = getattr(statistics, analysis)
                value = analysis_fn(timeseries)
                st.subheader(analysis)
                st.text(analysis_fn.__doc__.split("Parameters\n")[0])
                if analysis == "check_seasonality":
                    is_seasonal, seasonality_period = value
                    if is_seasonal:
                        st.info(
                            f"Time Series is seasonal with period {seasonality_period}"
                        )
                    else:
                        st.warning(
                            f"Time Series is not seasonal with an inferred period."
                        )
                elif analysis == "extract_trend_and_seasonality":
                    trend, seasonal = value
                    st.subheader("Trend Component")
                    st.plotly_chart(
                        px.line(
                            trend.pd_dataframe().rename(lambda c: f"trend_{c}", axis=1)
                        )
                    )
                    st.subheader("Seasonality Component")
                    st.plotly_chart(
                        px.line(
                            seasonal.pd_dataframe().rename(
                                lambda c: f"seasonal_{c}", axis=1
                            )
                        )
                    )
                elif analysis == "stationarity_test_adf":
                    adf, pvalue, usedlag, nobs, critical, icbest = value
                    st.metric("adf", adf)
                    st.metric("pvalue", pvalue)
                    st.metric("usedlag", usedlag)
                    st.metric("nobs", nobs)
                    st.write(critical)
                    st.metric("icbest", icbest)
                    st.text(analysis_fn.__doc__.split("Returns\n")[1])
                elif analysis == "stationarity_test_kpss":
                    kpss_stat, pvalue, lags, critical = value
                    st.metric("kpss_stat", kpss_stat)
                    st.metric("pvalue", pvalue)
                    st.metric("lags", lags)
                    st.write(critical)
                    st.text(analysis_fn.__doc__.split("Returns\n")[1])
            except Exception as e:
                st.error(str(e))

    plot_choices = st.multiselect(
        "Statistic Plots", PLOT_STATISTICS, ["plot_acf"], key="plot_choices"
    )
    for plot in plot_choices:
        st.subheader(plot)
        fig = plt.figure()
        axis = plt.gca()
        plot_fn = getattr(statistics, plot)
        st.text(plot_fn.__doc__.split("Parameters\n")[0])
        if plot != "plot_hist" and not timeseries.is_univariate:
            st.warning(
                f"Can't run {plot} on Multivariate Time Series. Please choose only one Value column to use this."
            )
            continue
        try:
            if plot != "plot_hist":
                plot_fn(timeseries, axis=axis)
            else:
                plot_fn(timeseries, ax=axis)
        except Exception as e:
            st.error(str(e))
            continue

        st.pyplot(fig)
with st.expander("Current Model Details"):
    st.write(model_kwargs)
    st.write(model_cls.__init__.__doc__)

st.sidebar.subheader("Customize Training")
if "forecast_horizon" not in st.session_state:
    update_state("forecast_horizon", 1)
else:
    st.session_state["forecast_horizon"] = int(st.session_state["forecast_horizon"])
forecast_horizon = st.sidebar.number_input(
    "Forecast Horizon",
    key="forecast_horizon",
    min_value=1,
    max_value=len(timeseries),
    help="(For Backtest and Historical Forecast) How many time steps separate the prediction time from the forecast time",
    on_change=update_state,
    args=("forecast_horizon",),
)
if "stride" not in st.session_state:
    update_state("stride", 1)
else:
    st.session_state["stride"] = int(st.session_state["stride"])
stride = st.sidebar.number_input(
    "Historical Forecast Stride",
    key="stride",
    min_value=1,
    max_value=forecast_horizon,
    help="(For Backtest and Historical Forecast) How many time steps between two consecutive predictions",
    on_change=update_state,
    args=("stride",),
)
st.sidebar.info(
    f"Model {model_choice} not a Neural Network. Will be retrained at every backtest stride."
)
historical_forecast_retrain = True
if "num_predictions" not in st.session_state:
    update_state("num_predictions", 36)
else:
    st.session_state["num_predictions"] = int(st.session_state["num_predictions"])
num_predictions = st.sidebar.number_input(
    "Number of validation time steps",
    key="num_predictions",
    min_value=2,
    max_value=len(timeseries),
    help="How many time steps worth of datapoints to exclude from training",
    on_change=update_state,
    args=("num_predictions",),
)
if not ALL_MODELS.at[model_choice, "Probabilistic"]:
    st.sidebar.info(
        f"Model {model_choice} not probabilistic (is deterministic). One line will be plotted."
    )
    num_samples = 1
    low_quantile = 0.1
    high_quantile = 0.9
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
        key="low_quantile",
        min_value=0.01,
        max_value=0.99,
        value=0.05,
        help="The quantile to use for the lower bound of the plotted confidence interval.",
    )
    mid_quantile = st.sidebar.slider(
        "Lower Percentile",
        key="mid_quantile",
        min_value=0.01,
        max_value=0.99,
        value=0.5,
        help="The quantile to use for the center of the plotted confidence interval.",
    )
    high_quantile = st.sidebar.slider(
        "High Percentile",
        key="high_quantile",
        min_value=0.01,
        max_value=0.99,
        value=0.95,
        help="The quantile to use for the upper bound of the plotted confidence interval.",
    )


@st.experimental_memo
def get_prediction(model_cls, model_args, model_kwargs, train, num_predictions):
    model = model_cls(*model_args, **model_kwargs)
    model.fit(TimeSeries.from_dataframe(train))
    prediction = model.predict(num_predictions, num_samples=num_samples)
    return prediction


@st.experimental_memo
def get_historical_forecast(
    model_cls,
    model_args,
    model_kwargs,
    timeseries,
    start,
    forecast_horizon,
    stride,
    retrain,
):
    model = model_cls(*model_args, **model_kwargs)
    historical_forecast = model.historical_forecasts(
        TimeSeries.from_dataframe(timeseries),
        start=start,
        forecast_horizon=forecast_horizon,
        stride=stride,
        retrain=retrain,
        overlap_end=False,
        last_points_only=True,
    )
    historical_df = historical_forecast.pd_dataframe()
    return historical_forecast, historical_df


@st.experimental_memo
def get_backtest(
    model_cls,
    model_args,
    model_kwargs,
    timeseries_df,
    start,
    forecast_horizon,
    stride,
    retrain,
    metric_name,
    reduction,
):
    model = model_cls(*model_args, **model_kwargs)
    metric_fn = ALL_METRICS[metric_name]
    return model.backtest(
        TimeSeries.from_dataframe(timeseries_df),
        start=start,
        forecast_horizon=forecast_horizon,
        stride=stride,
        retrain=retrain,
        metric=metric_fn,
        reduction=reduction,
    )


train, val = timeseries[:-num_predictions], timeseries[-num_predictions:]
toast.info("Training Model")
with st.spinner("Training..."):
    try:
        prediction = get_prediction(
            model_cls, model_args, model_kwargs, train.pd_dataframe(), num_predictions
        )
    except ValueError as e:
        if "lags" in str(e):
            st.error(str(e))
            st.stop()
        raise e
toast.success(f"Forecasted data")
if prediction.is_deterministic:
    prediction_df = prediction.pd_dataframe()
else:
    prediction_df = prediction.quantiles_df([low_quantile, mid_quantile, high_quantile])

toast.info("Historical Forecast Running")
with st.spinner("Historical Forecast..."):
    historical_forecast, historical_df = get_historical_forecast(
        model_cls,
        model_args,
        model_kwargs,
        timeseries.pd_dataframe(),
        timeseries.n_timesteps - num_predictions,
        forecast_horizon,
        stride,
        historical_forecast_retrain,
    )
toast.success("Historical Forecast Finished")

display_data = (
    timeseries.pd_dataframe()
    .rename(lambda c: f"observation_{c}", axis=1)
    .join(prediction_df.rename(lambda c: f"prediction_{c}", axis=1))
)
st.subheader("Data and Forecast Plot")
st.checkbox(
    "Show Historical Forecast",
    value=True,
    key="show_historical",
    help="Starting from the end of the training set, incrementally refit the model on new future values.",
)
if st.session_state.show_historical:
    display_data = display_data.join(
        historical_df.rename(lambda c: f"historical_forecast_{c}", axis=1)
    )
st.plotly_chart(px.line(display_data))
st.subheader("View Error Metrics Over Validation Periods")
if "metric" not in st.session_state:
    update_state("metric", ["mape"])
elif isinstance(st.session_state.metric, str):
    st.session_state.metric = [st.session_state.metric]

metric_choices = st.multiselect(
    "Scoring Metrics",
    ALL_METRICS,
    key="metric",
    help="Which metric functions are used to score the predictions against the ground truth values",
    on_change=update_state,
    args=("metric",),
)
if len(metric_choices):
    raw_scores = []
    all_backtests = {}
    for metric_name in metric_choices:
        try:
            metric_fn = ALL_METRICS[metric_name]
            if "mase" in metric_name:
                value = metric_fn(val, prediction, train)
            else:
                value = metric_fn(val, prediction)
                all_backtests[metric_name] = get_backtest(
                    model_cls,
                    model_args,
                    model_kwargs,
                    timeseries.pd_dataframe(),
                    start=timeseries.n_timesteps - num_predictions,
                    forecast_horizon=forecast_horizon,
                    stride=stride,
                    retrain=historical_forecast_retrain,
                    metric_name=metric_name,
                    reduction=None,
                )
            raw_scores.append(
                {
                    "Metric": metric_name,
                    "Value": value,
                    "Description": metric_fn.__doc__.splitlines()[0],
                }
            )
        except Exception as e:
            st.warning(
                f"Metric {metric_name} error: {str(e)}\nDescription {metric_fn.__doc__.splitlines()[0]}{'' if 'stochastic' not in str(e) else ' Try a probabilistic model.'}"
            )

    scores = pd.DataFrame(raw_scores)
    st.subheader(f"Raw Metric Errors ({num_predictions} time steps)")
    st.dataframe(scores)
    backtest_df = pd.DataFrame(all_backtests)
    st.subheader(
        f"Backtested Metric Errors ({forecast_horizon} time step forecast horizon)"
    )
    backtest_scores = pd.concat(
        (
            backtest_df.mean(),
            backtest_df.median(),
            backtest_df.min(),
            backtest_df.max(),
        ),
        axis=1,
    )
    backtest_scores.columns = ["mean", "median", "min", "max"]
    st.write(backtest_scores)
    st.subheader("Backtested Errors per Forecasted Time Step")
    backtest_df.index.name = "Forecasted Time Step"
    st.plotly_chart(px.line(backtest_df))
    st.subheader("Total Count of Backtested Error Values over Forecasted Time Steps")
    st.selectbox(
        "Show Histogram for Backtest Metric",
        metric_choices,
        key="backtest_histogram",
        help="Show raw number of each error value per metric.",
    )

    backtest_fig = plt.figure()
    axis = plt.gca()
    plot_hist(
        all_backtests[st.session_state.backtest_histogram],
        bins=np.arange(0, max(all_backtests[st.session_state.backtest_histogram]), 1),
        title="Backtest Histogram",
        ax=axis,
    )
    st.pyplot(backtest_fig)

with st.expander("Matplotlib plot"):
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


forecast_csv = convert_df(prediction_df)
st.download_button(
    label="Download Forecast as CSV",
    data=forecast_csv,
    file_name=f"forecast_{model_choice}_{datetime.now().strftime('%Y_%m_%d')}.csv",
    mime="text/csv",
)
all_csv = convert_df(display_data)
st.download_button(
    label="Download Data and Forecast as CSV",
    data=all_csv,
    file_name=f"all_data_{model_choice}_{datetime.now().strftime('%Y_%m_%d')}.csv",
    mime="text/csv",
)

model = model_cls(*model_args, **model_kwargs)
model.fit(train)
st.download_button(
    label="Download Fitted Model Pickle",
    data=pickle.dumps(model),
    file_name=f"{model_choice}_{datetime.now().strftime('%Y_%m_%d')}.pickle",
    mime="application/octet-stream",
)
