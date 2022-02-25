"""This script analyzes the signal of a sensor device"""
import itertools
import os
from pathlib import Path
from typing import List, Union, Tuple

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks
from tqdm.contrib import tqdm

DATA_PATH = "DATA.bin"


def plot_signal(
    df: pd.DataFrame,
    columns: List[str],
    filename: Union[str, Path],
    heartbeat: bool = True,
) -> None:
    """plot the columns of a dataframe
    :param df: dataframe
    :param columns: columns to plot
    :param filename: filename to save the plot
    :param heartbeat: if True, plot the heart_beat column as points
    """
    plt.figure().set_size_inches(14, 9)
    for col in columns:
        plt.plot(df[col])
    if heartbeat:
        heart_beat = df.heart_beat[df.heart_beat == 1]
        plt.plot(heart_beat.index, [0] * len(heart_beat.index), "bo")
        columns = columns + ["HeartActivityLabel"]
    plt.legend(columns)
    plt.savefig(filename, bbox_inches="tight")
    plt.close()


def calculate_inter_pearson_correlation(segments: List[pd.DataFrame]) -> pd.DataFrame:
    """This method calculates the correlation between the segments for each column.
    :param segments: list of segments
    :return: dataframe with the inter-correlation
    """
    multi_index = [
        (k, s)
        for k, s in itertools.product(range(len(segments)), range(len(segments)))
        if k < s
    ]
    feature_df = pd.DataFrame(
        index=multi_index,
        columns=["SideLeft", "SideRight", "FrontLeft", "FrontRight", "Back", "Top"],
        dtype=np.float32,
    )
    for k, k_segment in enumerate(segments):
        for s, s_segment in enumerate(segments):
            if k < s:
                for i, col in enumerate(["SideLeft", "SideRight", "FrontLeft", "FrontRight", "Back", "Top"]):
                    feature_df.xs((k, s))[f"{col}"] = k_segment[col].corr(
                        s_segment[col]
                    )
    return feature_df


def calculate_intra_pearson_correlation(segments: List[pd.DataFrame]) -> pd.DataFrame:
    """This method calculates the correlation between the columns.
    :param segments: list of segments
    :return: dataframe with the correlation between
    """
    columns = [
        f"{icol}-{jcol}"
        for (i, icol), (j, jcol) in (
            itertools.product(
                enumerate(["SideLeft", "SideRight", "FrontLeft", "FrontRight", "Back", "Top"]),
                enumerate(["SideLeft", "SideRight", "FrontLeft", "FrontRight", "Back", "Top"]),
            )
        )
        if i < j
    ]
    feature_df = pd.DataFrame(columns=columns, dtype=np.float32)
    for k, segment in enumerate(segments):
        for i, icol in enumerate(["SideLeft", "SideRight", "FrontLeft", "FrontRight", "Back", "Top"]):
            for j, jcol in enumerate(["SideLeft", "SideRight", "FrontLeft", "FrontRight", "Back", "Top"]):
                if i < j:
                    feature_df.loc[k, f"{icol}-{jcol}"] = segment[icol].corr(
                        segment[jcol]
                    )
    return feature_df


def align_segments_on_column(
    sg_segments: List[pd.DataFrame], align_column: str = "Top", align_ix: int = 0
) -> List[pd.DataFrame]:
    """align segments based on max correlation with the segment with index align_ix. The order is preserved
    :param sg_segments: list of segments
    :param align_column: column to align on
    :param align_ix: index of the segment to align on
    :return: list of aligned segments
    """
    align_segment = sg_segments[align_ix]
    new_segments = []
    for k, segment in tqdm(enumerate(sg_segments), total=len(sg_segments)):
        if k != align_ix:
            cross_corr = [
                align_segment[align_column].corr(segment[align_column].shift(i))
                for i in range(-250, 250)
            ]
            shift = np.argmax(cross_corr) - 250
            new_segments.append(segment.shift(shift))
        else:
            new_segments.append(segment)
    return new_segments


def smoother_segments(
    segments: List[pd.DataFrame], rolling_window: int = 20
) -> List[pd.DataFrame]:
    """smooth segments based on rolling window. The columns 'HeartActivityLabel' and 'heart_beat' are removed.
    :param segments: list of segments
    :param rolling_window: rolling window
    :return: list of smoothed segments
    """
    new_segments = []
    for segment in tqdm(segments, total=len(segments)):
        new_segment = segment.rolling(rolling_window).mean()
        new_segment = new_segment.drop(columns=["HeartActivityLabel", "heart_beat"])
        new_segment.dropna(inplace=True)
        new_segment.reset_index(drop=True, inplace=True)
        new_segments.append(new_segment)
    return new_segments


def align_segments(
    segments: List[pd.DataFrame], origin_df: pd.DataFrame, picture_path: Path
) -> Tuple[List[pd.DataFrame], str]:
    """correlation analysis
    This method analyzes the correlation between the segments based on different columns.
    For each column, the segments are aligned based on the maximal correlation with the first segment.
    To measure the success of the alignment, the correlation between the aligned segments is calculated
    for each column and aggregated as mean.
    Then, the mean of all aggregated means of each column is calculated.
    Based on this value, we pick the column which is most promising to align the segments.

    Also, it generates pictures of the correlation.
    :param segments: list of segments
    :param origin_df: original dataframe
    :param picture_path: path to save pictures
    :return: list of aligned segments and name of the most promising column
    """
    print("aligning segments")
    print("#################################################################")
    if not isinstance(picture_path, Path):
        picture_path = Path(picture_path)
    os.makedirs(picture_path, exist_ok=True)
    # intra pearson correlation
    print("intra pearson correlation")
    intra_corr_df = calculate_intra_pearson_correlation(segments)
    print(intra_corr_df)
    print(intra_corr_df.describe())
    print("intra hole segment")
    intra_corr_hole_segment_df = calculate_intra_pearson_correlation([origin_df])
    print(intra_corr_hole_segment_df)

    # inter pearson correlation
    print("inter pearson correlation")
    inter_corr_df = calculate_inter_pearson_correlation(segments)
    print(inter_corr_df)
    print(inter_corr_df.describe())
    print("show the ones which are less correlated")
    print(inter_corr_df.Top.abs().sort_values().head(20))

    plt.figure().set_size_inches(14, 9)
    plt.plot(segments[2].Top)
    plt.plot(segments[6].Top)
    plt.legend(["segment 2", "segment 6"])
    plt.savefig(picture_path / "correlation_segments_Top_2_6.jpg", bbox_inches="tight")
    plt.close()

    plt.figure().set_size_inches(14, 9)
    plt.plot(segments[2].Top)
    plt.plot(segments[1].Top)
    plt.legend(["segment 2", "segment 1"])
    plt.savefig(picture_path / "correlation_segments_Top_1_2.jpg", bbox_inches="tight")
    plt.close()

    plt.figure().set_size_inches(14, 9)
    cross_corr = [
        segments[2].Top.corr(segments[6].Top.shift(i)) for i in range(-250, 250)
    ]
    shift = np.argmax(cross_corr) - 250
    plt.plot(segments[2].Top)
    plt.plot(segments[6].Top)
    plt.plot(segments[6].Top.shift(shift))
    plt.legend(["segment 2", "segment 6", "shifted segment 6"])
    plt.savefig(
        picture_path / "correlation_segments_Top_2_6_shifted.jpg", bbox_inches="tight"
    )
    plt.close()

    new_segments_list = align_segments_on_column(segments, "Top")
    plt.figure().set_size_inches(14, 9)
    plt.plot(new_segments_list[2].Top)
    plt.plot(new_segments_list[6].Top)
    plt.title("group shifted by Top")
    plt.legend(["segment 2", "segment 6"])
    plt.savefig(
        picture_path / "correlation_segments_Top_2_6_group_shifted.jpg",
        bbox_inches="tight",
    )
    plt.close()

    os.makedirs(picture_path / "alignments", exist_ok=True)

    # plot all segments in one picture for each column
    for col in ["SideLeft", "SideRight", "FrontLeft", "FrontRight", "Back", "Top"]:
        plt.figure().set_size_inches(14, 9)
        for segment in segments:
            plt.plot(segment[col])
        plt.savefig(
            picture_path / f"alignments/unaligned_segments_{col}.jpg",
            bbox_inches="tight",
        )
        plt.close()

    means = {"origin": inter_corr_df.describe().loc["mean"].mean()}
    std = {"origin": inter_corr_df.describe().loc["std"].mean()}

    for col in ["SideLeft", "SideRight", "FrontLeft", "FrontRight", "Back", "Top"]:

        new_segments_list = align_segments_on_column(segments, col)
        inter_correlation_df = calculate_inter_pearson_correlation(new_segments_list)

        print(f"------------------{col}------------------")
        descr = inter_correlation_df.describe()
        print(descr)

        means[col] = descr.loc["mean"].mean()
        std[col] = descr.loc["std"].mean()

        # plot all aligned segments in one plot for each column
        for col2 in ["SideLeft", "SideRight", "FrontLeft", "FrontRight", "Back", "Top"]:
            plt.figure().set_size_inches(14, 9)
            for segment in new_segments_list:
                plt.plot(segment[col2])
            plt.savefig(
                picture_path / f"alignments/aligned_segments_{col}_{col2}.jpg",
                bbox_inches="tight",
            )
            plt.close()

    print("mean of mean of the correlation between segments for each column")
    print(means)
    print("mean of std of the correlation between segments for each column")
    print(std)

    max_column = max(means, key=means.get)
    if max_column == "origin":
        return segments, max_column
    new_segments_list = align_segments_on_column(segments, max_column)

    return new_segments_list, max_column


def analyze_heart_beat(df: pd.DataFrame, root_picture_path: Path) -> pd.DataFrame:
    """This method will analyze the heart beats in the signal, generate plots and correct the missing labels.
    It will return a copy of the dataframe with the heart beats corrected.

    :param df: pandas dataFrame. Must contain the columns "heart_beat" and "HeartActivity"
    :param root_picture_path: the path to the root picture path
    :return: a copy of the dataframe with the heart beats corrected
    """
    new_df = df.copy()
    heart_beat = df.heart_beat[df.heart_beat == 1]

    plt.figure().set_size_inches(14, 9)
    plt.plot(heart_beat.index, df.HeartActivity[heart_beat.index], "bo")
    plt.plot(df.HeartActivity)
    plt.legend(["HeartActivityLabel", "HeartActivity"])
    plt.savefig(root_picture_path / "heart_beat.jpg", bbox_inches="tight")
    plt.close()

    # heart beat close up
    plt.figure().set_size_inches(14, 9)
    plt.plot(
        heart_beat.loc[250000:300000].index,
        df.HeartActivity[heart_beat.index].loc[250000:300000],
        "bo",
    )
    plt.plot(df.HeartActivity.loc[250000:300000])
    plt.legend(["HeartActivityLabel", "HeartActivity"])
    plt.savefig(root_picture_path / "heart_beat_close_up.jpg", bbox_inches="tight")
    plt.close()

    # missing meta points
    plt.figure().set_size_inches(14, 9)
    plt.plot(
        heart_beat.loc[100000:135000].index,
        df.HeartActivity[heart_beat.index].loc[100000:135000],
        "bo",
    )
    plt.plot(df.HeartActivity.loc[100000:135000])
    plt.legend(["HeartActivityLabel", "HeartActivity"])
    plt.savefig(
        root_picture_path / "heart_beats_close_up_without_labels.jpg",
        bbox_inches="tight",
    )
    plt.close()

    # correct missing heart beats labels
    peaks, _ = find_peaks(df.HeartActivity.loc[100000:135000], prominence=0.03)
    plt.figure().set_size_inches(14, 9)
    plt.plot(
        heart_beat.loc[100000:135000].index,
        df.HeartActivity[heart_beat.index].loc[100000:135000],
        "bo",
    )
    plt.plot(df.HeartActivity.loc[100000:135000])
    plt.plot(peaks + 100000, df.HeartActivity.loc[100000:135000][peaks + 100000], "rx")
    plt.legend(["HeartActivityLabel", "HeartActivity", "corrected HeartActivityLabel"])
    plt.savefig(
        root_picture_path / "heart_beats_close_up_with_corrected_labels.jpg",
        bbox_inches="tight",
    )
    plt.close()

    peaks, _ = find_peaks(df.HeartActivity, prominence=0.03)
    missing_peaks = np.array(
        [
            peak
            for peak in peaks
            if not (df.heart_beat[peak - 500: peak + 500] != 0).any()
        ]
    )
    new_df.loc[missing_peaks, "heart_beat"] = 1
    heart_beat = new_df.heart_beat[new_df.heart_beat == 1]

    # heart beat
    plt.figure().set_size_inches(14, 9)
    plt.plot(heart_beat.index, new_df.HeartActivity[heart_beat.index], "bo")
    plt.plot(new_df.HeartActivity)
    plt.legend(["HeartActivityLabel", "HeartActivity"])
    plt.savefig(
        root_picture_path / "heart_beats_with_corrected_labels.jpg", bbox_inches="tight"
    )
    plt.close()

    return new_df


def find_peak(
    stream: pd.Series,
    index: int,
    prominence: float = 0.03,
    distance: int = 200,
    up: bool = True,
) -> int:
    """This method will find the peak in the stream, if it exists.
    :param stream: to be searched
    :param index: index of the heart beat to search around
    :param prominence: prominence of the peak
    :param distance: maximal distance between the heart beat and the peak
    :param up: if True, peaks pointing up will be found, otherwise down
    """
    if not up:
        stream = -stream
    peaks, properties = find_peaks(
        stream.loc[index - distance: index + distance], prominence=prominence
    )
    peak = peaks[np.argmax(properties["prominences"])]
    return peak + index - distance


def align_columns(df: pd.DataFrame, root_picture_path: Path) -> pd.DataFrame:
    """This method search for the Top peak in the segment 192000:194000 close to the heartbeat
    and will shift the data stream Top accordingly.
    It will return a copy of the data frame with the Top shifted.
    Also, it generates plots of segments of the data stream and the shifted data stream.
    :param df: the original data frame. Must contain the columns "SideLeft", "SideRight", "Top", "FrontLeft", "FrontRight", "Back", "heart_beat"
    :param root_picture_path: the path to the root picture path
    :return: a copy of the data frame with the Top column shifted
    """
    heart_beat = df.heart_beat[df.heart_beat == 1]

    peak_cor = find_peak(
        df.Top, heart_beat.loc[192000:194000].index[0], distance=100, prominence=0.0001
    )
    cor_shift = heart_beat.loc[192000:194000].index[0] - peak_cor
    shifted_df = df.copy()
    shifted_df.Top = shifted_df.Top.shift(cor_shift)

    plt.figure().set_size_inches(14, 9)

    plt.plot(df.SideLeft[192000:194000])
    plt.plot(df.SideRight[192000:194000])
    plt.plot(df.Top[192000:194000])
    plt.plot(df.FrontLeft[192000:194000])
    plt.plot(df.FrontRight[192000:194000])
    plt.plot(df.Back[192000:194000])
    plt.plot(
        heart_beat.loc[192000:194000].index,
        [0] * len(heart_beat.loc[192000:194000].index),
        "bo",
    )
    plt.plot(peak_cor, df.Top[peak_cor], "rx")
    plt.legend(["SideLeft", "SideRight", "Top", "FrontLeft", "FrontRight", "Back", "HeartActivityLabel", "Top_peak"])
    plt.savefig(
        root_picture_path / f"segment_{192000}_{194000}_cor_peak.jpg",
        bbox_inches="tight",
    )
    plt.close()

    for i, k in [
        (192000, 194000),
        (195000, 197000),
        (197500, 199500),
        (206500, 208500),
    ]:

        plot_signal(
            df[i:k],
            ["SideLeft", "SideRight", "Top", "FrontLeft", "FrontRight", "Back"],
            root_picture_path / f"segment_{i}_{k}.jpg",
            heartbeat=True,
        )

        plot_signal(
            shifted_df[i:k],
            ["SideLeft", "SideRight", "Top", "FrontLeft", "FrontRight", "Back"],
            root_picture_path / f"segment_{i}_{k}_cor_shifted.jpg",
            heartbeat=True,
        )

    return shifted_df


def segment_signal(df: pd.DataFrame, overlap: int = 500) -> List[pd.DataFrame]:
    """This method will segment the data frame into smaller ones.
    :param df: the data frame to segment
    :param overlap: the overlap between two segments
    :return: a list of data frames
    """
    heart_beat = df.heart_beat[df.heart_beat == 1]
    segment_indices = heart_beat.index.to_numpy()
    segments = []
    for i in range(len(segment_indices) - 1):
        new_segment = df.loc[
            segment_indices[i] - overlap: segment_indices[i + 1] + overlap - 1
        ]
        new_segment.reset_index(drop=True, inplace=True)
        segments.append(new_segment)
    return segments


def load_data(data_path: str) -> pd.DataFrame:
    """This method loads the data from the data path and returns a data frame.
    :param data_path: the path to the data
    :return: the data frame
    """
    dt = np.dtype(
        [
            ("SideLeft", np.float32),
            ("FrontLeft", np.float32),
            ("SideRight", np.float32),
            ("FrontRight", np.float32),
            ("Back", np.float32),
            ("Top", np.float32),
            ("SoundPressureLevel", np.float32),
            ("HeartActivity", np.float32),
            ("HeartActivityLabel", np.float32),
        ]
    )
    data = np.fromfile(data_path, dtype=dt)
    data_df = pd.DataFrame(data)

    return data_df


if __name__ == "__main__":
    # Adjusting the default pandas settings
    pd.set_option("max_columns", 15)
    pd.set_option("expand_frame_repr", False)
    pd.set_option("display.precision", 9)

    root_picture_path = Path("pictures")
    os.makedirs(root_picture_path, exist_ok=True)

    data_df = load_data(DATA_PATH)

    print(data_df.head(10))
    print(data_df.describe())

    data_df["heart_beat"] = data_df.HeartActivityLabel.apply(lambda x: int(bool(x)))

    data_df = analyze_heart_beat(data_df, root_picture_path)

    data_aligned_df = align_columns(data_df, root_picture_path)

    plot_signal(data_aligned_df, ["SideLeft", "SideRight", "Top"], root_picture_path / "hole_data.jpg")
    plot_signal(data_aligned_df, ["SoundPressureLevel"], root_picture_path / "SoundPressureLevel.jpg")

    plot_signal(
        data_aligned_df.loc[190000:250000],
        ["SideLeft", "SideRight"],
        root_picture_path / "segment_190000_250000.jpg"
    )

    short_df = data_aligned_df.loc[190000:250000]
    short_df.reset_index(drop=True, inplace=True)

    short_origin_df = data_df.loc[190000:250000]
    short_origin_df.reset_index(drop=True, inplace=True)

    data_segments = segment_signal(short_df, overlap=0)
    data_origin_segments = segment_signal(short_origin_df, overlap=0)

    os.makedirs(root_picture_path / "segments", exist_ok=True)
    for k, segment in enumerate(data_segments):
        plot_signal(segment, ["SideLeft", "SideRight"], root_picture_path / "segments" / f"segment_{k}.jpg")
    os.makedirs(root_picture_path / "segments_allcol", exist_ok=True)
    for k, segment in enumerate(data_segments):
        plot_signal(segment, ["SideLeft", "SideRight", "Top", "FrontLeft", "SideRight", "Back"], root_picture_path / "segments_allcol" / f"segment_{k}.jpg")

    print("cor not shifted")
    aligned_origin_segments, _ = align_segments(data_origin_segments, short_origin_df, root_picture_path / "cor_not_shifted")

    print("cor shifted")
    aligned_segments, _ = align_segments(
        data_segments, short_df, root_picture_path / "correlation_analyzes"
    )

    # smooth the segments and apply the method align_segments. Compare the results with the not smoothed segments
    # This is only for analyzes and possibly alignment of the segments not for further processing.
    smooth_segments = smoother_segments(data_segments, rolling_window=20)
    aligned_smooth_segments, _ = align_segments(
        smooth_segments, short_df, root_picture_path / "correlation_smooth_analyzes"
    )
