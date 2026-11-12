"""Evaluates given models on test data, writing evaluation metrics in a summary CSV file.
Detailed results on the test data may also be written if desired.

Currently only Buckeye test split data is supported for evaluation.
"""

import argparse
from collections import defaultdict, Counter
from pathlib import Path
from typing import Any

import datasets
import evaluate
import ipatok
import kaldialign
import numpy as np
import pandas as pd
import panphon.distance
import transformers
import torch

from multipa.data_utils import load_buckeye_split, clean_text, EMPTY_TRANSCRIPTION, decode_audio, GOLD_STANDARD_KEY

PHONE_ERRORS_EVALUATOR = evaluate.load("ginic/phone_errors")
DETAILED_PREDICTIONS_CSV_SUFFIX = "detailed_predictions.csv"
HALLUCINATIONS_SUFFIX = "hallucinations.csv"

# Headers for detailed predictions CSV outputs
PREDICTION_KEY = "prediction"
PER_KEY = "phone_error_rates"
PFER_KEY = "phone_feature_error_rates"
FER_KEY = "feature_error_rates"
HALLUCINATIONS_KEY = "panphone_phone_hallucinations"  # Computed by panphone

# Null symbol for kaldi align that shouldn't ever occur in IPA strings
EPS = "***"


def clean_model_name(model_name: str | Path) -> str:
    """Removes path characters from models names or directories.

    Args:
        model_name: str, name of a model, usually from HuggingFace
    """
    return str(model_name).replace("/", "_").replace("\\", "_")


def compute_edit_distance_errors(prediction: str, reference: str, use_ipa_tokenise: bool = True, **kwargs):
    """Tokenizes and compares two strings and returns counts of the substitions, deletions and insertions
    between them.

    Tokenization is done in this function to ensure that the same tokenization strategy is used on both
    prediction and reference. This is important to make sure that differences due to symbol inventories and
    model vocabularies are minimized. To reduce complexity, we also choose to use the ipatok
    library, rather than the wav2vec model tokenizers.

    Results are in the following format:
    - Substitutions: dictionary mapping reference token to dict mapping substituted token to count
    - Deletions: dictionary mapping deleted tokens to count of number of times deleted
    - Insertions: dictionary mapping inserted token to number of times its inserted

    Args:
        prediction: Predicted transcription
        reference: Reference Transcription
        is_use_ipa_tokenise: Set flag to use IPA phone tokenization heuristics rather than character level
            differences. Defaults to True.
        kwargs: any arguments to pass to the ipatok tokenise function

    Returns:
        tuple[Counter[tuple[str, str]], Counter[str], Counter[str], Counter[str]: Substitutions, deletions, insertions,
            true token counts
    """
    if use_ipa_tokenise:
        try:
            pred_tokens = ipatok.tokenise(prediction, **kwargs)
            ref_tokens = ipatok.tokenise(reference, **kwargs)
        except ValueError:
            # Fall back to characters if tokenise fails
            pred_tokens = list(prediction)
            ref_tokens = list(reference)
    else:
        pred_tokens = list(prediction)
        ref_tokens = list(reference)

    true_token_counts = Counter(ref_tokens)
    subs = Counter()
    insertions = Counter()
    deletions = Counter()

    aligned_pairs = kaldialign.align(ref_tokens, pred_tokens, EPS)

    for r, p in aligned_pairs:
        if r == EPS:
            insertions[p] += 1
        elif p == EPS:
            deletions[r] += 1
        elif r != p:
            subs[(r, p)] += 1

    return subs, deletions, insertions, true_token_counts


def collate_edit_distances(edit_dist_counter: list[Counter[Any]]) -> Counter[Any]:
    """
    Collates edit distance errors from a list of counters.

    Args:
        edit_dist_counter (list[Counter[Any]]): List of counters containing edit distance errors.

    Returns:
        Counter[Any]: A single counter with the aggregated counts.
    """
    final_counts = Counter()
    for c in edit_dist_counter:
        final_counts += c
    return final_counts


def calculate_by_token_error_rates(
    token_counts: Counter[str], substitutions: Counter[tuple[str, str]], deletions: Counter[str]
) -> dict[str, float]:
    """Computes error rates for each token based on substitutions and deletions.

    Args:
        token_counts: Counter mapping tokens to their total occurrences in references.
        substitutions: Counter mapping (reference_token, predicted_token) pairs to substitution counts.
        deletions: Counter mapping deleted tokens to deletion counts.

    Returns:
        dict[str, float]: Dictionary mapping each token to its error rate (ratio of errors to total occurrences).
    """
    # Flatten substitutions to make them easier to work with
    subs_by_references = Counter()
    for sub_pair, sub_count in substitutions.items():
        sub_reference = sub_pair[0]
        subs_by_references[sub_reference] += sub_count

    error_rates = {}
    for tok, tok_count in token_counts.items():
        if tok_count > 0:
            tok_errors = subs_by_references[tok] + deletions[tok]
            error_rates[tok] = tok_errors / tok_count

    return error_rates


def get_token_confusion_matrix(
    substitutions: Counter[tuple[str, str]],
    deletions: Counter[str],
    insertions: Counter[str],
    true_token_counts: Counter[str] | dict[str, int],
    default_keys: None | list[str] = None,
    empty_token_symbol=EPS,
):
    """Constructs a confusion matrix DataFrame from edit distance error counts.

    The confusion matrix shows the relationship between reference tokens (true labels) and predicted tokens.
    Perfect predictions appear where reference == predicted.
    Errors appear as follows:
        - substitutions as (ref_token, pred_token)
        - deletions as (ref_token, empty_symbol)
        - insertions as (empty_symbol, pred_token).

    Args:
        substitutions: Counter mapping (reference_token, predicted_token) pairs to substitution counts.
        deletions: Counter mapping deleted reference tokens to deletion counts.
        insertions: Counter mapping inserted tokens to insertion counts.
        true_token_counts: Counter or dict mapping reference tokens to their total occurrence counts.
        default_keys: Optional list of tokens to include in the matrix even if not present in true_token_counts.
            Defaults to None (empty list).
        empty_token_symbol: Symbol to represent empty/null tokens for deletions and insertions.
            Defaults to EPS ("***").

    Returns:
        pd.DataFrame: Confusion matrix with columns ["reference", "predicted", "count"], where each row
            represents a (reference, predicted) pair and its count. Includes both correct predictions
            (diagonal) and errors (off-diagonal).
    """
    # Nested counters which will be turned into dataframe
    if default_keys is None:
        default_keys = []

    desired_keys = set(true_token_counts.keys()) | set(default_keys)
    # start by assuming perfect performance, then subtract results from there
    # {(ref, prediction) -> count}
    conf_matrix_dict = Counter()
    for k in desired_keys:
        conf_matrix_dict[(k, k)] = true_token_counts.get(k, 0)

    # Handle substitutions
    for (ref, pred), count in substitutions.items():
        conf_matrix_dict[(ref, pred)] += count
        conf_matrix_dict[(ref, ref)] -= count

    for ref, count in deletions.items():
        conf_matrix_dict[(ref, empty_token_symbol)] += count
        conf_matrix_dict[(ref, ref)] -= count

    for extra_tok, count in insertions.items():
        conf_matrix_dict[(empty_token_symbol, extra_tok)] += count

    # Convert the combined list to a DataFrame
    confusion_matrix_df = pd.DataFrame(
        [(t[0], t[1], count) for t, count in conf_matrix_dict.items()], columns=["reference", "predicted", "count"]
    )

    return confusion_matrix_df


def compute_error_rate_confidence_intervals_df(
    error_rate_df: pd.DataFrame,
    count_df: pd.DataFrame,
    error_rate_join_key: str,
    count_join_key: str,
    error_rate_col: str,
    count_col: str,
    interval_const: float = 1.96,
    confidence_interval_col: str = "confidence_interval",
):
    """Computes error rates for each vowel with a confidence interval of according to
    https://machinelearningmastery.com/report-classifier-performance-confidence-intervals/.
    The default settings give a confidence interval of 95%.

    Args:
        error_rate_df: DataFrame containing phone error rates.
        count_df: DataFrame containing phone counts for computing confidence intervals.
        error_rate_join_key: Column name in error_rate_df to use for inner join.
        count_join_key: Column name in count_df to use for inner join.
        error_rate_col: Column name in error_rate_df containing the error rate values.
        count_col: Column name in count_df containing the phone count values.
        interval_const: Constant multiplier for confidence interval. Defaults to 1.96 for 95% CI.
        confidence_interval_col: Desired Name for the output confidence interval column.
            Defaults to 'confidence_interval'.

    Returns:
        Joined DataFrame with both error rates and computed confidence intervals.
    """
    joined_df = pd.merge(error_rate_df, count_df, left_on=error_rate_join_key, right_on=count_join_key, how="inner")
    error_series = joined_df[error_rate_col]
    joined_df[confidence_interval_col] = interval_const * np.sqrt((error_series * (1 - error_series)) / joined_df[count_col])
    return joined_df


class ModelEvaluator:
    model_key = "model"
    # Metric names that will become column headers
    per_key = "mean_phone_error_rate"
    pfer_key = "mean_phone_feature_error_rate"
    fer_key = "mean_feature_error_rate"
    phone_hallucinations_key = "total_phone_hallucinations"
    substitutions_key = "substitutions"
    deletions_key = "deletions"
    insertions_key = "insertions"
    by_token_error_rates = "by_token_error_rates"

    def __init__(self, use_ipa_tokenise: bool = True, tokenise_options: None | dict[str, Any] = None):
        """Configure distance calculations and save results for each model

        Args:
            use_ipa_tokenise: Set as True to tokenise IPA phonemes before edit distance calculations or False
                to compute edit distance at the character level. Defaults to True.
        """
        self.distance_computer = panphon.distance.Distance()
        # Final results will have these keys
        # {model name -> {metric_key: metric_value}}
        self.results_to_write = defaultdict(dict)
        self.use_ipa_tokenise = use_ipa_tokenise
        if tokenise_options is None:
            self.tokenise_options = {}
        else:
            self.tokenise_options = tokenise_options

        # Stores the token/phone counts for each model according
        # to the references seen to that point
        self._true_token_counts = defaultdict(Counter)

    def eval_edit_distances(self, model_name, predictions, references, compute_by_token_error_rates=False):
        """Computes edit distance errors and by-token (by-phoneme) error rates for the specified model.
        Model-level results and token counts from the references
        are saved in the current ModelEvaluator instance.

        Example-level results for substitutions, deletions and insertions
        are returned in the form
        {error_type_key -> [{affected_token -> count_of_errors}]}.

        Args:
            model_name: The model currently being evaluated.
            predictions: Predictions from the model
            references: Reference transcriptions
            compute_by_token_error_rates: If True, compute and store by-token error rates. Defaults to False.
            kwargs: Passed to ipatok.tokenise

        Returns:
            dict[str, list[dict[Any, int]]]: example level edit distance errors
        """
        # Compute errors for each example
        subs, deletions, inserts = [], [], []

        true_token_counts = Counter()

        for p, r in zip(predictions, references):
            s, d, i, ref_token_counts = compute_edit_distance_errors(
                p, r, use_ipa_tokenise=self.use_ipa_tokenise, **self.tokenise_options
            )
            subs.append(s)
            deletions.append(d)
            inserts.append(i)
            true_token_counts += ref_token_counts

        # Save totals for each symbol edit for the current model
        total_subs = collate_edit_distances(subs)
        total_deletions = collate_edit_distances(deletions)
        total_inserts = collate_edit_distances(inserts)
        self._true_token_counts[model_name] = self._true_token_counts[model_name] + true_token_counts

        for k, curr_counts in [
            (ModelEvaluator.substitutions_key, total_subs),
            (ModelEvaluator.deletions_key, total_deletions),
            (ModelEvaluator.insertions_key, total_inserts),
        ]:
            if model_name in self.results_to_write and k in self.results_to_write[model_name]:
                prev_counts = self.results_to_write[model_name][k]
                self.results_to_write[model_name][k] = dict(collate_edit_distances([curr_counts, prev_counts]))
            else:
                self.results_to_write[model_name][k] = dict(curr_counts)

        # Only compute by-token error rates when requested (for non-empty transcriptions)
        if compute_by_token_error_rates:
            by_token_error_rates = calculate_by_token_error_rates(
                self._true_token_counts[model_name], total_subs, total_deletions
            )
            self.results_to_write[model_name][ModelEvaluator.by_token_error_rates] = dict(by_token_error_rates)

        # Return example-level edit-distance results
        edit_dist_dict = {
            ModelEvaluator.substitutions_key: subs,
            ModelEvaluator.deletions_key: deletions,
            ModelEvaluator.insertions_key: inserts,
        }
        return edit_dist_dict

    def eval_non_empty_transcriptions(self, model_name, predictions, references, **kwargs) -> dict[str, list[Any]]:
        """Compare the predictions and gold-standard references for a model,
        then add results to model results tracker.
        Returns the full detailed evaluation results as a dictionary.
        """
        metrics = PHONE_ERRORS_EVALUATOR.compute(predictions=predictions, references=references)
        for k in [ModelEvaluator.per_key, ModelEvaluator.pfer_key, ModelEvaluator.fer_key]:
            self.results_to_write[model_name][k] = metrics[k]

        edit_dist_dict = self.eval_edit_distances(
            model_name, predictions, references, compute_by_token_error_rates=True, **kwargs
        )
        metrics.update(edit_dist_dict)
        return metrics

    def eval_empty_transcriptions(self, model_name, predictions):
        """Count number of phone hallucinations for this model and save to write later.
        Returns the detailed evaluation results as a list with one entry per prediction.
        """
        # Insertions according to panphon's feature model
        phone_lengths = [len(self.distance_computer.fm.ipa_segs(p)) for p in predictions]
        total_phone_hallucinations = sum(phone_lengths)
        self.results_to_write[model_name][ModelEvaluator.phone_hallucinations_key] = total_phone_hallucinations

        # Insertions according to edit distance
        edit_dist_dict = self.eval_edit_distances(model_name, predictions, [""] * len(predictions))
        edit_dist_dict[HALLUCINATIONS_KEY] = phone_lengths
        return edit_dist_dict

    def to_csv(self, csv_path: Path | str):
        """Write the aggregate evaluation results stored in this object to the specified CSV file.
        Each model is a row, with aggregate (average or total) metrics stored per column
        """
        df = pd.DataFrame.from_dict(self.results_to_write, orient="index")
        desired_cols = [
            ModelEvaluator.per_key,
            ModelEvaluator.pfer_key,
            ModelEvaluator.fer_key,
            ModelEvaluator.phone_hallucinations_key,
            ModelEvaluator.substitutions_key,
            ModelEvaluator.insertions_key,
            ModelEvaluator.deletions_key,
            ModelEvaluator.by_token_error_rates,
        ]
        df.index.name = ModelEvaluator.model_key
        df.to_csv(csv_path, columns=[c for c in desired_cols if c in df.columns])

    def get_token_confusion_matrix(self, model_name: str) -> pd.DataFrame:
        """Returns the by-token or by-phoneme confusion matrix for the model
        computed by the edit distances on the predictions, references seen thus far.

        Args:
            model_name: str, the model_name or identifier to lookup results for

        Returns:
            pd.DataFrame: Confusion matrix with columns ["reference", "predicted", "count"], where each row
            represents a (reference, predicted) pair and its count. Includes both correct predictions
            (diagonal) and errors (off-diagonal)
        """
        return get_token_confusion_matrix(
            self.results_to_write[model_name][ModelEvaluator.substitutions_key],
            self.results_to_write[model_name][ModelEvaluator.deletions_key],
            self.results_to_write[model_name][ModelEvaluator.insertions_key],
            self._true_token_counts[model_name],
        )

    def write_edit_distance_results(self, model_name: str | Path, directory: Path):
        """Writes counts of edit distance errors and the confusion matrix by symbol to CSV files.
        Each model will have 4 corresponding CSVs, one each for subsitutions, deletions and insertions.

        Args:
            model_name: The name of the model being evaluated
            directory: The desired output directory where CSV files will be written
        """
        csv_base_name = clean_model_name(model_name)
        for k in [ModelEvaluator.deletions_key, ModelEvaluator.insertions_key]:
            count_col = f"total_{k}"
            edit_dist_df = pd.DataFrame.from_records(
                list(self.results_to_write[model_name][k].items()), columns=["symbol", count_col]
            )
            edit_dist_df.sort_values(by=count_col, ascending=False, inplace=True)
            edit_dist_df.to_csv(directory / f"{csv_base_name}_{k}.csv", index=False)

        subs_col = f"total_{ModelEvaluator.substitutions_key}"
        substitutions_df = pd.DataFrame.from_records(
            [(s[0], s[1], v) for s, v in self.results_to_write[model_name][ModelEvaluator.substitutions_key].items()],
            columns=["original", "substitution", subs_col],
        )
        substitutions_df.sort_values(by=subs_col, inplace=True, ascending=False)
        substitutions_df.to_csv(directory / f"{csv_base_name}_{ModelEvaluator.substitutions_key}.csv", index=False)

        conf_matrix = self.get_token_confusion_matrix(model_name)
        conf_matrix.to_csv(directory / f"{csv_base_name}_confusion_matrix.csv", index=False)


def preprocess_test_data(
    test_dataset: datasets.Dataset, is_remove_space: bool = False, is_normalize_ipa=False, num_proc: int | None = None
):
    """
    Filters the test dataset into examples with non-empty and empty transcriptions,
    since they should be evaluated separately.
    Also performs additional text cleaning if specified.

    Args:
        test_dataset: Huggingface dataset you'll use for evaluation
        is_remove_space: Filter out spaces in IPA strings if true
        is_normalize_ipa: Set to True to make common IPA-compliant substitutions
        num_proc: The number of processes to use for multiprocessing. If None, no multiprocessing is used.

    Returns:
        non_empty_transcriptions_dataset, empty_transcriptions_dataset: a tuple of Huggingface datasets
    """
    # Set sampling rate to 16K
    input_data = test_dataset.cast_column("audio", datasets.Audio(sampling_rate=16_000)).map(
        lambda x: clean_text(x, is_remove_space=is_remove_space, is_normalize_ipa=is_normalize_ipa), num_proc=num_proc
    )

    empty_test_data = input_data.filter(lambda x: x[GOLD_STANDARD_KEY] == EMPTY_TRANSCRIPTION, num_proc=num_proc)
    non_empty_test_data = input_data.filter(lambda x: x[GOLD_STANDARD_KEY] != EMPTY_TRANSCRIPTION, num_proc=num_proc)

    return non_empty_test_data, empty_test_data


def get_torch_device(use_gpu: bool = False):
    """Return the torch.device to load the model to.

    Args:
        use_gpu (bool, optional): True if the user desires GPU for model inference. Defaults to False.
    """
    if use_gpu and torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def drop_extra_csv_output_columns(dataset: datasets.Dataset, extra_columns: list[str] | None = None) -> datasets.Dataset:
    """Removes the specified columns from dataset in place.

    Args:
        dataset: HuggingFace dataset to drop columns from
        extra_columns: Column names to drop. Defaults to ["audio", "__index_level_0__"] if not specified

    Returns:
        dataset with columns removed
    """
    if extra_columns is None:
        extra_columns = ["audio", "__index_level_0__"]
    for c in extra_columns:
        if c in dataset.column_names:
            dataset = dataset.remove_columns(c)

    return dataset


def write_detailed_hallucination_results(
    verbose_results_dir: Path,
    clean_model_id: str,
    empty_test_data_predictions: datasets.Dataset,
    hallucinations_with_edit_dist: dict[str, list[Any]],
):
    hallucinations_csv = verbose_results_dir / (f"{clean_model_id}_{HALLUCINATIONS_SUFFIX}")

    empty_test_to_write = empty_test_data_predictions.add_column(
        HALLUCINATIONS_KEY, hallucinations_with_edit_dist[ModelEvaluator.phone_hallucinations_key]
    )
    empty_test_to_write = empty_test_to_write.add_column(
        ModelEvaluator.insertions_key,
        [str(dict(s)) for s in hallucinations_with_edit_dist[ModelEvaluator.insertions_key]],
    )
    empty_test_to_write = drop_extra_csv_output_columns(empty_test_to_write)
    empty_test_to_write.to_csv(hallucinations_csv, index=False)


def write_detailed_prediction_results(
    verbose_results_dir: Path,
    clean_model_id: str,
    non_empty_test_data: datasets.Dataset,
    predictions: datasets.Dataset,
    detailed_metrics: dict[str, list[Any]],
):
    detailed_results_csv = verbose_results_dir / (f"{clean_model_id}_{DETAILED_PREDICTIONS_CSV_SUFFIX}")

    detailed_results = non_empty_test_data.add_column(PREDICTION_KEY, predictions[PREDICTION_KEY])
    for k in [PER_KEY, PFER_KEY, FER_KEY]:
        detailed_results = detailed_results.add_column(k, detailed_metrics[k])

    # These are stored as counter objects and need to be converted to dict strings before adding columns
    # This is a bit hacky, but it the output should just be for manual inspection
    for k in [ModelEvaluator.deletions_key, ModelEvaluator.insertions_key, ModelEvaluator.substitutions_key]:
        detailed_results = detailed_results.add_column(k, [str(dict(d)) for d in detailed_metrics[k]])

    detailed_results = drop_extra_csv_output_columns(detailed_results)
    detailed_results.to_csv(detailed_results_csv, index=False)


def get_clean_predictions(
    audio_dataset: datasets.Dataset,
    transformer_pipe: transformers.Pipeline,
    num_proc: int | None = None,
    audio_key: str = "audio",
    text_key: str = "text",
    is_remove_space: bool = True,
    is_normalize_ipa: bool = False,
):
    """Predicts transcriptions for the audio dataset using the transform pipeline, then
    returns the original Dataset with the clean transcription text in the "prediction" column

    Args:
        audio_dataset: HuggingFace format dataset with "audio" column
        transformer_pipe: HuggingFace speech recognition pipeline
        num_proc: Number of processors for map functions. Defaults to None.
        audio_key: Column storing audio in dataset. Defaults to "audio".
        text_key: Column where transcriptions are put by the pipeline object. Defaults to "text".
        is_remove_space: Whether or not to remove spaces from transcriptions. Defaults to True.
        is_normalize_ipa: Set to true to make common substitutions to IPA-compliant symbols using ipatok.tokenise.
            Defaults to False.

    Returns:
        datasets.Dataset with clean transcription text in "prediction"
    """
    # Extract audio inputs using row-level access (avoids Column object issue)
    # and decode AudioDecoder objects if present
    audio_inputs = []
    # Track failed audios, mapping index to data to keep
    failed_audios = {}
    for i in range(len(audio_dataset)):
        audio = audio_dataset[i][audio_key]
        try:
            audio_inputs.append(decode_audio(audio))
        except RuntimeError as e:
            # Catches TorchCodec RuntimeError (getFramesPlayedInRangeAudio)
            # and any other decoding failures (corrupted files, empty audio, etc.)
            sample = audio_dataset[i]
            print("Returning empty prediction for sample", i, "due to RuntimeError. Error:", e, " Sample:", sample)
            sample[PREDICTION_KEY] = ""
            failed_audios[i] = sample

    # Run pipeline directly — no .map(), no pickling needed
    predictions_dataset = datasets.Dataset.from_list(transformer_pipe(audio_inputs))
    predictions_dataset = predictions_dataset.map(
        lambda x: clean_text(x, text_key=text_key, is_remove_space=is_remove_space, is_normalize_ipa=is_normalize_ipa),
        num_proc=num_proc,
    )
    predictions_dataset = predictions_dataset.rename_column(text_key, PREDICTION_KEY)
    print("Predictions successfully obtained for", len(predictions_dataset), "samples.")
    print("Example of prediction:", predictions_dataset[0])
    print("Failures in predicting", len(failed_audios), "audio files.")

    # Insert failed predictions back to final list
    prediction_list = predictions_dataset[PREDICTION_KEY]
    full_predictions = []
    success_idx = 0
    for i in range(len(audio_dataset)):
        if i in failed_audios:
            full_predictions.append("")
        else:
            full_predictions.append(prediction_list[success_idx])
            success_idx += 1

    return audio_dataset.add_column(PREDICTION_KEY, full_predictions)


def main(
    input_data: datasets.Dataset,
    eval_csv: Path,
    local_models: list[Path] | None = None,
    hf_models: list[str] | None = None,
    verbose_results_dir: Path | None = None,
    is_remove_space: bool = False,
    is_normalize_ipa: bool = False,
    use_gpu: bool = False,
    num_proc: int | None = None,
    edit_dist_dir: Path | None = None,
):
    if local_models is None:
        local_models = []
    if hf_models is None:
        hf_models = []

    print("Loading test data")
    non_empty_test_data, empty_test_data = preprocess_test_data(input_data, is_remove_space, num_proc)
    print("Test dataset with NON-empty transcriptions:", non_empty_test_data)

    print("Test dataset with empty transcriptions:", empty_test_data)

    model_eval_tracker = ModelEvaluator()

    selected_torch_device = get_torch_device(use_gpu)

    if verbose_results_dir:
        verbose_results_dir.mkdir(parents=True, exist_ok=True)
        print("Detailed transcription predictions and metrics will be written to", verbose_results_dir)

    if edit_dist_dir:
        edit_dist_dir.mkdir(parents=True, exist_ok=True)
        print("Edit distance results will be written to", edit_dist_dir)

    for model in local_models + hf_models:
        print("Evaluating model:", model)
        pipe = transformers.pipeline("automatic-speech-recognition", model=model, device=selected_torch_device)

        clean_model_id = clean_model_name(model)

        if len(non_empty_test_data) > 0:
            print("Getting predictions for audio with non-empty gold-standard transcriptions")
            predictions = get_clean_predictions(
                non_empty_test_data,
                pipe,
                num_proc=num_proc,
                is_remove_space=is_remove_space,
                is_normalize_ipa=is_normalize_ipa,
            )
            print("Predictions data preview:")
            print(predictions[0])

            print("Computing performance metrics for non-empty audio transcriptions")
            metrics = model_eval_tracker.eval_non_empty_transcriptions(
                model, predictions[PREDICTION_KEY], non_empty_test_data[GOLD_STANDARD_KEY]
            )

            if verbose_results_dir:
                print("Writing detailed predictions metrics for", model, "to", verbose_results_dir)
                write_detailed_prediction_results(
                    verbose_results_dir, clean_model_id, non_empty_test_data, predictions, metrics
                )

        if len(empty_test_data) > 0:
            print("Getting predictions for audio with empty gold-standard transcriptions")
            empty_test_data_predictions = get_clean_predictions(
                empty_test_data, pipe, num_proc=num_proc, is_remove_space=is_remove_space, is_normalize_ipa=is_normalize_ipa
            )
            print("Predictions for hallucinations data preview:")
            print(empty_test_data_predictions[0])

            hallucinations_with_edit_dist = model_eval_tracker.eval_empty_transcriptions(
                model, empty_test_data_predictions[PREDICTION_KEY]
            )

            if verbose_results_dir:
                print("Writing detailed hallucination results for", model, "to", verbose_results_dir)
                write_detailed_hallucination_results(
                    verbose_results_dir, clean_model_id, empty_test_data_predictions, hallucinations_with_edit_dist
                )

        # Write by-model edit distance errors with counts
        if edit_dist_dir:
            model_eval_tracker.write_edit_distance_results(model, edit_dist_dir)

    # Write final aggregate metrics results for all models
    eval_csv.parent.mkdir(exist_ok=True, parents=True)
    model_eval_tracker.to_csv(eval_csv)


def main_cli():
    parser = argparse.ArgumentParser(description="Evaluate audio to IPA transcriptions on a held out test corpus.")
    parser.add_argument(
        "-l", "--local_models", type=Path, nargs="*", help="List of paths to model files saved locally to evaluate."
    )

    parser.add_argument(
        "-f",
        "--hf_models",
        type=str,
        nargs="*",
        help="List of names of models stored on Hugging Face to download and evaluate.",
    )

    parser.add_argument(
        "-d",
        "--data_dir",
        type=Path,
        required=True,
        help="Path local Buckeye format pre-processed dataset to use for testing.",
    )

    parser.add_argument("-e", "--eval_out", type=Path, required=True, help="Output CSV files to write evaluation metrics to.")
    parser.add_argument(
        "-v",
        "--verbose_results_dir",
        type=Path,
        help=(
            "If desired, specify a path to a directory to dump by-model transcriptions with by-example "
            "performance metrics to for later inspection if desired. For each model, one CSV for detailed "
            "predictions and one for hallucinations (audio without speech, but where transcripts) are created."
        ),
    )

    parser.add_argument(
        "-ed",
        "--edit_dist_dir",
        type=Path,
        help=(
            "If desired, specify a path to a directory for storing by-model edit distance results, "
            "3 CSVs for each model: one for substitions, insertions and deletions"
        ),
    )

    parser.add_argument("-ns", "--no_space", action="store_true", help="Use this flag to remove spaces in IPA transcription.")

    parser.add_argument(
        "-ni", "--normalize_ipa", action="store_true", help="Use this flag to normalize common IPA Unicode symbols."
    )

    parser.add_argument(
        "-g",
        "--use_gpu",
        action="store_true",
        help="Use a GPU for inference if available. Otherwise the model runs on CPUs.",
    )

    parser.add_argument(
        "--num_proc",
        type=int,
        help="Specify the number of CPUs for preprocessing. If unset, no multiprocessing is used.",
    )

    args = parser.parse_args()

    buckeye_test = load_buckeye_split(args.data_dir, "test")
    main(
        buckeye_test,
        args.eval_out,
        args.local_models,
        args.hf_models,
        args.verbose_results_dir,
        args.no_space,
        args.normalize_ipa,
        args.use_gpu,
        args.num_proc,
        args.edit_dist_dir,
    )


if __name__ == "__main__":
    main_cli()
