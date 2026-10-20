"""
tidy_repo.py  --  move the OLD test files out of the way. NOTHING IS DELETED: files go into a folder called  _old/

Run from the project folder (telco-absa):
    python3 preprocessing/tidy_repo.py          <- only SHOWS what it would move (dry run)
    python3 preprocessing/tidy_repo.py --go     <- really moves the files into _old/

To undo: drag a file back from _old/ to where it was (same sub-folder: _old/data/processed/... -> data/processed/...).
After you are happy (after the meeting), delete the _old folder yourself.

KEEP list = what the current pipeline and the deck need. The script refuses to move anything on the KEEP list.
"""
import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
P = "preprocessing/"
D = "data/processed/"

KEEP = [P + f for f in [
    "3_normalize_reviews.py", "4_dictionary_map_reviews.py", "detect_aspects.py", "segment_reviews.py",
    "manglish_normalizer.py", "dictionary_mapper.py", "text_versions.py",
    "label_parts_xlmr.py", "label_parts_distilbert.py", "run_part_experiment.py", "make_blind_sheet.py",
    "run_labeler_check.py", "run_manual_svm.py", "run_full_tables.py", "run_same420.py",
    "run_deberta_check.py", "run_mbert_check.py", "run_nanot5_check.py", "run_baselines.py",
    "suggest_aspect_keywords.py", "tidy_repo.py"]] + [D + f for f in [
    "3_normalized_reviews.csv", "4_dictionary_mapped_reviews.csv", "aspect_keywords.csv", "aspect_keyword_suggestions.csv",
    "manglish_dictionary.csv", "manglish_dictionary_BACKUP.csv", "malay_slang.csv", "normalizer_short_forms.csv",
    "dictionary_telco_only.csv", "review_segments_raw.csv", "part_xlmr_labels.csv", "part_distil_labels.csv",
    "part_versions.csv", "part_split.csv", "blind_label_sheet.csv", "blind_key.csv",
    "labeler_check.csv", "labeler_check.xlsx", "deberta_check.csv", "deberta_check.xlsx", "mbert_check.csv", "mbert_check.xlsx",
    "nanot5_check.csv", "nanot5_check.xlsx", "manual_svm.csv", "manual_svm.xlsx", "full_tables.csv", "full_tables.xlsx",
    "same420.csv", "same420.xlsx", "baseline_results.csv"]]

MOVE_SCRIPTS = [P + f for f in [
    "5_translate_reviews.py", "audit_normalizer_changes.py", "build_label_versions.py", "build_samples.py",
    "build_segment_versions.py", "build_train_versions.py", "check_app_keyword.py", "check_dictionary_survival.py",
    "check_slang_candidates.py", "compare_dictionaries.py", "compare_labellers.py", "compare_segment_check.py",
    "count_intensity_words.py", "count_v4_questions.py", "diagnose_broken.py", "diagnose_dictionary.py", "diagnose_vader.py",
    "explore_manglish_patterns.py", "find_idiom_phrases.py", "find_negative_words.py", "find_standalone_slang.py",
    "find_telco_terms.py", "make_label_sheet.py", "make_results_tables.py", "make_segment_check_sheet.py", "pool_summary.py",
    "rebuild_label_sheet.py", "renormalize_selected.py", "run_final_tables.py", "run_segment_models.py", "run_svm.py",
    "run_svm_segments.py", "score_blind_check.py", "translate_online.py", "compare_labellers.py"]]

MOVE_DATA = [D + f for f in [
    "4_dictionary_mapped_reviews_OLD.csv", "5_translated_sample.csv", "aspect_detection.csv", "aspect_detection_appkeyword.csv",
    "blind_check_results.csv", "dictionary_broken_B_malay.csv", "dictionary_variants_vader_english_tuning.csv",
    "dictionary_variants_xlmr_english_tuning.csv", "dictionary_variants_xlmr_malay_tuning.csv", "final_tables.csv", "final_tables.xlsx",
    "hand_label_key.csv", "hand_label_key_OLD.csv", "hand_label_sheet.csv", "hand_label_sheet_OLD.csv", "idiom_literal.csv",
    "label_disagreements_malaya_tuning_malay.csv", "label_disagreements_mbert_tuning.csv", "label_disagreements_mbert_tuning_malay.csv",
    "label_disagreements_roberta_tuning.csv", "label_disagreements_vader_tuning.csv", "label_disagreements_xlmr_tuning.csv",
    "label_disagreements_xlmr_tuning_malay.csv", "label_predictions_malaya_tuning_malay.csv", "label_predictions_mbert_tuning.csv",
    "label_predictions_mbert_tuning_malay.csv", "label_predictions_roberta_tuning.csv", "label_predictions_vader_tuning.csv",
    "label_predictions_xlmr_tuning.csv", "label_predictions_xlmr_tuning_malay.csv", "label_summary_malaya_tuning_malay.csv",
    "label_summary_mbert_tuning.csv", "label_summary_mbert_tuning_malay.csv", "label_summary_roberta_tuning.csv",
    "label_summary_vader_tuning.csv", "label_summary_xlmr_tuning.csv", "label_summary_xlmr_tuning_malay.csv", "label_versions.csv",
    "macam_phrase_candidates.csv", "manglish_candidate_words.csv", "manglish_dictionary_OLD.csv", "my_new_dictionary.csv",
    "my_new_dictionaryBACKUP.csv", "my_new_dictionary_v4.csv", "negative_word_candidates.csv", "normalizer_changes_audit.csv",
    "part_results.csv", "part_results.xlsx", "part_results_per_class.csv", "review_segments.csv", "samples_report.txt",
    "samples_report_DRYRUN.txt", "samples_selected.csv", "samples_selected_DRYRUN.csv", "samples_versions.csv",
    "segment_check_changes.csv", "segment_check_key.csv", "segment_check_sheet.csv", "segment_predictions.csv", "segment_results.csv",
    "segment_results.md", "segment_results.xlsx", "segment_versions.csv", "svm_results_tuning.csv", "telco_specific_candidates.csv",
    "train_versions.csv", "translation_cache_english.json", "translation_cache_malay.json", "translation_cache_online_english.json",
    "v4_questions_examples.csv", "xlmr_label_cache.csv", "~$segment_results.xlsx"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true", help="really move the files (default: only show)")
    args = ap.parse_args()

    todo = sorted(set(MOVE_SCRIPTS + MOVE_DATA))
    clash = [f for f in todo if f in KEEP]
    if clash:
        raise SystemExit(f"Mistake in my list, these are on the KEEP list too: {clash}")

    found = [f for f in todo if (ROOT / f).exists()]
    print(f"{'MOVING' if args.go else 'DRY RUN: would move'} {len(found)} files into _old/  ({len(todo) - len(found)} on my list were not found, fine)\n")
    for f in found:
        print("  ", f)
        if args.go:
            dest = ROOT / "_old" / f
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(ROOT / f), str(dest))

    missing_keep = [f for f in KEEP if not (ROOT / f).exists()]
    if missing_keep:
        print("\nOn the KEEP list but not found (ok if you never made them):")
        for f in missing_keep:
            print("  ", f)
    print("\nNot touched: dashboard/, data/external/, everything already tracked by git, the venv.")
    if args.go:
        print("\nDone. Check it still works:")
        print("  python3 -c \"import sys; sys.path.insert(0,'preprocessing'); import text_versions, manglish_normalizer, dictionary_mapper, segment_reviews; print('ok')\"")
        print("Then add these lines to .gitignore (so they stay out of git):\n  _old/\n  .DS_Store\n  ~$*\n  venv/")
    else:
        print("\nLooks right? Run again with --go")


if __name__ == "__main__":
    main()
    