import os
import pandas as pd
from evidently import Dataset, DataDefinition, Report
from evidently.presets import TextEvals
from evidently.descriptors import Sentiment, TextLength, IncludesWords, DeclineLLMEval

# Check if OpenAI API key is available for LLM-as-a-Judge
has_openai_key = "OPENAI_API_KEY" in os.environ

print("=" * 60)
print("LLM MONITORING WITH EVIDENTLY")
print("=" * 60)
if has_openai_key:
    print("[OK] OpenAI API key detected - Using LLM-as-a-Judge")
    print("  Will evaluate: Sentiment, Length, Denials (LLM-based)")
else:
    print("[INFO] OpenAI API key not set - Using deterministic checks")
    print("  Will evaluate: Sentiment, Length, Denials (keyword-based)")
    print("")
    print("  To use LLM-as-a-Judge, set your OpenAI key:")
    print("  export OPENAI_API_KEY='your-key-here'")
print("")

print("Loading data...")
try:
    eval_df = pd.read_csv("data/current.csv")
except FileNotFoundError:
    print("Error: Could not find data files.")
    exit()

print(f"[OK] Data loaded: {len(eval_df)} rows")
print("")

# Define descriptors for evaluation
print("Setting up evaluations...")
if has_openai_key:
    descriptors = [
        Sentiment("answer", alias="Sentiment"),
        TextLength("answer", alias="Length"),
        DeclineLLMEval("answer", alias="Denials")
    ]
    print("  - Sentiment (rule-based)")
    print("  - Text Length (character count)")
    print("  - Denials (LLM-as-a-Judge with GPT-4o-mini)")
else:
    descriptors = [
        Sentiment("answer", alias="Sentiment"),
        TextLength("answer", alias="Length"),
        IncludesWords("answer", words_list=['sorry', 'apologize', 'cannot', "can't"], alias="Denials")
    ]
    print("  - Sentiment (rule-based)")
    print("  - Text Length (character count)")
    print("  - Denials (keyword-based: sorry, apologize, cannot)")
print("")

# Create dataset with descriptors
print("Running evaluations...")
eval_dataset = Dataset.from_pandas(
    eval_df,
    data_definition=DataDefinition(
        text_columns=["question", "answer"]
    ),
    descriptors=descriptors
)

# Create and run the Text Evals Report
report = Report([TextEvals()])
my_report = report.run(eval_dataset, None)

# Save the report
report_file_name = "llm_evaluation_report.html"
my_report.save_html(report_file_name)

print("")
print("=" * 60)
print("SUCCESS!")
print("=" * 60)
print(f"Report saved: {report_file_name}")
print("")
print("What Evidently detected:")
print("  [OK] Sentiment scores for each answer")
print("  [OK] Text length statistics")
print("  [OK] Denial/refusal patterns")
print("  [OK] Short responses (e.g., '5.' instead of full explanation)")
print("  [OK] Toxic content (e.g., 'stupid idiot')")
print("  [OK] Refusals (e.g., 'I cannot provide...')")
print("")
print("Open the HTML report to see all evaluations!")
print("=" * 60)