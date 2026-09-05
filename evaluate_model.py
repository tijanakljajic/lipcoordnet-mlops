from fastapi.testclient import TestClient

from api import app


SAMPLE_COUNT = 3
MAX_AVERAGE_WER_PERCENT = 10.0

def main():
    wer_values = []

    with TestClient(app) as client:
        for sample_index in range(SAMPLE_COUNT):
            response = client.get(f"/predict/{sample_index}")
            response.raise_for_status()

            result = response.json()
            wer = float(result["wer_percent"])
            wer_values.append(wer)

            print(f"\nSample {sample_index}")
            print(f"Prediction: {result['prediction']}")
            print(f"Reference:  {result['reference']}")
            print(f"WER:        {wer:.2f}%")

    average_wer = sum(wer_values) / len(wer_values)
    print(f"\nAverage WER: {average_wer:.2f}%")
    
    if average_wer > MAX_AVERAGE_WER_PERCENT:
        raise SystemExit(
            f"Evaluation failed: average WER {average_wer:.2f}% "
            f"exceeds limit {MAX_AVERAGE_WER_PERCENT:.2f}%."
        )

    print("Evaluation passed.")

if __name__ == "__main__":
    main()