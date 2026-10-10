# Medical Image Segmentation + XAI

An engineering-focused medical image segmentation project built with PyTorch. The project combines reproducible model development, segmentation evaluation, error analysis, and a FastAPI inference service.

**Dataset:** Kvasir-SEG
**Model:** U-Net
**Runtime:** CPU inference
**Image size:** 256 × 256

> **Research and educational use only.** This project is not validated for clinical diagnosis or treatment decisions.

## Project Highlights

- Reproducible dataset preparation and leakage-safe train/validation/test splits
- U-Net segmentation model implemented with PyTorch
- BCE + Dice loss and AdamW optimization
- Reusable training and validation engine
- Best-checkpoint selection, learning-rate scheduling, and early stopping
- Dice and IoU evaluation
- Experiment tracking and training-curve visualization
- Threshold and boundary-metric analysis
- Visual error analysis and explainability experiments
- FastAPI inference service with input validation
- Automated tests for API behavior and invalid uploads
- CPU-only execution support

## Dataset and Evaluation

The Kvasir-SEG dataset is divided into:

| Split | Images |
|---|---:|
| Training | 700 |
| Validation | 150 |
| Test | 150 |

The baseline model achieved the following test-set results in the recorded experiment:

| Metric | Score |
|---|---:|
| Dice | 0.6144 |
| IoU | 0.4812 |

These are results from a particular experimental configuration, not a guarantee of performance on other datasets or clinical images.

The project also investigates threshold sensitivity, segmentation boundary quality, false-positive and false-negative errors, and alternative loss functions. Results should be interpreted in the context of each experiment's configuration.

## Architecture

```text
Kvasir-SEG Images and Masks
            |
            v
   Data Validation
            |
            v
 Leakage-Safe Data Splits
            |
            v
 Preprocessing and DataLoader
            |
            v
       U-Net Model
            |
            v
  Training and Validation
            |
            v
 Checkpoint and Experiment Logs
            |
            v
 Evaluation and Error Analysis
            |
            v
    FastAPI Inference API
            |
            v
 Probability Map + Binary Mask
```

## Project Structure

```text
medical-segmentation-xai/
├── app/
│   └── main.py
├── artifacts/
│   └── checkpoints/
│       └── best_model.pt
├── configs/
├── data/
├── docker/
├── docs/
├── scripts/
├── src/
│   └── medseg/
├── tests/
│   └── test_api.py
├── pyproject.toml
└── README.md
```

The exact contents of `artifacts/`, `data/`, and other directories depend on the experiments and local setup. Dataset files and trained checkpoints may need to be obtained or generated separately.

## Environment Setup

The project was developed with Python 3.12 and tested in a CPU-only environment.

Create and activate a virtual environment from the project directory:

```cmd
py -3.12 -m venv .venv
call .venv\Scripts\activate.bat
```

Install the project and development dependencies:

```cmd
python -m pip install -e ".[dev]"
```

Check dependency consistency:

```cmd
python -m pip check
```

The declared PyTorch dependencies must be compatible with the target operating system and Python environment. For a different platform, follow the official PyTorch installation guidance if the default package source does not provide a compatible build.

## Run the API

The API loads the model checkpoint during application startup. Make sure the expected checkpoint exists before starting the service.

From the project root:

```cmd
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open the interactive API documentation in a browser:

- Swagger UI: `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/health`
- Model information: `http://127.0.0.1:8000/model-info`

The health endpoint reports whether the model has loaded. The model-information endpoint reports the model name, device, input size, and configured threshold.

## Make a Prediction

Using `curl` from a compatible terminal:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -F "file=@sample.jpg"
```

Alternatively, open `/docs`, select `POST /predict`, upload an image, and execute the request.

The API accepts image uploads up to 10 MiB and returns HTTP errors for unsupported media types, invalid image data, oversized uploads, or an unavailable model.

## Prediction Response

The response includes:

- `filename`: uploaded filename
- `model`: model name
- `input_size`: original image width and height
- `output_size`: output mask width and height
- `threshold`: threshold used to convert probabilities into a binary mask
- `probability_summary`: minimum, maximum, and mean predicted probabilities
- `foreground_ratio`: fraction of output pixels classified as foreground
- `probability_map`: Base64-encoded grayscale PNG of the probability map
- `binary_mask`: Base64-encoded grayscale PNG of the binary segmentation mask

Both PNG images are encoded as Base64 strings. Decode the `data` field to recover the PNG bytes. The probability visualization quantizes probabilities into 8-bit grayscale; it is not a lossless representation of the original floating-point probability array. In the binary mask, foreground pixels are encoded as 255 and background pixels as 0.

The prediction threshold is configurable in the API code. Any deployment threshold should be consistent with the intended evaluation configuration and documented accordingly.

## Run Tests

Run the API tests:

```cmd
python -m pytest tests\test_api.py -v
```

Run the complete project test suite:

```cmd
python -m pytest -v
```

The complete suite has previously passed with **145 tests passed**. Re-run the tests after changing the code or dependencies to verify the current state.

## Limitations

- CPU inference prioritizes accessibility and reproducibility over throughput.
- Predictions may be unreliable on images that differ from the training dataset.
- Foreground ratios and probability summaries are not substitutes for segmentation-quality metrics.
- A successful API response does not establish clinical validity.
- Production deployment requires additional operational safeguards, including external request-size limits, monitoring, and deployment-specific security review.

## Future Engineering Work

- Containerized deployment and reproducible Docker builds
- Structured logging and API performance measurements
- Explicit model/checkpoint versioning
- Deployment smoke tests and CI integration
- Further evaluation of generalization and robustness across data sources
