# PDF Outline Extractor - Adobe Hackathon Challenge 1A

This project is a solution for Challenge 1A of the Adobe India Hackathon, "Connecting the Dots." It's designed to extract a structured outline (Title, H1, H2, H3) from a given PDF document and output it in a clean, hierarchical JSON format.

The solution is built to be fast, accurate, and fully compliant with the hackathon's constraints, including offline execution, a small model footprint (≤ 200MB), and fast processing time (≤ 10 seconds for a 50-page document) on a CPU-only environment.

---

## Our Approach

To meet the strict performance and accuracy requirements, we implemented a **Hybrid Machine Learning + Heuristic Approach**. This method combines the speed of deterministic feature extraction with the intelligence of a lightweight machine learning model, avoiding the overhead of larger Deep Learning architectures.

The core philosophy is to treat structure detection as a classification problem: each line of text in the PDF is classified as a `Title`, `H1`, `H2`, `H3`, or `Body` text.

Our pipeline works as follows:

1.  **High-Speed PDF Parsing**: We use the `PyMuPDF` library to rapidly parse the input PDF. For every line of text, we extract a rich set of features that go beyond simple font size.

2.  **Advanced Feature Engineering**: Our key to accuracy lies in the features we generate for each line:
    * **Font Properties**: Absolute font size, bold status.
    * **Positional Properties**: Normalized X-position (to detect indentation), and vertical spacing from the previous line (headings often have more whitespace).
    * **Textual Properties**: Word count, character count, whether the text is in ALL CAPS, and if it starts with a common numbering pattern (e.g., "1.2", "A.", etc.).
    * **Relative Properties**: The line's font size relative to the most common (body) font size on the page. This helps normalize documents with varying font scales.

3.  **Lightweight Classification Model**: These feature vectors are fed into a pre-trained **LightGBM** classifier. We chose LightGBM because it is exceptionally fast on CPUs, has a very small memory footprint, and provides high accuracy for tabular data, making it a perfect fit for the hackathon's constraints.

4.  **Inference and Output**: The final script uses the trained model to predict the role of each line in the input PDFs. The predictions are then assembled into the final, required JSON structure.

---

## Models and Libraries Used

This solution is built entirely in Python and relies on a few key, high-performance libraries.

* **`PyMuPDF`**: The core library for all PDF parsing. It's used to extract text, bounding boxes, fonts, and other low-level details with high speed.
* **`LightGBM`**: A gradient boosting framework used as our classification model. It's responsible for predicting the role of each text line based on its features.
* **`Pandas`**: Used for efficient data manipulation during the feature extraction and training phases.
* **`Scikit-learn`**: Used for its `LabelEncoder` to prepare class labels for the model and for splitting data during training.
* **`Joblib`**: Used to serialize and save our trained LightGBM model to a single, compact file (`document_structure_model.joblib`).

---

## How to Build and Run

The solution is packaged in a Docker container for easy and consistent execution.

### Prerequisites

* Docker must be installed and running on your system.

### Step 1: Build the Docker Image

Navigate to the root directory of the project (where the `Dockerfile` is located) and run the following command to build the image.

```bash
docker build --platform linux/amd64 -t mysolutionname:somerandomidentifier .
````

### Step 2: Run the Solution

Once the image is built, use the following command to run the container. This command will automatically mount your local `input` and `output` directories into the container, allowing the script to process the PDFs.

```bash
docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output --network none mysolutionname:somerandomidentifier
```

The container will automatically:

1.  Scan the `/app/input` directory for all `.pdf` files.
2.  Process each PDF using the trained model.
3.  Generate a corresponding `.json` file in the `/app/output` directory for each input PDF.

<!-- end list -->
