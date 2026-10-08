# Industrial Defect Detector

An AI system that checks photos of manufactured products and tells you whether a product is defective, where the defect is, and what kind of defect it is.

Built on PatchCore, an unsupervised anomaly detection method. It only needs photos of good, defect free products to learn what normal looks like. Anything that looks different from normal is flagged automatically, even defect types the system has never seen before.

## Live Demo

Try it here:  https://industrial-defect-detector.vercel.app/

Pick a product category, then click a sample photo to see a result. No upload required to test it.

## Demo Video

https://github.com/user-attachments/assets/7e5aa647-54e9-4acb-af3c-5b471ab9d053

## What It Does

- Detects whether a product is defective or not
- Shows exactly where the defect is on the photo, using a heatmap and an outline
- Names the type of defect, for example a crack, a scratch, or contamination
- Flags photos that do not match any known product as uncertain, instead of guessing
- Works across seven different product categories out of the box
- Can be taught a brand new product using only photos of good samples

## How It Works

1. The system is shown around 200 or more photos of good, undamaged products
2. It builds a memory of what a normal product looks like
3. A new photo is compared against that memory
4. If the photo matches normal, the result is pass
5. If it differs, the result is defect, with the location and type shown
6. If the photo looks nothing like any known product, the result is uncertain

## Methodology

Two phases: teaching the system what a good product looks like, then checking new products against that memory.

```mermaid
flowchart TB
    subgraph TRAIN["Phase 1: Teach the system, done once per product"]
        direction LR
        A["Photos of GOOD products only<br/>around 200 or more"] --> B["Extract visual features<br/>using a pretrained network"]
        B --> C["Build a memory bank<br/>of normal patterns"]
        D["Optional: labeled photos of<br/>known defect types"] --> E["Train a defect type<br/>classifier"]
    end

    subgraph CHECK["Phase 2: Check new products, done every time"]
        direction TB
        F["New photo arrives"] --> G["Compare against<br/>the memory bank"]
        G --> H{"Does it match<br/>a normal product?"}
        H -->|"Yes"| I["Result: PASS"]
        H -->|"No, but looks like<br/>a known product"| J["Result: DEFECT<br/>show location and type"]
        H -->|"No, looks like nothing<br/>the system knows"| K["Result: UNCERTAIN"]
    end

    C --> G
    E --> J
```

## Dataflow: Detecting a Specific Defect

Step by step view of what happens to a single photo, from upload to the final labeled result.

```mermaid
flowchart TD
    A["User uploads a photo<br/>and selects a product category"] --> B["Photo resized to a<br/>standard size"]
    B --> C["Photo split into a<br/>grid of small patches"]
    C --> D["Each patch compared to<br/>the memory bank for that product"]
    D --> E["Each patch gets an<br/>anomaly score"]
    E --> F["Highest patch score becomes<br/>the overall image score"]
    F --> G{"Is the score above<br/>that product's threshold?"}
    G -->|"No"| H["Result: PASS<br/>no defect found"]
    G -->|"Yes"| I["Result: DEFECT"]
    I --> J["Heatmap and outline drawn<br/>over the flawed region"]
    J --> K["Flawed region passed to the<br/>defect type classifier"]
    K --> L["Classifier names the defect<br/>for example crack, scratch,<br/>contamination"]
    L --> M["Final result shown:<br/>verdict, location, defect type,<br/>and confidence"]
    H --> M
```

## Tech Stack

- PatchCore and anomalib for anomaly detection
- PyTorch for running the model
- scikit learn for defect type classification
- FastAPI and Gradio for serving the model
- React for the frontend
- Hugging Face Spaces and Vercel for hosting

## Project Background

This project extends research from my undergraduate thesis on video anomaly detection, applying the same core idea to industrial quality inspection.


## Notes

This project uses the MVTec AD dataset, licensed under CC BY NC SA 4.0, for training and demonstration. It is intended for portfolio and demonstration purposes.
