# Databricks Computer Vision Architecture

An advanced, modular, and extensible reference architecture designed to simplify the adoption and deployment of sophisticated computer vision pipelines. This architecture leverages standard frameworks and protocols, promoting consistent and efficient workflows by integrating Databricks with PyTorch Lightning for structured model training, Ray for distributed computation and hyperparameter optimization, Hugging Face Transformers for a standardized, robust model repository, MLflow for uniform experiment tracking and model logging, Albumentations for consistent and effective data augmentation, and PyCOCOTools for standardized data annotations and management.

---

## 🎯 Why This Project?

Implementing production-ready computer vision solutions can be complex. This architecture aims to:

* **Simplify Deployment:** Abstract the complexities of distributed training, hyperparameter tuning, model tracking, and monitoring.
* **Best Practices:** Integrate industry-leading tools and frameworks to ensure scalability, reproducibility, and maintainability.
* **Ease of Adoption:** Provide clear, structured, and easy-to-follow workflows for rapid development and deployment.

---

## 🚀 Technology Stack and Its Significance

The architecture integrates standardized frameworks and tools to achieve a robust and maintainable computer vision pipeline:

* **PyTorch Lightning**: Establishes a standardized protocol for model training, validation, and testing, reducing boilerplate code and ensuring reproducibility through structured code.

* **Ray**: Provides standardized and scalable distributed training and hyperparameter tuning capabilities, crucial for leveraging large-scale computing clusters efficiently.

* **Hugging Face Transformers**: Serves as a unified model repository offering well-established architectures and pretrained models, ensuring rapid integration and deployment of state-of-the-art models such as DETR and YOLO.

* **MLflow**: Implements a uniform system for experiment tracking, logging metrics, and managing model versions, enhancing reproducibility, traceability, and deployment readiness.

* **Albumentations**: Provides standardized data augmentation techniques to enhance model generalization, consistency, and performance across diverse datasets.

* **PyCOCOTools**: Standardizes data annotation and management using the widely recognized COCO format, enabling consistent data handling, evaluation metrics, and interoperability across datasets and tasks.

---

## 🧩 Modularity and Extensibility

The project emphasizes modularity and extensibility through clear abstractions:

* **UnifiedTrainer**: Abstracts training logic, seamlessly managing local and distributed environments.
* **DetectionModel & DataModule**: Separately handle data and model logic, promoting independent maintenance and ease of integration.
* **Output Adapters**: Facilitate straightforward integration of new Hugging Face models, enabling easy extension of the architecture.

---

## 🔧 How to Introduce New Models via Adapters

To integrate a new Hugging Face detection model into this architecture, follow these steps:

### Step 1: Create a New Adapter

Implement the `OutputAdapter` abstract class to facilitate integration of new Hugging Face models. This abstraction standardizes interactions with different model architectures, enabling new models to seamlessly plug into the existing workflow without requiring refactoring of the core `model.py` or `data.py` files.

Each method within `OutputAdapter` serves a specific and crucial purpose:

* **`adapt_output(outputs)`**: Converts the raw outputs from your model into a standardized format expected by the training pipeline (e.g., bounding boxes format). This ensures compatibility across different model outputs.

* **`adapt_targets(targets)`**: Transforms the target annotations into the format required by your model during training. This standardizes how ground truth data is fed into models, irrespective of individual model-specific requirements.

* **`format_predictions(outputs)`**: Structures model predictions for metric computations such as mean Average Precision (mAP). This method ensures consistency in how evaluation metrics are computed and interpreted across different model types.

* **`format_targets(targets)`**: Formats the target annotations to match the standardized metric computation format. This guarantees consistent evaluation against ground truths.

Example:

```python
from adapter import OutputAdapter

class YourModelOutputAdapter(OutputAdapter):
    def adapt_output(self, outputs):
        # Convert model-specific outputs into standardized dictionary format
        standardized_outputs = {
            "boxes": outputs["boxes"],
            "logits": outputs["scores"],
            "loss": outputs.get("loss"),
        }
        return standardized_outputs

    def adapt_targets(self, targets):
        # Format targets into your model's required structure
        model_targets = [{
            "labels": t["class_labels"],
            "boxes": t["boxes"]
        } for t in targets]
        return model_targets

    def format_predictions(self, outputs):
        # Format predictions for metric computations
        predictions = [{
            "boxes": output["boxes"],
            "scores": output["logits"].max(dim=1).values,
            "labels": output["logits"].argmax(dim=1)
        } for output in outputs]
        return predictions

    def format_targets(self, targets):
        # Ensure targets are correctly formatted for metrics
        formatted_targets = [{
            "boxes": t["boxes"],
            "labels": t["class_labels"]
        } for t in targets]
        return formatted_targets
```

### Step 2: Register Your Adapter

Update the adapter factory:

```python
def get_output_adapter(model_name: str) -> OutputAdapter:
    if "your_model_name" in model_name.lower():
        return YourModelOutputAdapter()
    elif "detr" in model_name.lower():
        return DETROutputAdapter()
    # Add other models similarly
```

### Step 3: Configure Your Model

Create a new configuration in your YAML config:

```yaml
model:
  task_type: detection
  model_name: your_model_identifier
  num_classes: 80
```

---

## 🚦 Getting Started

Clone the repository into your Databricks workspace:

```bash
git clone https://github.com/Aradhya0510/databricks-cv-architecture.git
cd databricks-cv-architecture
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Explore the provided notebooks to:

* Set up the required Unity Catalog schema and volume.
* Build your dataset and dataloaders tailored for training.
* Conduct model training, evaluation, and deployment.

The project includes a series of notebooks that demonstrate the complete workflow for training a DETR (DEtection TRansformer) model on the COCO 2017 dataset:

1. `00_setup_and_config.ipynb`: Environment setup and configuration
2. `01_data_preparation.ipynb`: Dataset preparation and preprocessing
3. `02_model_training.ipynb`: Model training and evaluation
4. `03_hparam_tuning.ipynb`: Hyperparameter optimization
5. `04_model_evaluation.ipynb`: Comprehensive model evaluation
6. `05_model_registration_deployment.ipynb`: Model registration and deployment
7. `06_model_monitoring.ipynb`: Model monitoring and maintenance

These notebooks provide a step-by-step guide to:
- Setting up the environment
- Preparing and preprocessing data
- Training and evaluating models
- Optimizing hyperparameters
- Registering and deploying models
- Monitoring model performance

Walk through these notebooks sequentially to familiarize yourself with the end-to-end workflow, and feel encouraged to adapt and customize them for your specific use cases.

---

## 🌱 Future Directions

* Expand adapters for segmentation and classification tasks.
* Enhance support for more diverse data formats.
* Introduce plugin-based adapter registry for easier extensibility.

---

## 📚 Documentation & Contributions

Contributions are welcomed! Please document thoroughly and maintain consistent coding standards when contributing.

---

### 📬 Questions & Feedback

Reach out via GitHub issues or email for support and suggestions!

## Contributing

We welcome contributions to improve the reference architecture. Please see our [contributing guidelines](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 