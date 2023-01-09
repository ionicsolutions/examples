# Neptune.ai Examples

## What is Neptune?

Neptune is a metadata store for MLOps, built for teams that run a lot of experiments.

It's used for:

* **Experiment tracking:** Log, display, organize, and compare ML experiments in a single place.
* **Model registry:** Version, store, manage, and query trained models, and model building metadata.
* **Monitoring ML runs live:** Record and monitor model training, evaluation, or production runs live

## Examples

In this repo, you'll find examples of using Neptune to log and retrieve your ML metadata.

You can run every example with zero setup as an "ANONYMOUS" Neptune user (no registration needed).

<i><b>Note</b>: This readme is best viewed in the GitHub Light theme.</i>

## Getting started

| | Docs | Neptune | GitHub | Colab
| ----------- | :---: | :---: | :------: | :---:
| Quickstart | [![docs]](https://docs.neptune.ai/usage/quickstart/) | [![neptune]](https://app.neptune.ai/o/common/org/quickstarts/experiments) | [![github]](how-to-guides/hello-neptune/scripts/hello_neptune.py) | [![colab]](https://colab.research.google.com/github/neptune-ai/examples/blob/master/how-to-guides/hello-neptune/notebooks/hello_neptune.ipynb) |

## How-to guides

### Experiment Tracking

| | Docs | Neptune | GitHub | Colab
| ----------- | :---: | :---: | :------: | :---:
| Organize ML experiments | [![docs]](https://docs.neptune.ai/how-to-guides/experiment-tracking/organize-ml-experiments) | [![neptune]](https://app.neptune.ai/o/common/org/quickstarts/experiments) | [![github]](how-to-guides/organize-ml-experimentation/scripts/Organize_ML_runs.py) | [![colab]](https://colab.research.google.com/github/neptune-ai/examples/blob/master/how-to-guides/organize-ml-experimentation/notebooks/Organize_ML_runs.ipynb) |
| DDP training experiments | [![docs]](https://docs.neptune.ai/tutorials/running_distributed_training) | [![neptune]](https://app.neptune.ai/o/common/org/showroom/experiments?split=tbl&dash=charts&viewId=978feb4d-8f8f-4341-ac50-64e65fbd95bc) | [![github]](how-to-guides/ddp-training/scripts) | |
| Re-run failed training | [![docs]](https://docs.neptune.ai/tutorials/re-running_failed_training/) | [![neptune]](https://app.neptune.ai/o/common/org/showroom/experiments?split=tbl&dash=charts&viewId=97d6d37e-fcb3-4049-af6a-7d45c9f1478d) | [![github]](how-to-guides/re-run-failed-training/scripts/re_run_failed_training.py) | [![colab]](https://colab.research.google.com/github/neptune-ai/examples/blob/master/how-to-guides/re-run-failed-training/notebooks/re_run_failed_training.ipynb) |
| Use Neptune in HPO training job | [![docs]](https://docs.neptune.ai/tutorials/hpo) | [![neptune]](https://app.neptune.ai/o/common/org/pytorch-integration/experiments?split=tbl&dash=Loss-vs-Accuracy-bf72be6c-d771-457f-8f51-30fef2bee3d5&viewId=97f35039-00d1-43ac-9422-3f0ee5b2b0df) | [![github]](how-to-guides/neptune-hpo/scripts/Neptune_hpo_single_run.py) | [![colab]](https://colab.research.google.com/github/neptune-ai/examples/blob/master/how-to-guides/neptune-hpo/notebooks/Neptune_hpo.ipynb) |

### Model Registry

| | Docs | Neptune | GitHub | Colab
| ----------- | :---: | :---: | :------: | :---:
| Log model building metadata | [![docs]](https://docs.neptune.ai/how-to-guides/model-registry) | | | |

### Monitoring ML Runs Live

| | Docs | Neptune | GitHub | Colab
| ----------- | :---: | :---: | :------: | :---:
| Monitor model training runs live | [![docs]](https://docs.neptune.ai/how-to-guides/ml-run-monitoring/monitor-model-training-runs-live) | [![neptune]](https://app.neptune.ai/o/common/org/quickstarts/experiments?viewId=26231575-517f-4d55-acb3-1640bcf537e4) | [![github]](how-to-guides/monitor-ml-runs/scripts/Monitor_ML_runs_live.py) | [![colab]](https://colab.research.google.com/github/neptune-ai/examples/blob/master/how-to-guides/monitor-ml-runs/notebooks/Monitor_ML_runs_live.ipynb) |

### Data Versioning

| | Docs | Neptune | GitHub | Colab
| ----------- | :---: | :---: | :------: | :---:
| Version datasets in model training runs | [![docs]](https://docs.neptune.ai/how-to-guides/data-versioning/version-datasets) | [![neptune]](https://app.neptune.ai/o/common/org/data-versioning/experiments?compare=IwdgNMQ&split=tbl&dash=artifacts&viewId=0d305ea6-3257-4193-9bf0-a7eb571343a1) | [![github]](how-to-guides/data-versioning/scripts/Version_datasets_in_model_training_runs.py) | [![colab]](https://colab.research.google.com/github/neptune-ai/examples/blob/master/how-to-guides/data-versioning/notebooks/Version_datasets_in_model_training_runs.ipynb) |
| Compare datasets between runs | [![docs]](https://docs.neptune.ai/how-to-guides/data-versioning/compare-datasets) | [![neptune]](https://app.neptune.ai/o/common/org/data-versioning/experiments?compare=IwdgNMQ&split=tbl&dash=artifacts&viewId=2b313653-1aa2-40e8-8bf2-cd13f0f96862) | [![github]](how-to-guides/data-versioning/scripts/Compare_model_training_runs_on_dataset_versions.py) | [![colab]](https://colab.research.google.com/github/neptune-ai/examples/blob/master/how-to-guides/data-versioning/notebooks/Compare_model_training_runs_on_dataset_versions.ipynb) |

### Neptune API

| | Docs | Neptune | GitHub | Colab
| ----------- | :---: | :---: | :------: | :---:
| Resume run | [![docs]](https://docs.neptune.ai/how-to-guides/neptune-api/resume-run) | | | |
| Pass run object between files | [![docs]](https://docs.neptune.ai/how-to-guides/neptune-api/pass-run-object-between-files) | | | |
| Use Neptune in distributed computing | [![docs]](https://docs.neptune.ai/how-to-guides/neptune-api/distributed-computing) | | | |
| Use Neptune in parallel computing | [![docs]](https://docs.neptune.ai/how-to-guides/neptune-api/parallel-computing) | | | |
| Use Neptune in Pipelines | [![docs]](https://docs.neptune.ai/how-to-guides/neptune-api/pipelines) | | | |
| Log to multiple runs in one script | [![docs]](https://docs.neptune.ai/how-to-guides/neptune-api/log-to-multiple-runs-in-one-script) | | | |
| Create and delete projects | [![docs]](https://docs.neptune.ai/how-to-guides/neptune-api/create-and-delete-projects) | | [![github]](how-to-guides/create-delete-projects/scripts/create_delete_projects.py) | [![colab]](https://colab.research.google.com/github/neptune-ai/examples/blob/master/how-to-guides/create-delete-projects/notebooks/Create_delete_projects.ipynb) |

### Neptune app

| | Docs | Neptune | GitHub | Colab
| ----------- | :---: | :---: | :------: | :---:
| Do GroupBy on runs | [![docs]](https://docs.neptune.ai/how-to-guides/neptune-ui/groupby) | | | |
| Do sorting | [![docs]](https://docs.neptune.ai/how-to-guides/neptune-ui/sorting-runs) | | | |

## Integrations and Supported Tools

### Languages

| | Docs | Neptune | GitHub | Colab
| ----------- | :---: | :---: | :------: | :---:
| Python | [![docs]](https://docs.neptune.ai/integrations-and-supported-tools/languages/neptune-client-python) | [![neptune]](https://app.neptune.ai/o/common/org/quickstarts/experiments?viewId=d48562e1-a494-4fd0-b3bb-078240516a4f) | [![github]](how-to-guides/hello-neptune/scripts/hello_neptune.py) | [![colab]](https://colab.research.google.com/github/neptune-ai/examples/blob/master/how-to-guides/hello-neptune/notebooks/hello_neptune.ipynb) |
| R | [![docs]](https://docs.neptune.ai/integrations-and-supported-tools/languages/neptune-client-r) | | | |

### Model Training

| | Docs | Neptune | GitHub | Colab
| ----------- | :---: | :---: | :------: | :---:
| Catalyst | [![docs]](https://docs.neptune.ai/integrations-and-supported-tools/model-training/catalyst) | [![neptune]](https://app.neptune.ai/o/common/org/catalyst-integration/e/CATALYST-1486/charts) | [![github]](integrations-and-supported-tools/catalyst/scripts) | [![colab]](https://colab.research.google.com/github/neptune-ai/examples/blob/main/integrations-and-supported-tools/catalyst/notebooks/Neptune_Catalyst.ipynb) |
| fastai | [![docs]](https://docs.neptune.ai/integrations-and-supported-tools/model-training/fastai) | [![neptune]](https://app.neptune.ai/o/common/org/fastai-integration/e/FAS-61/dashboard/fastai-dashboard-1f456716-f509-4432-b8b3-a7f5242703b6) | [![github]](integrations-and-supported-tools/fastai/scripts) | [![colab]](https://colab.research.google.com/github/neptune-ai/examples/blob/main/integrations-and-supported-tools/fastai/notebooks/Neptune_fastai.ipynb) |
| lightGBM | [![docs]](https://docs.neptune.ai/integrations-and-supported-tools/model-training/lightgbm) | [![neptune]](https://app.neptune.ai/o/common/org/lightgbm-integration/e/LGBM-86/dashboard/train-cls-9d622664-d419-42db-b32a-c44c12bd44d1) | [![github]](integrations-and-supported-tools/lightgbm/scripts) | [![colab]](https://colab.research.google.com/github/neptune-ai/examples/blob/main/integrations-and-supported-tools/lightgbm/notebooks/Neptune_LightGBM.ipynb) |
| Prophet | [![docs]](https://docs.neptune.ai/integrations-and-supported-tools/model-training/prophet) | [![neptune]](https://app.neptune.ai/common/fbprophet-integration/e/FBPROP-249/all) | [![github]](integrations-and-supported-tools/prophet/scripts) | [![colab]](https://colab.research.google.com/github/neptune-ai/examples/blob/main/integrations-and-supported-tools/prophet/notebooks/Neptune_prophet.ipynb) |
| PyTorch | [![docs]](https://docs.neptune.ai/integrations-and-supported-tools/model-training/pytorch) | [![neptune]](https://app.neptune.ai/o/common/org/pytorch-integration/e/PYTOR1-54/dashboard/Experiment--4d82bf2c-2515-476d-bf56-728e87e491d4) | [![github]](integrations-and-supported-tools/pytorch/scripts) | [![colab]](https://colab.research.google.com/github/neptune-ai/examples/blob/main/integrations-and-supported-tools/pytorch/notebooks/Neptune_PyTorch_Support.ipynb) |
| PyTorch Ignite | [![docs]](https://docs.neptune.ai/integrations-and-supported-tools/model-training/pytorch-ignite) | | | |
| PyTorch Lightning | [![docs]](https://docs.neptune.ai/integrations-and-supported-tools/model-training/pytorch-lightning) | [![neptune]](https://app.neptune.ai/o/common/org/pytorch-lightning-integration/e/PTL-11/dashboard/simple-6ff16e4c-c529-4c63-b437-dfb883131793) | [![github]](integrations-and-supported-tools/pytorch-lightning/scripts) | [![colab]](https://colab.research.google.com/github/neptune-ai/examples/blob/master/integrations-and-supported-tools/pytorch-lightning/notebooks/Neptune_PyTorch_Lightning.ipynb) |
| scikit-learn | [![docs]](https://docs.neptune.ai/integrations-and-supported-tools/model-training/sklearn) | [![neptune]](https://app.neptune.ai/o/common/org/sklearn-integration/e/SKLEAR-97/all?path=rfr_summary%2Fdiagnostics_charts&attribute=feature_importance) | [![github]](integrations-and-supported-tools/sklearn/scripts) | [![colab]](https://colab.research.google.com/github/neptune-ai/examples/blob/master/integrations-and-supported-tools/sklearn/notebooks/Neptune_Scikit_learn.ipynb) |
| skorch | [![docs]](https://docs.neptune.ai/integrations-and-supported-tools/model-training/skorch) | [![neptune]](https://app.neptune.ai/o/common/org/skorch-integration/e/SKOR-32/dashboard/skorch-dashboard-97de6fa9-92dd-4b76-9842-b1fbe9cc992e) | [![github]](integrations-and-supported-tools/skorch/scripts) | [![colab]](https://colab.research.google.com/github/neptune-ai/examples/blob/master/integrations-and-supported-tools/skorch/notebooks/Neptune_Skorch.ipynb) |
| 🤗 Transformers | [![docs]](https://docs.neptune.ai/integrations-and-supported-tools/model-training/hugging-face) | [![neptune]](https://app.neptune.ai/o/showcase/org/project-text-summarization-hf/e/PROJ-138/dashboard/Custom-Dashboard-97370bc5-ee32-48ce-a630-fc806a370e13)  | [![github]](integrations-and-supported-tools/transformers/scripts) | [![colab]](https://colab.research.google.com/github/neptune-ai/examples/blob/master/integrations-and-supported-tools/transformers/notebooks/Neptune_Transformers.ipynb) |
| TensorFlow | | [![neptune]](https://app.neptune.ai/o/common/org/tensorflow-support/e/TFSUP-101/dashboard/Overview-97f6ac04-2c4b-4d10-97da-cd3a51bbeec8) | [![github]](integrations-and-supported-tools/tensorflow/scripts) | [![colab]](https://colab.research.google.com/github/neptune-ai/examples/blob/master/integrations-and-supported-tools/tensorflow/notebooks/Neptune_Tensorflow.ipynb) |
| TensorFlow / Keras | [![docs]](https://docs.neptune.ai/integrations-and-supported-tools/model-training/tensorflow-keras) | [![neptune]](https://app.neptune.ai/o/common/org/tf-keras-integration/e/TFK-18/all) | [![github]](integrations-and-supported-tools/tensorflow-keras/scripts) | [![colab]](https://colab.research.google.com/github/neptune-ai/examples/blob/master/integrations-and-supported-tools/tensorflow-keras/notebooks/Neptune_TensorFlow_Keras.ipynb) |
| XGBoost | [![docs]](https://docs.neptune.ai/integrations-and-supported-tools/model-training/xgboost) | [![neptune]](https://app.neptune.ai/o/common/org/xgboost-integration/e/XGBOOST-84/all?path=training) | [![github]](integrations-and-supported-tools/xgboost/scripts) | [![colab]](https://colab.research.google.com/github/neptune-ai/examples/blob/main/integrations-and-supported-tools/xgboost/notebooks/Neptune_XGBoost.ipynb) |

### Hyperparameter Optimization

| | Docs | Neptune | GitHub | Colab
| ----------- | :---: | :---: | :------: | :---:
| Keras Tuner | [![docs]](https://docs.neptune.ai/integrations-and-supported-tools/hyperparameter-optimization/keras-tuner) | | | |
| Optuna | [![docs]](https://docs.neptune.ai/integrations-and-supported-tools/hyperparameter-optimization/optuna) | [![neptune]](https://app.neptune.ai/o/common/org/optuna-integration/experiments?split=bth&dash=parallel-coordinates-plot&viewId=b6190a29-91be-4e64-880a-8f6085a6bb78) | [![github]](integrations-and-supported-tools/optuna/scripts) | [![colab]](https://colab.research.google.com/github/neptune-ai/examples/blob/master/integrations-and-supported-tools/optuna/notebooks/Neptune_Optuna_integration.ipynb) |
| Scikit-Optimize | [![docs]](https://docs.neptune.ai/integrations-and-supported-tools/hyperparameter-optimization/scikit-optimize) | | | |

### Model Visualization and Debugging

| | Docs | Neptune | GitHub | Colab
| ----------- | :---: | :---: | :------: | :---:
| Altair | [![docs]](https://docs.neptune.ai/integrations-and-supported-tools/model-visualization-and-debugging/altair) | [![neptune]](https://app.neptune.ai/common/altair-support/e/AL-1/all?path=&attribute=interactive_img) | [![github]](integrations-and-supported-tools/altair/scripts/Neptune_Altair_Support.py) | [![colab]](https://colab.research.google.com/github/neptune-ai/examples/blob/main/integrations-and-supported-tools/altair/notebooks/Neptune_Altair_Support.ipynb) |
| Bokeh | [![docs]](https://docs.neptune.ai/integrations-and-supported-tools/model-visualization-and-debugging/bokeh) | [![neptune]](https://app.neptune.ai/common/bokeh-support/e/BOK-1/all?path=&attribute=interactive_img) | [![github]](integrations-and-supported-tools/bokeh/scripts/Neptune_Bokeh_Support.py) | [![colab]](https://colab.research.google.com/github/neptune-ai/examples/blob/main/integrations-and-supported-tools/bokeh/notebooks/Neptune_Bokeh_Support.ipynb) |
| Dalex | [![docs]](https://docs.neptune.ai/integrations-and-supported-tools/model-visualization-and-debugging/dalex) | | | |
| HiPlot | [![docs]](https://docs.neptune.ai/integrations-and-supported-tools/model-visualization-and-debugging/hiplot) | | | |
| HTML | [![docs]](https://docs.neptune.ai/integrations-and-supported-tools/model-visualization-and-debugging/html) | [![neptune]](https://app.neptune.ai/common/html-support/e/HTMLSUP-3/all?path=&attribute=html_obj) | [![github]](integrations-and-supported-tools/html/scripts/Neptune_HTML_Support.py) | [![colab]](https://colab.research.google.com/github/neptune-ai/examples/blob/master/integrations-and-supported-tools/html/notebooks/Neptune_HTML_Support.ipynb) |
| Matplotlib | [![docs]](https://docs.neptune.ai/integrations-and-supported-tools/model-visualization-and-debugging/matplotlib) | [![neptune]](https://app.neptune.ai/common/matplotlib-support/e/MAT-1/all?path=&attribute=interactive-img) | [![github]](integrations-and-supported-tools/matplotlib/scripts) | [![colab]](https://colab.research.google.com/github/neptune-ai/examples/blob/master/integrations-and-supported-tools/matplotlib/notebooks/Neptune_Matplotlib_Support.ipynb) |
| Pandas | [![docs]](https://docs.neptune.ai/integrations-and-supported-tools/model-visualization-and-debugging/pandas) | | | |
| Plotly | [![docs]](https://docs.neptune.ai/integrations-and-supported-tools/model-visualization-and-debugging/plotly) | [![neptune]](https://app.neptune.ai/common/plotly-support/e/PLOT-2/all?path=&attribute=interactive_img) | [![github]](integrations-and-supported-tools/plotly/scripts/Neptune_Plotly_Support.py) | [![colab]](https://colab.research.google.com/github/neptune-ai/examples/blob/main/integrations-and-supported-tools/plotly/notebooks/Neptune_Plotly_Support.ipynb) |

### Automation Pipelines

| | Docs | Neptune | GitHub | Colab
| ----------- | :---: | :---: | :------: | :---:
| Kedro | [![docs]](https://docs.neptune.ai/integrations-and-supported-tools/automation-pipelines/kedro) | [![neptune]](https://app.neptune.ai/o/common/org/kedro-integration/e/KED-632/dashboard/Basic-pipeline-metadata-42874940-da74-4cdc-94a4-315a7cdfbfa8) | [![github]](integrations-and-supported-tools/kedro/scripts) | |

### Experiment Tracking

| | Docs | Neptune | GitHub | Colab
| ----------- | :---: | :---: | :------: | :---:
| MLflow | [![docs]](https://docs.neptune.ai/integrations-and-supported-tools/experiment-tracking/mlflow) | | | |
| Sacred | [![docs]](https://docs.neptune.ai/integrations-and-supported-tools/experiment-tracking/sacred) | [![neptune]](https://app.neptune.ai/o/common/org/sacred-integration/e/SAC-11/dashboard/Sacred-Dashboard-6741ab33-825c-4b25-8ebb-bb95c11ca3f4) | [![github]](integrations-and-supported-tools/sacred/scripts) | [![colab]](https://colab.research.google.com/github/neptune-ai/examples/blob/main/integrations-and-supported-tools/sacred/notebooks/Neptune_Sacred.ipynb) |
| TensorBoard | [![docs]](https://docs.neptune.ai/integrations-and-supported-tools/experiment-tracking/tensorboard) | | | |

### IDEs and Notebooks

| | Docs | Neptune | GitHub | Colab
| ----------- | :---: | :---: | :------: | :---:
| Any IDE | [![docs]](https://docs.neptune.ai/integrations-and-supported-tools/ide-and-notebooks/any-ide) | | | |
| Amazon SageMaker notebooks | [![docs]](https://docs.neptune.ai/integrations-and-supported-tools/ide-and-notebooks/amazon-sagemaker) | | | |
| Deepnote | [![docs]](https://docs.neptune.ai/integrations-and-supported-tools/ide-and-notebooks/deepnote) | | | |
| Google Colab | [![docs]](https://docs.neptune.ai/integrations-and-supported-tools/ide-and-notebooks/google-colab) | [![neptune]](https://app.neptune.ai/o/common/org/showroom/e/SHOW-37) | [![github]](integrations-and-supported-tools/colab/Neptune_Colab.ipynb) | [![colab]](https://colab.research.google.com/github/neptune-ai/examples/blob/master/integrations-and-supported-tools/colab/Neptune_Colab.ipynb) |
| Jupyter Notebook and JupyterLab | [![docs]](https://docs.neptune.ai/integrations-and-supported-tools/ide-and-notebooks/jupyter-lab-and-jupyter-notebook) | | | |

### Data Versioning

| | Docs | Neptune | GitHub | Colab
| ----------- | :---: | :---: | :------: | :---:
| AWS S3 | [![docs]](https://docs.neptune.ai/integrations-and-supported-tools/data-versioning/aws-s3) | | | |
| Data Version Control (DVC) | [![docs]](https://docs.neptune.ai/integrations-and-supported-tools/data-versioning/data-version-control-dvc) | | | |
| Local filesystem | [![docs]](https://docs.neptune.ai/integrations-and-supported-tools/data-versioning/local-files-and-directories) | | | |

### Continuous Integration and Delivery (CI/CD)

| | Docs | Neptune | GitHub | Colab
| ----------- | :---: | :---: | :------: | :---:
| GitHub Actions | [![docs]](https://docs.neptune.ai/integrations-and-supported-tools/continuos-integration-and-delivery-ci-cd/github-actions) | | | |

### Amazon SageMaker support

| | Docs | Neptune | GitHub |
| ----------- | :---: | :---: | :------: |
| Using Neptune in training jobs with custom Docker containers | | [![neptune]](https://app.neptune.ai/common/showroom/e/SHOW-29006) | [![github]](integrations-and-supported-tools/sagemaker/custom-docker-container/) |
| Using Neptune in training jobs with PyTorch Estimator | | [![neptune]](https://app.neptune.ai/common/showroom/e/SHOW-29007) | [![github]](integrations-and-supported-tools/sagemaker/pytorch/) |



[docs]: https://neptune.ai/wp-content/uploads/documentaton-icon.png "Read the documentation"

[neptune]: https://neptune.ai/wp-content/uploads/neptune-icon.png "See Neptune example"

[github]: https://neptune.ai/wp-content/uploads/github-icon.png "See code on GitHub"

[colab]: https://neptune.ai/wp-content/uploads/colab-icon.png "Open in Colab"
