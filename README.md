\# Customer Churn Prediction - End-to-End ML Project



\## Project Overview



This project is an end-to-end Machine Learning application that predicts whether a customer is likely to churn.



The project covers:



\- Data ingestion

\- Data preprocessing and transformation

\- Model training

\- Model evaluation

\- MLflow experiment tracking

\- Prediction pipeline

\- FastAPI model serving

\- Docker containerization

\- Git and GitHub version control



\## Tech Stack



\- Python

\- Pandas

\- NumPy

\- Scikit-learn

\- Machine Learning

\- MLflow

\- FastAPI

\- Docker

\- Git

\- GitHub

\- Jupyter Notebook



\## Project Architecture



Customer Churn Dataset

&#x20;       |

&#x20;       v

Data Ingestion

&#x20;       |

&#x20;       v

Data Transformation

&#x20;       |

&#x20;       v

Model Training

&#x20;       |

&#x20;       v

Model Evaluation

&#x20;       |

&#x20;       v

Trained Model

&#x20;       |

&#x20;       v

FastAPI

&#x20;       |

&#x20;       v

Docker Container

&#x20;       |

&#x20;       v

Prediction API



\## Project Structure



customer-churn-mlops/

|

|-- data/

|   |-- raw/

|       |-- Telco-Customer-Churn.csv

|

|-- notebooks/

|   |-- customer\_churn.ipynb

|

|-- src/

|   |-- mlproject/

|       |-- components/

|       |   |-- data\_ingestion.py

|       |   |-- data\_transformation.py

|       |   |-- model\_trainer.py

|       |

|       |-- pipeline/

|       |   |-- train\_pipeline.py

|       |   |-- predict\_pipeline.py

|       |

|       |-- exception.py

|       |-- utils/

|           |-- utils.py

|

|-- app.py

|-- dockerfile

|-- requirements.txt

|-- .gitignore

|-- README.md



\## Machine Learning Pipeline



\### 1. Data Ingestion



The raw customer churn dataset is loaded and split into training and testing datasets.



\### 2. Data Transformation



The data is prepared for machine learning using preprocessing techniques.



This includes:



\- Handling missing values

\- Encoding categorical variables

\- Feature transformation

\- Numerical feature preprocessing



\### 3. Model Training



Machine Learning models are trained using the processed training data.



\### 4. Model Evaluation



The trained models are evaluated using the test dataset and appropriate classification metrics.



\### 5. MLflow



MLflow is used for experiment tracking and recording model-related information.



\### 6. Prediction Pipeline



The prediction pipeline loads the trained preprocessing objects and model and generates predictions for new customer data.



\## FastAPI



The trained model is exposed through a REST API using FastAPI.



\### API Endpoint



POST /predict



Example response:



{

&#x20; "prediction": 0

}



The API can be tested using the FastAPI Swagger UI:



http://localhost:8000/docs



\## Docker



The application is containerized using Docker.



Build the Docker image:



docker build -t customer-churn-api .



Run the container:



docker run -p 8000:8000 customer-churn-api



After starting the container, open:



http://localhost:8000/docs



\## Running the Project Locally



Clone the repository:



git clone https://github.com/mehrasatyam121-code/customer-churn-mlops.git



Navigate to the project:



cd customer-churn-mlops



Install dependencies:



pip install -r requirements.txt



Run the training pipeline:



python src/mlproject/pipeline/train\_pipeline.py



Start the FastAPI application:



uvicorn app:app --reload



Open the API documentation:



http://localhost:8000/docs



\## Future Improvements



\- Add CI/CD using GitHub Actions

\- Deploy the API to a cloud platform

\- Add automated model monitoring

\- Add data validation

\- Add unit tests

\- Add model versioning

\- Add automated Docker image builds



\## Author



Satyam Mehra



GitHub:

https://github.com/mehrasatyam121-code

