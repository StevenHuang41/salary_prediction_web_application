# Salary Prediction Web Application

A **full-stack machine learning application** that predicts salaries through a production-style ML pipeline with cloud-native deployment.

<p align="center">
    <a href="docs/demo.mp4">
        <img src="docs/demo.gif" width="800" />
    </a>
</p>

[**Quick Start**](#installation--setup)

[**Public URL**](https://storage.googleapis.com/salary-prediction-frontend/index.html) (might close backend services due to limited budget)


## Overview

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Workflow](#workflow)
    - [System Architecture](#system-architecture)
    - [Model training process](#model-training-process)
    - [Model Retrain](#model-retrain)
    - [Cloud training](#cloud-training)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
    - [Local machine](#local-machine)
    - [Mobile](#mobile)
    - [App Instructions](#app-instructions)
- [Testing](#testing)
- [TODOs](#todos)
- [Contributing](#contributing)
- [License](#license)
- [Credits](#credits)

## Features

- **End-to-End Machine Learning System**
    - Automated data preprocessing (cleaning, encoding, feature engineering)
    - Modular ML pipelines for training and inference
    - Model selection on candidate models using cross validation
    - Bayesian hyperparameter tuning (Optuna)

- **Interactive Web Application**
    - Responsive UI built with **React + Bootstrap**
    - Real-time salary prediction with interactive distribution visualizations (histogram & boxplot)

- **Data & Model Management**
    - Persistent **PostgreSQL database** for salary records
    - User-triggered model retraining workflow when new data is added
    - Asynchronous background training without blocking inference APIs

- **Scalable Backend Architecture**
    - RESTful API built with **FastAPI** for real-time inference
    - Decoupled training and inference using **Cloud Run Jobs**
    - Structured service layer architecture (DataService, ModelService, PredictionService)

- **Cloud-Native Deployment**
    - Fully containerized system with **Docker**
    - Deployed on **Google Cloud Run + Cloud SQL + Cloud Storage**
    - CI/CD-ready pipeline using **GitHub Actions**

## Tech Stack

| Layer | Tools |
| :--- | :--- |
| **Frontend:** | React / Vite / Axios / Bootstrap|
| **Backend:** | Python / FastAPI / Uvicorn |
| **Database:** | PostgreSQL / SQLAlchemy |
| **ML:** | Scikit-learn / TensorFlow / Optuna / Pandas / NumPy |
| **Visualization:** | Matplotlib / Seaborn |
| **Cloud:** | Google Cloud Run / Cloud Storage / Cloud SQL |
| **DevOps:** | Docker / Git / Bash / uv|

## Project Structure

```text
.
├── apps/
│   ├── backend/
│   └── frontend/
├── setup                   # a script to help system run
├── .env.example
├── .github/
├── .docs/
├── docker-compose.yml
├── LICENSE
└── README.md
```

## Workflow

### System Architecture:
```text
User (Browser / Mobile)
 ⇩
React Frontend (Cloud Storage)
 ⇩
FastAPI Backend (Cloud Run Service)
 ⇩
Service layer
    DataService       ⬄  PostgreSQL (Cloud SQL)
    PredictionService
    ModelService
     ⇩
ML model (in-memory or Cloud Storage)
 ⇩
Response
 ⇩
Frontend UI
```

### Model training process:
```text
Raw data
 ⇩
Clean data  ➩  Database
 ⇩          ⬃
Split data
 ⇩
Preprocess data
    encode
    feature engineering
    scale
 ⇩
Compare candidate models
 ⇩
Hyperparameter tuning on best model
 ⇩
Final training
 ⇩
Evaluate model
 ⇩
Save artifacts
```

### Model Retrain:
```text
User adds new data to Database
 ⇩
User triggers retraining
 ⇩
Fetch dataset from Database
 ⇩
Start training process
 ⇩
Save Artifacts
 ⇩
Backend loads updated model
 ⇩
New prediction flow
```

### Cloud training:
```text
                      User triggers retrain
                    ⬃                      ⬂
      Cloud Run Service                  Frontend starts polling
             ⇩
    Trigger Cloud Run Job                           ⇩             ⬁ No
             ⇩
    Cloud Run Job starts                Check if model finished training
    (ephemeral container)
             ⇩                                       ⇩  yes
 Save artifacts to Cloud Storage
             ⇩                         Reload artifacts from Cloud Storage request
  Job finishes (container dies)
             ⇩                          ⬃
Cloud Run Service load new artifacts
```

## Installation & Setup

### Prerequisites
- Docker: >=29
- npm: >=11.9.0
- node: >=25.6.1
- python: >=3.13

### Clone repo
```bash
# clone the repo
git clone https://github.com/StevenHuang41/salary_prediction_web_application.git
cd salary_prediction_web_application

# setup .env file
setup env

# modify .env file to your setting

# run docker compose
setup -dbv
```

- check on:
    - Frontend: [http://localhost:3000](http://localhost:3000)
    - Backend:  [http://localhost:8080/docs](http://localhost:8080/docs)


- mobile device can access frontend through local IP address:
```bash
setup
# local IP address: xxx.xxx.xxx.xxx
# Updated .env with LAN IP
```

use `http://[local IP address]:3000` in your mobile browser

- more informations about `setup` script:
```bash
setup help
```

---

## Usage

### Local Machine

**UI preview:**

- frontend:
![browser frontend](./docs/browser_frontend.png)

- backend:
![browser backend](./docs/browser_backend.png)

---

### Mobile

**UI preview:**
![mobile frontend](./docs/mobile_frontend.png)

### App Instructions

- Fill out the form -> click **Predict** button
![instruction1](./docs/type_form_predict.gif)

- Click **see detail** button for extended options
![instruction2](./docs/see_detail.gif)

- Change predict value using keyborad or slider
![instruction3](./docs/toggle_value.gif)

- Click **Add Data** button to store changed prediction
![instruction4](./docs/add_data.gif)

- Click **Retrain Model** button to train on new records
After retraining, prediction value changes, and the number of records in Train and Test dataset change
![instruction6](./docs/retrain_predict.gif)

- Click **Reset Database** button to clear added data in database, and click
**Retrain Model** button again to retrain model with original data.
![instruction7](./docs/reset_model.gif)

## Testing

- Backend:
```bash
cd apps/backend
# get into environment and install dependencies
pytest --verbosity=0
```
Result:
![backend_test](./docs/backend_test.png)

- Frontend:
```bash
cd apps/frontend
npm ci
npm test
```
Result:
![frontend_test](./docs/frontend_test.png)

## TODOs

- Allow input of job title by keyborad (accept unknown jobs).
- Add an AI assistant chatbox for user interaction

## Contributing

1. Fork
2. Clone
3. Create a new branch
   ```sh
   git switch -c feature-branch
   ```
4. Commit changes
   ```sh
   git commit -m "Add some feature"
   ```
5. Push
   ```sh
   git push origin feature-branch
   ```
6. Create a Pull Request.

## License

This project is licensed under the [MIT License](./LICENSE).

## Credits

Thanks to all contributors!

[<img src="https://github.com/StevenHuang41.png" width="50"/>](https://github.com/StevenHuang41)  [<img src="https://github.com/evelynhuang22.png" width="50"/>](https://github.com/evelynhuang22)


See the [contributors list](https://github.com/StevenHuang41/salary_prediction/graphs/contributors)
