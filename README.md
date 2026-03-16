# 📈 Salary Prediction Web Application

A **full-stack machine learning application** that predicts salaries through a production-style ML pipeline.

This project includes automated data preprocessing, feature engineering, hyperparameter optimization, model retraining, and interactive prediction UI –– all containerized for scalable deployment.

<p align="center">
    <a href="docs/demo.mp4">
        <img src="docs/demo.gif" width="800" />
    </a>
</p>

[**Quick Start**](#)

## 🔎 Overview

- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Installation & Setup](#️-installation--setup)
    - [Manual](#1-️-manual)
    - [Docker](#2--docker)
- [Usage](#-usage)
    - [Local machine](#️-local-machine-access)
    - [Mobile](#-mobile)
    - [App Instructions](#-app-instructions)
- [Contributing](#-contributing)
- [License](#-license)
- [Credits](#-credits)

## ✨ Features

- **End-to-End Machine Learning System**
    - Automated data preprocessing (cleaning, encoding, feature engineering)
    - Modular ML pipelines for training and inference
    - Model artifact management (`model.joblib`, `metadata.json`)
    - Dynamic model retraining with updated dataset

- **Machine Learning Optimization**
    - Bayesian hyperparameter optimization (Optuna)
    - Multiple ML backends (scikit-learn, Keras/TensorFlow)

- **Interactive Web Application**
    - Responsive UI built with **React + Bootstrap**
    - Real-time salary prediction
    - Interactive salary distribution visualizations (histogram & boxplot)

- **Data & Model Management**
    - Persistent **PostgreSQL database** for salary records
    - Automatic update of training dataset when new data is added
    - User-triggered model retraining workflow
    - Background training process without blocking prediction API

- **API-Based ML Service**
    - REST API built with **FastAPI**
    - Prediction endpoint for real-time inference
    - Model status endpoint for training monitoring
    - Structured schema validation using **Pydantic**

- **Cloud-Ready Infrastructure**
    - Dockerized frontend and backend services
    - Compatible with **Google Cloud Run deployment**
    - CI/CD pipeline support with **GitHub Actions**

## 🛠 Tech Stack

| Layer | Tools |
| :--- | :--- |
| **Frontend:** | React / Vite / Axios / Bootstrap|
| **Backend:** | Python / FastAPI / Uvicorn |
| **Database:** | PostgreSQL / SQLAlchemy |
| **ML:** | Scikit-learn / Tensorflow / Optuna / Pandas / Numpy |
| **visualizations:** | Matplotlib / Seaborn |
| **Cloud:** | Google Cloud Run / Cloud Storage / Cloud SQL |
| **DevOps:** | Docker / Git / Bash / uv|

## 📁 Project Structure

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


## ⚙️ Installation & Setup

### 🐳 Docker:

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

- more informations about `setup` script:
```bash
setup help
```

- mobile device can access frontend through `FRONTEND_ORIGINS` url in `.env`:
```bash
# salary_prediction_web_application/.env
...

FRONTEND_ORIGINS=http://localhost:3000,http://[local IP address]:3000

...
```

use `http://[local IP address]:3000` in your mobile browser

---

## 🚀 Usage

**UI preview:**

- frontend:
![browser frontend](./docx/browser_frontend.png)

- backend:
![browser backend](./docs/browser_backend.png)

---

### 📱 Mobile

**UI preview:**
![mobile frontend](./docs/mobile_frontend.png)


### 📝 App Instructions

- Fill out the form -> click **Predict Salary** button
![instruction1](./readme_images/instruction1.gif)

- Click **see detail** button for extended options
![instruction2](./readme_images/instruction2.gif)

- Change predict value using keyborad or slider
![instruction3](./readme_images/instruction3.gif)

- Click **Add Data** button to store changed prediction
![instruction4](./readme_images/instruction4.gif)

- Click **Retrain Model** button to train on new records
![instruction6](./readme_images/instruction5.gif)

- After retraining, prediction value changes, and the number of records in Train and Test dataset change
![instruction7](./readme_images/instruction6.png)

- Click **Reset Database** button to clear added data in database, and click
**Retrain Model** button again to retrain model with original data.
![instruction8](./readme_images/instruction7.gif)

## 📋 TODO

- Allow input of job title by keyborad (accept unknown jobs).
- A chatbot for user asking questions.

## 🛠 Development Workflow
- Git-based feature branches
- Dockerized reproducible environments
- Auto-retraining compatible with both scikit-learn and TensorFlow pipelines

## 🤝 Contributing

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

## 📄 License

This project is licensed under the [MIT License](./LICENSE).

## 👏 Credits

Thanks to all contributors!

[<img src="https://github.com/StevenHuang41.png" width="50"/>](https://github.com/StevenHuang41)  [<img src="https://github.com/evelynhuang22.png" width="50"/>](https://github.com/evelynhuang22)


See the [contributors list](https://github.com/StevenHuang41/salary_prediction/graphs/contributors)
