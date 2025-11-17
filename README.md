# 📈 Salary Prediction Web Application

A self-hosted full-stack web application that predicts your salary
based on user input using machine learning models. Includes dynamc retraining, interactive data visualization, and modular architecture built for sclability.

<!-- [![GitHub stars](https://img.shields.io/github/stars/StevenHuang41/salary_prediction)](https://github.com/StevenHuang41/salary_prediction/stargazers) -->
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)
![number of visitors](https://visitor-badge.laobi.icu/badge?page_id=StevenHuange41.salary_prediction)


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

- **Interactive frontend** with clean UI
- **Real-time** salary predictions
- **Editable** prediction values and retraining
- **Data preprocessing**——cleaning, encoding, and feature engineering
- **Automated model selection**
- **Hyperparameter Optimization**
    - `BayesSearchCV` for scikit-learn models
    - `Keras Tuner` for MLP neural network
- **Data persistence**——SQLite
- **Data visualization**——matplotlib and seaborn libraies
- **Containerized development**——Docker

## 🛠 Tech Stack

| Layer | Tools |
| :---: | :--- |
| **Frontend:** | React / Vite / Vitest / Bootstrap|
| **Backend:** | Python / FastAPI / uvicorn |
| **Database:** | Sqlite3 / Pandas |
| **ML:** | scikit-learn / Tensorflow / Keras Tuner / BayesSearchCV |
| **Other:** | Docker / Git / Bash / uv|

## 📁 Project Structure

### root

```sh
.
├── README.md
├── docker-compose.yml
├── setup
├── frontend/
├── backend/
├── readme_images/
└── .gitignore
```

### Frontend

```sh
frontend/
├── README.md
├── Dockerfile
├── package.json
├── package-lock.json
├── vite.config.js
├── eslint.config.js
├── index.html
├── src/
├── public/
├── .gitignore
└── .dockerignore
```

### Backend

```sh
backend/
├── README.md
├── Dockerfile
├── main.py
├── uv.lock
├── pyproject.toml
├── requirements.txt
├── my_package/
├── database/
├── .gitignore
└── .dockerignore
```

## ⚙️ Installation & Setup

### • 🧩 Prerequisites

- **Python:** >=3.8

- **Node.js:** >=20.9.0 (LTS recommended)

- **npm:** >=10.0.0

- (Optional) **[uv](https://github.com/astral-sh/uv/)** for faster python packages installation.

- (Optional) **[Docker](https://www.docker.com/)** for containerized setup.

Check version with:
```sh
[tool] --version
```
Replace `[tool]` with `python`, `node`, `npm`, etc.

### • 🔐 Clone the repo:

```sh
# SSH:
git clone git@github.com:StevenHuang41/salary_prediction.git

# or HTTPS:
git clone https://github.com/StevenHuang41/salary_prediction.git

cd salary_prediction
```

---

### • 🧱 Installation & Setup Methods:

- [Manual Installation](#1-️-manual)

- [Docker Installation & Setup](#2--docker) (Recommended)

---

### 1. 🕹️ Manual

- #### Frontend Installation:
    ```sh
    cd frontend
    npm install
    cd ..
    ```

- #### Backend Installation:
    ```sh
    cd backend

    # make sure you activated a virtual environment
    pip install -r requirements.txt

    cd ..

    # or use uv to install packages (install uv first) faster than pip install
    curl -LsSf https://astral.sh/uv/install.sh | sh
    cd backend
    uv sync --locked
    cd ..
    ```

- #### Setup

    use `setup` to get __local IP address__ and create `.env.local` files
    ```sh
    ./setup
    ```
    **Expected result:**
    ![setup image](./readme_images/setup_image.png)

- #### Start Servers

    open 4 terminals, and run each command respectively.

    - **Frontend test**

        ```sh
        cd frontend
        npm test
        ```

        **Expected result:**
        ![frontend test](./readme_images/frontend_test.png)

    ---

    - **Frontend server**

        ```sh
        cd frontend
        npm run dev
        ```

        expected result:
        ![frontend server](./readme_images/frontend_server.png)

    ---

    - **Backend server**

        for basic api request

        ```sh
        cd backend
        python main.py 8001

        # or use uv to run
        uv run main.py 8001
        ```

        expected result:
        ![backend server](./readme_images/backend_server.png)

    ---

    - **Backend training server**

        ```sh
        cd backend
        python main.py 8000

        # or use uv to run
        uv run main.py 8000
        ```
        expected result:
        ![training server](./readme_images/training_server.png)

---

### 2. 🐳 Docker

Docker handles packages installation & setup, which is much easier than manual installation.

```sh
cd salary_prediction
./setup build
```
`./setup build` will create .env.local and run `docker compose up --build`, which automatically installs packages, creates environment and runs the app.

see `./setup --help` for further setup shell script imformations

**Expected result:**
![setup build](./readme_images/setup_build.png)

wait until all servers are successfully built, and then click the URL in the next section


## 🚀 Usage

### 🖥️ Local Machine Access

- **Frontend:** <http://localhost:3000>

- **Backend:** <http://localhost:8001/docs>

- **Training:** <http://localhost:8000/docs>

**UI preview:**

- frontend:
![browser frontend](./readme_images/browser_frontend.png)

- backend:
![browser backend](./readme_images/browser_backend.png)

---

### 📱 Mobile

- Enter `http://[local IP address]:3000/` in your mobile browser
    Replace `[local IP address]` with your local machine [IP address](#setup)

**UI preview:**
![mobile frontend](./readme_images/mobile_frontend.png)


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

## 🤝 Contributing

1. Fork the repository.
2. Clone your fork to your local machine.
   ```sh
   git clone https://github.com/[your-username/repo-name].git
   cd [repo-name]
   ```
3. Create a new branch
   ```sh
   git switch -c feature-branch
   ```
4. Commit your changes
   ```sh
   git commit -m "Add some feature"
   ```
5. Push to the branch
   ```sh
   git push origin feature-branch
   ```
6. Create a new Pull Request.

## 📄 License

This project is licensed under the [MIT License](./LICENSE).

## 👏 Credits

Thanks to all contributors!

[<img src="https://github.com/StevenHuang41.png" width="50"/>](https://github.com/StevenHuang41)  [<img src="https://github.com/evelynhuang22.png" width="50"/>](https://github.com/evelynhuang22)


See the [contributors list](https://github.com/StevenHuang41/salary_prediction/graphs/contributors)
