Listed directory Daleel
Viewed docker-compose.yml:1-31
Viewed .env.example:1-26
Listed directory frontend
Viewed .env.example:1-3
Viewed package.json:1-29

Here is a step-by-step guide you can provide for anyone trying to set up the project locally on another device from your GitHub repository. 

This guide assumes the user has **Docker**, **Node.js/npm**, and **Python** installed.

### 1. Clone the Repository
First, they need to clone your repository and navigate into it:
```bash
git clone <your-github-repo-url>
cd Daleel
```

### 2. Set Up the Databases (MinIO & Qdrant)
The databases run inside Docker containers. Since the user doesn't need your data, they can simply spin up empty instances using your existing configuration:
```bash
# From the root of the project, run:
docker-compose up -d
```
*Note: This will start MinIO (storage) on ports `9000` & `9001` and Qdrant (vector db) on ports `6333` & `6334`. You can access the MinIO console at `http://localhost:9001` using `minioadmin` / `minioadmin123`.*

### 3. Set Up the Environment Variables
Before running the backend or frontend, they need to set up the environment variables.

**For the Backend:**
1. In the root directory, copy the example environment file:
   ```bash
   cp .env.example .env
   ```
2. Open the `.env` file and fill in necessary API keys (like `GEMINI_API_KEY` if they need the language model functionality). The database connections are already pre-configured for the Docker containers.

**For the Frontend:**
1. Navigate to the frontend directory and copy the environment file:
   ```bash
   cd frontend
   cp .env.example .env
   ```
2. The default values in `.env` should already be pointing to `http://localhost:8000/api` which is correct for local testing.

### 4. Run the Backend Server (FastAPI)
Even if they just want the frontend and databases, the frontend will need the backend running to route its requests to the databases.
But first, they need to install the dependencies, using either a conda environment or standard venv. The python version should be 3.10.20.

**Conda**
```bash
# Open a new terminal in the root directory (Daleel)
# Create a conda environment
conda create -n daleel python=3.10.20

# Activate it
conda activate daleel

# Install the required dependencies
pip install --use-deprecated=legacy-resolver -r requirements.txt
```
*Note: A conflict error is expected, but the installation should still work. Due to the specific requirements of boxmot & realesrgan, this specific torch version is required, but will show as incompatible with other packages, yet the code should still work. This error is expected:*
```bash
ERROR: pip's legacy dependency resolver does not consider dependency conflicts when selecting packages. This behaviour is the source of the following dependency conflicts.
boxmot 18.0.0 requires torch<3.0.0,>=2.2.1, but you'll have torch 2.0.1+cu118 which is incompatible.
boxmot 18.0.0 requires torchvision<1.0.0,>=0.17.1, but you'll have torchvision 0.15.2+cu118 which is incompatible.
transformers 5.7.0 requires regex>=2025.10.22, but you'll have regex 2024.11.6 which is incompatible.
Successfully installed ...
```


**Standard venv**
```bash
# Open a new terminal in the root directory (Daleel)
# Create a virtual environment (optional but recommended)
python -m venv venv

# Activate it (Windows)
venv\Scripts\activate
# Activate it (Mac/Linux)
source venv/bin/activate

# Install the required dependencies
pip install --use-deprecated=legacy-resolver -r requirements.txt
```

**Finally, start the backend server:**
```bash
# Start the FastAPI server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Run the Frontend App (React/Vite)
Now they can start the user interface:

```bash
# Open a new terminal and navigate to the frontend directory
cd frontend

# Install Node modules
npm install

# Start the development server
npm run dev
```

The frontend will now be accessible (usually at `http://localhost:5173`). They will have a clean, working instance of the platform without any of the pre-ingested dataset images.