Write-Host "=== [SETUP] Initialisation de l'environnement Conda ==="

$envName = "papertrading"
$pythonVersion = "3.10"

# 0) Vérification conda
if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Error "Conda n'est pas disponible dans ce shell. Charge Anaconda/Miniconda avant de relancer (ex: 'conda init powershell' puis rouvre le terminal)."
    exit 1
}

# 1) Création env conda si absent
$envExists = (conda env list | Select-String " $envName ").Length -gt 0
if (-not $envExists) {
    Write-Host "Création de l'environnement conda '$envName' (python=$pythonVersion)..."
    conda create -y -n $envName python=$pythonVersion
}

# 2) Activation env
Write-Host "Activation de l'environnement '$envName'..."
conda activate $envName

# 2bis) Nettoyage préventif des polices matplotlib (verrous sous Windows)
if ($env:CONDA_PREFIX) {
    $fontFile = Join-Path $env:CONDA_PREFIX "Lib\site-packages\matplotlib\mpl-data\fonts\ttf\DejaVuSans.ttf"
    if (Test-Path $fontFile) {
        try {
            Remove-Item -Force $fontFile -ErrorAction SilentlyContinue
            Write-Host "Police matplotlib supprimée pour éviter les verrous: $fontFile"
        } catch {
            Write-Warning "Impossible de supprimer $fontFile (peut être verrouillé)."
        }
    }
}

# 3) Upgrade pip + wheel
python -m pip install --upgrade pip wheel

# 4) Installation des dépendances
# - Si l'env existe déjà : on n'installe pas tout à zéro, mais on peut mettre à jour via requirements.txt.
# - Si requirements.txt est absent et l'env existe déjà : on ne touche pas aux packages.
if (Test-Path "requirements.txt") {
    Write-Host "Installation/mise à jour depuis requirements.txt..."
    try {
        pip install --upgrade -r requirements.txt
    } catch {
        Write-Warning "Installation pip échouée, tentative de nettoyage matplotlib puis retry..."
        if ($env:CONDA_PREFIX) {
            $fontDir = Join-Path $env:CONDA_PREFIX "Lib\site-packages\matplotlib\mpl-data\fonts"
            if (Test-Path $fontDir) {
                try { Remove-Item -Recurse -Force $fontDir } catch { Write-Warning "Cleanup fonts échoué." }
            }
        }
        pip install --upgrade --no-cache-dir -r requirements.txt
    }
    Write-Host "Installing Playwright browsers..." -ForegroundColor Cyan
    python -m playwright install --with-deps
} elseif (-not $envExists) {
    Write-Host "Installation minimale (streamlit/numpy/pandas/scipy/plotly/torch/tensorflow)..."
    pip install streamlit numpy pandas scipy plotly tensorflow torch
} else {
    Write-Host "Env existant et aucun requirements.txt : aucune installation de paquets."
}

# 5) Préparation des répertoires
Write-Host "Préparation des répertoires..."
mkdir cache -ErrorAction Ignore | Out-Null
mkdir data -ErrorAction Ignore | Out-Null
mkdir logs -ErrorAction Ignore | Out-Null

# 6) Création fichier .env si absent
if (-not (Test-Path ".env")) {
    "OPENAI_API_KEY=" | Out-File ".env"
}

# 7) Sanity-check compilation (ignore conda internals)
Write-Host "=== [SYNTAX CHECK] ==="
$pyFiles = Get-ChildItem -Recurse -Filter *.py -File | Where-Object {
    $_.FullName -notlike "*\__pycache__\*"
}
foreach ($file in $pyFiles) {
    python -m py_compile $file.FullName
}

Write-Host "=== Setup Conda terminé ==="
