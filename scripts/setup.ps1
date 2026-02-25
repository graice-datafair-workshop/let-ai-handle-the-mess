# Change to project root directory (parent of scripts/)
Set-Location (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Workshop Setup - Let AI Handle the Mess"   -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Find a compatible Python version (3.12 or 3.13)
$pythonCmd = $null
$fullVersion = $null

# Try py launcher first
if (Get-Command py -ErrorAction SilentlyContinue) {
    foreach ($ver in @("3.13", "3.12")) {
        try {
            $output = & py "-$ver" --version 2>&1
            if ($LASTEXITCODE -eq 0) {
                $pythonCmd = "py"
                $pythonArgs = @("-$ver")
                $fullVersion = $output
                break
            }
        } catch {}
    }
}

# Fallback: try python3 and python directly
if (-not $pythonCmd) {
    foreach ($cmd in @("python3", "python")) {
        if (Get-Command $cmd -ErrorAction SilentlyContinue) {
            try {
                $output = & $cmd --version 2>&1
                if ($output -match "Python 3\.(12|13)\.") {
                    $pythonCmd = $cmd
                    $pythonArgs = @()
                    $fullVersion = $output
                    break
                }
            } catch {}
        }
    }
}

if (-not $pythonCmd) {
    Write-Host "WARNING: Could not find Python 3.12 or 3.13." -ForegroundColor Yellow
    Write-Host "   Python 3.14 is NOT supported by the packages used in this workshop." -ForegroundColor Yellow
    Write-Host ""

    # Try to install via winget
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        Write-Host "   winget is not available. Please install Python 3.13 manually from:" -ForegroundColor Red
        Write-Host "   https://www.python.org/downloads/" -ForegroundColor Blue
        exit 1
    }

    $confirm = Read-Host "   Install Python 3.13 automatically via winget? (Y/n)"
    if ($confirm -eq "" -or $confirm -eq "Y" -or $confirm -eq "y") {
        Write-Host ""
        Write-Host "Installing Python 3.13 via winget..." -ForegroundColor Cyan
        winget install Python.Python.3.13 --accept-package-agreements --accept-source-agreements
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Installation failed. Please install Python 3.13 manually from:" -ForegroundColor Red
            Write-Host "   https://www.python.org/downloads/" -ForegroundColor Blue
            exit 1
        }

        Write-Host "Python 3.13 installed." -ForegroundColor Green
        Write-Host ""

        # Refresh PATH so the new Python is found without restarting
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")

        # Try to find the newly installed Python
        if (Get-Command py -ErrorAction SilentlyContinue) {
            try {
                $output = & py "-3.13" --version 2>&1
                if ($LASTEXITCODE -eq 0) {
                    $pythonCmd = "py"
                    $pythonArgs = @("-3.13")
                    $fullVersion = $output
                }
            } catch {}
        }

        if (-not $pythonCmd) {
            Write-Host "Python was installed, but you may need to restart your terminal." -ForegroundColor Yellow
            Write-Host "Please close this window, open a new PowerShell, and run .\setup.ps1 again." -ForegroundColor Yellow
            exit 0
        }
    } else {
        Write-Host ""
        Write-Host "   Please install Python 3.13 manually from:" -ForegroundColor White
        Write-Host "   https://www.python.org/downloads/" -ForegroundColor Blue
        exit 1
    }
}

Write-Host "Found compatible Python: $fullVersion ($pythonCmd $($pythonArgs -join ' '))" -ForegroundColor Green
Write-Host ""

# Create virtual environment
if (Test-Path "venv") {
    Write-Host "A 'venv' folder already exists." -ForegroundColor Yellow
    $confirm = Read-Host "   Do you want to delete it and create a new one? (y/N)"
    if ($confirm -eq "y" -or $confirm -eq "Y") {
        Remove-Item -Recurse -Force "venv"
        Write-Host "   Removed old venv." -ForegroundColor Gray
    } else {
        Write-Host "   Keeping existing venv." -ForegroundColor Gray
    }
}

if (-not (Test-Path "venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Cyan
    & $pythonCmd @pythonArgs -m venv venv
    Write-Host "Virtual environment created." -ForegroundColor Green
}

Write-Host ""

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Cyan
& .\venv\Scripts\Activate.ps1

# Install dependencies
Write-Host "Installing required packages..." -ForegroundColor Cyan
Write-Host ""
pip install --upgrade pip -q
pip install -r requirements.txt
Write-Host ""

Write-Host "============================================" -ForegroundColor Green
Write-Host "  Setup complete!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""
Write-Host "To activate the environment in the future, run:"
Write-Host ""
Write-Host "  .\venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host ""
Write-Host "To start Jupyter Lab, run:"
Write-Host ""
Write-Host "  jupyter lab" -ForegroundColor White
Write-Host ""
