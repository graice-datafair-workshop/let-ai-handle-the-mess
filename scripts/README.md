# Setup Scripts

Automated setup scripts for the **Let AI Handle the Mess** workshop. Each script handles Python version detection, virtual environment creation, and dependency installation.

## Requirements

- **Python 3.12 or 3.13** (Python 3.14 is not supported)
- Internet connection (to install packages)

If a compatible Python version is not found, the scripts will offer to install Python 3.13 automatically using your platform's package manager.

## Usage

All scripts can be run from either the **project root** or the **scripts/** directory -- they will automatically change to the project root before executing.

### macOS / Linux

```bash
bash scripts/setup.sh
```

Or from within the scripts directory:

```bash
cd scripts
bash setup.sh
```

> You can also make it executable and run directly:
> ```bash
> chmod +x scripts/setup.sh
> ./scripts/setup.sh
> ```

### Windows (PowerShell)

```powershell
.\scripts\setup.ps1
```

If you get an execution policy error, run this first:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

### Windows (Command Prompt)

```cmd
scripts\setup.bat
```

## What the scripts do

1. **Detect Python** -- Searches for Python 3.12 or 3.13 on your system.
2. **Auto-install (optional)** -- If no compatible version is found, offers to install Python 3.13 via Homebrew (macOS), apt/dnf (Linux), or winget (Windows).
3. **Create virtual environment** -- Creates a `venv/` directory in the project root using the detected Python.
4. **Install dependencies** -- Activates the environment and installs all packages from `requirements.txt`.

## After setup

Activate the environment and start Jupyter Lab or use your preferred editor:

```bash
# macOS / Linux
source venv/bin/activate
jupyter lab

# Windows (PowerShell)
.\venv\Scripts\Activate.ps1
jupyter lab

# Windows (Command Prompt)
venv\Scripts\activate.bat
jupyter lab
```

## Troubleshooting

| Problem | Solution |
|---|---|
| Script says Python not found but it is installed | Make sure Python 3.12 or 3.13 is on your PATH. Run `python3 --version` to verify. |
| Permission denied (Linux/macOS) | Run `chmod +x scripts/setup.sh` then try again. |
| PowerShell execution policy error | Run `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` |
| Packages fail to install | Check your internet connection and try running `pip install -r requirements.txt` manually inside the activated venv. |
| venv already exists | The script will ask if you want to recreate it. Choose **y** to start fresh. |
