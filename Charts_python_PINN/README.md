# PINN interface (Charts_python_PINN)

## Once
1. Build the C++ project (the main that reads the JSON file is needed):
   ```powershell
   cd ..\C++_Project
   cmake -S . -B build -G "Visual Studio 17 2022" -A x64
   cmake --build build --config Release
   ```
2. Install the Python libraries (from the VS Code terminal, in this folder):
   ```powershell
   python -m pip install -r requirements.txt
   ```

## Every time
```powershell
cd Charts_python_PINN
python -m streamlit run app.py
```
The page opens in the browser. To close it: Ctrl+C in the terminal.

## Files
- `app.py`              the page (inputs, start, progress, results)
- `settings.py`         default values and construction of the JSON file for pinn.exe
- `runner.py`           start of pinn.exe and reading of the progress
- `results.py`          reading of the CSV files of a run
- `charts.py`           charts (Plotly: the camera icon saves them as PNG)
- `example_config.json` complete example of the file read by `pinn.exe --config`
- `runs/`               one folder per run (config.json, log.txt, CSV files)

From the command line, without the interface:
`pinn.exe --config example_config.json --out results_folder`

To compare the PINN prices with Longstaff-Schwartz and the binomial tree, use the Monte Carlo interface
in `../Charts_python_MonteCarlo` with the same option, market and S0.
