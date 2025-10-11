# TOPSISX Installation & Setup Guide

## 📦 Package Structure

After following this guide, your package structure should be:

```
topsisx/
├── topsisx/
│   ├── __init__.py
│   ├── topsis.py
│   ├── vikor.py
│   ├── ahp.py
│   ├── entropy.py
│   ├── pipeline.py
│   ├── cli.py
│   ├── app.py          ← Web interface (copy from artifact)
│   ├── reports.py
│   ├── utils.py
│   └── visualizations.py
├── setup.py
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
└── MANIFEST.in         ← New file needed
```

---

## 🚀 Step-by-Step Setup

### Step 1: Copy the Web App

Create `topsisx/app.py` by copying the complete web interface code from the artifact named "app.py - Complete Web Interface".

### Step 2: Create MANIFEST.in

Create a file named `MANIFEST.in` in the root directory:

```
include README.md
include LICENSE
include requirements.txt
recursive-include topsisx *.py
```

### Step 3: Update Files

Replace these files with the corrected versions from the artifacts:
- `topsisx/topsis.py`
- `topsisx/vikor.py`
- `topsisx/ahp.py`
- `topsisx/pipeline.py`
- `topsisx/cli.py`
- `setup.py`
- `README.md`

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Install Package Locally (for testing)

```bash
# Development mode
pip install -e .

# Or regular installation
pip install .
```

---

## ✅ Testing Installation

### Test 1: Check Installation

```bash
topsisx --version
```

Expected output: `TOPSISX 0.1.3`

### Test 2: Launch Web Interface

```bash
topsisx --web
```

This should:
1. Start a Streamlit server
2. Open your browser automatically
3. Display the TOPSISX web interface

### Test 3: CLI Analysis

Create a test file `test_data.csv`:

```csv
Alternative,Cost,Quality,Time
A,250,16,12
B,200,20,8
C,300,12,10
```

Run analysis:

```bash
topsisx test_data.csv --impacts "-,+,-" --output test_results.csv
```

### Test 4: Python API

```python
from topsisx.pipeline import DecisionPipeline
import pandas as pd

data = pd.DataFrame({
    'Cost': [250, 200, 300],
    'Quality': [16, 20, 12]
})

pipeline = DecisionPipeline(weights='entropy', method='topsis')
result = pipeline.run(data, impacts=['-', '+'])
print(result)
```

---

## 📤 Publishing to PyPI

### Step 1: Prepare for Publishing

```bash
# Install build tools
pip install build twine

# Clean previous builds
rm -rf dist/ build/ *.egg-info
```

### Step 2: Build Distribution

```bash
python -m build
```

This creates:
- `dist/topsisx-0.1.3.tar.gz` (source distribution)
- `dist/topsisx-0.1.3-py3-none-any.whl` (wheel distribution)

### Step 3: Test on TestPyPI (Optional but Recommended)

```bash
# Upload to TestPyPI
python -m twine upload --repository testpypi dist/*

# Test installation
pip install --index-url https://test.pypi.org/simple/ topsisx
```

### Step 4: Upload to PyPI

```bash
# Upload to PyPI
python -m twine upload dist/*

# You'll need to enter your PyPI credentials
# Username: __token__
# Password: pypi-your-token-here
```

### Step 5: Verify Installation

```bash
# Install from PyPI
pip install topsisx

# Test
topsisx --version
topsisx --web
```

---

## 🔧 Troubleshooting

### Issue: "topsisx: command not found"

**Solution:**
```bash
# Reinstall package
pip uninstall topsisx
pip install topsisx

# Or if developing locally
pip install -e .
```

### Issue: "app.py not found"

**Solution:**
Ensure `app.py` is in the `topsisx/` package directory, not the root directory.

### Issue: Streamlit not launching

**Solution:**
```bash
# Install streamlit explicitly
pip install streamlit

# Try manual launch
python -m streamlit run topsisx/app.py
```

### Issue: Import errors

**Solution:**
```bash
# Check if package is installed
pip list | grep topsisx

# Reinstall with dependencies
pip install --force-reinstall topsisx
```

### Issue: FPDF errors in report generation

**Solution:**
```bash
# Install correct FPDF version
pip install fpdf==1.7.2
```

---

## 🎯 Quick Commands Reference

```bash
# Install package
pip install topsisx

# Launch web interface
topsisx --web

# CLI analysis
topsisx data.csv --impacts "+,-,+"

# Get help
topsisx --help

# Check version
topsisx --version

# Update package
pip install --upgrade topsisx

# Uninstall
pip uninstall topsisx
```

---

## 📝 For Users Installing from PyPI

Once published on PyPI, users just need to:

```bash
# 1. Install
pip install topsisx

# 2. Launch web interface
topsisx --web

# That's it! No configuration needed.
```

The web interface will automatically:
- ✅ Open in their default browser
- ✅ Provide sample datasets
- ✅ Allow CSV upload
- ✅ Show interactive visualizations
- ✅ Enable result downloads

---

## 🌐 User Experience Flow

### For Non-Technical Users:

1. Install: `pip install topsisx`
2. Run: `topsisx --web`
3. Upload CSV or use samples
4. Click "Run Analysis"
5. Download results

### For Technical Users:

```python
from topsisx import DecisionPipeline
import pandas as pd

# Load data
df = pd.read_csv('data.csv')

# Analyze
pipeline = DecisionPipeline(weights='entropy', method='topsis')
result = pipeline.run(df, impacts=['+', '-', '+'])

# Export
result.to_csv('results.csv')
```

---

## ✨ Next Steps

After successful installation:

1. **Try the web interface**: `topsisx --web`
2. **Read the docs**: Check README.md for examples
3. **Test with your data**: Upload your own CSV files
4. **Share feedback**: Report issues on GitHub

---

## 📞 Need Help?

- 📖 [Full Documentation](https://github.com/SuvitKumar003/ranklib)
- 🐛 [Report Issues](https://github.com/SuvitKumar003/ranklib/issues)
- 💬 [Ask Questions](https://github.com/SuvitKumar003/ranklib/discussions)
- 📧 Email: suvitkumar03@gmail.com

---

**Happy Decision Making! 🎯**