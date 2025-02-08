
# Contributing to CRISPy

Thank you for considering contributing to **CRISPy**! Here’s how to set up your environment and build the documentation locally.

## Setting Up a Virtual Environment

1. **Create and Activate a Virtual Environment**

   On Linux/macOS:
   ```
   python3.12 -m venv crispy-env
   source crispy-env/bin/activate
   ```

   On Windows:
   ```
   python -m venv crispy-env
   crispy-env\Scripts\activate
   ```

2. **Install Build Tools**
   ```
   pip install --upgrade pip setuptools wheel build
   ```

3. **Install Project Dependencies**
   ```
   pip install .[docs]
   ```

4. **Build the Documentation**
   Navigate to the `docs` directory:
   ```
   cd docs
   ```

   Run the Sphinx build command:
   ```
   sphinx-build -b html source build
   ```

   The HTML files will be in the `docs/build` directory.

5. **Deactivate the Virtual Environment**
   When finished, deactivate the virtual environment:
   ```
   deactivate
   ```
   
## For building and distributing to PyPI:

1. **Install from Test PyPI**

   Navigate to the `docs` directory:
   ```
   pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ crispy-learn
   ```

## Additional Notes

- Ensure you’re using Python 3.8 or higher.
- If you encounter issues, check that all dependencies are correctly installed in the virtual environment.
- Use `pip install` to add any missing dependencies.
