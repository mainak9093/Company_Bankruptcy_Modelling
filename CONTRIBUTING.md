# Contributing Guidelines

Thank you for your interest in contributing to the Company Bankruptcy Prediction project! We welcome contributions that improve the model, documentation, or codebase.

## Ways to Contribute

1. **Report Bugs**: Open an issue describing the bug and how to reproduce it
2. **Suggest Enhancements**: Share ideas for new features or improvements
3. **Improve Documentation**: Fix typos, clarify instructions, or add examples
4. **Submit Code**: Implement new features or fix bugs with a pull request

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork**: `git clone https://github.com/YOUR_USERNAME/Company_Bankruptcy_Modelling.git`
3. **Create a branch**: `git checkout -b feature/your-feature-name`
4. **Make your changes** and commit with clear messages
5. **Push to your fork**: `git push origin feature/your-feature-name`
6. **Create a Pull Request** with a description of your changes

## Development Setup

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Format code
black src/ tests/

# Check code style
flake8 src/ tests/
isort src/ tests/
```

## Code Standards

- **Style**: Follow PEP 8 (use `black` for formatting)
- **Docstrings**: Add docstrings to functions and classes
- **Type Hints**: Use type hints for better code clarity
- **Comments**: Add comments for non-obvious logic

Example function:

```python
def calculate_class_weights(y_train: np.ndarray) -> dict:
    """Calculate class weights for imbalanced data.
    
    Args:
        y_train: Array of training labels
        
    Returns:
        Dictionary mapping class label to weight
    """
    majority_count = np.sum(y_train == 0)
    minority_count = np.sum(y_train == 1)
    weight_minority = majority_count / minority_count if minority_count > 0 else 1
    return {0: 1, 1: weight_minority}
```

## Testing

- Write tests for new functions in `tests/`
- Run all tests before submitting PR: `pytest tests/`
- Aim for >80% code coverage

## Commit Messages

Use clear, descriptive commit messages:

```
Good:
- feat: Add SHAP feature importance calculations
- fix: Handle missing values in data preprocessing
- docs: Update README with new training steps
- refactor: Simplify threshold optimization logic

Avoid:
- "fixed bug"
- "updated code"
- "test"
```

## Pull Request Process

1. **Update documentation** if you changed functionality
2. **Add/update tests** for your changes
3. **Run `pytest`** to ensure all tests pass
4. **Run `black src/`** to format code
5. **Write a clear PR description** including:
   - What changes were made
   - Why they were made
   - Any related issues

Example PR description:

```markdown
## Description
Implements SHAP-based feature importance analysis for model interpretability.

## Changes
- Added `model_interpretability.py` module
- New function `get_shap_values()` for feature importance
- Updated README with SHAP section

## Related Issues
Closes #42

## Testing
- Added tests in `tests/test_interpretability.py`
- Verified on sample dataset
- No breaking changes to existing API
```

## Areas for Contribution

### High Priority
- [ ] Add model deployment (Flask/FastAPI)
- [ ] Improve model interpretability (SHAP, LIME)
- [ ] Performance optimization
- [ ] Comprehensive unit tests

### Medium Priority
- [ ] Add data validation utilities
- [ ] Create visualization tools
- [ ] Add cross-validation support
- [ ] Hyperparameter tuning utilities

### Documentation
- [ ] Add architecture documentation
- [ ] Create Jupyter notebook tutorials
- [ ] Add API documentation
- [ ] Create deployment guides

## Questions?

- Check [GitHub Issues](https://github.com/mainak9093/Company_Bankruptcy_Modelling/issues) for similar questions
- Create a new issue with the `question` label
- Review existing code and comments

## Code of Conduct

- Be respectful and constructive
- Assume good intentions
- Help others learn and grow
- Report violations to maintainers

## Recognition

Contributors will be recognized in:
- This CONTRIBUTING.md file
- GitHub contributors page
- Release notes

Thank you for making this project better! 🎉
