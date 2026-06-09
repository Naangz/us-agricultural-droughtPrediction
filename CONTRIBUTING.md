Contributing
============

Thank you for your interest in contributing to this undergraduate final-year project.
Please follow these steps to report issues or submit improvements.

1. Filing an issue
   - Use the repository Issues to report bugs, propose enhancements, or ask questions.
   - Provide a clear title, a minimal reproduction (if applicable), and any relevant logs or screenshots.

2. Submitting a Pull Request
   - Fork the repository and create a feature branch from `main` named `feature/<short-desc>`.
   - Make your changes and ensure they are small and focused.
   - Run the project locally and verify there are no obvious regressions.
   - Submit a PR with a clear description of the changes and link to the related issue (if any).

3. Development environment
   - Recommended: create a conda environment from `requirements.txt`.

```bash
conda create -n bilstm-dev python=3.9
conda activate bilstm-dev
pip install -r requirements.txt
```

4. Code style and tests
   - Keep changes readable and documented. There are no automated tests currently; open an issue if you want to add test coverage.

5. Licensing and data
   - This project is licensed under the MIT License (see `LICENSE`).
   - Some datasets may not be redistributable; check file headers or the dataset source scripts before packaging data in a PR.

6. Contact and maintainers
   - Author: Undergraduate project author (add your name and contact in `README.md` or `MAINTAINERS` if desired).

7. Code of conduct
   - Treat contributors respectfully. For formal projects, consider adding a `CODE_OF_CONDUCT.md`.
