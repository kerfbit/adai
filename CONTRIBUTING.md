# Contributing to ADAI

Thank you for your interest in contributing to the Advanced Deep Learning AI (ADAI) project! This document provides guidelines for contributing.

## Quick Links

- **[Full Contributing Guide](docs/guides/contributing.md)** - Comprehensive contribution guidelines
- **[Building Guide](docs/guides/building.md)** - Build instructions and troubleshooting
- **[Git Workflow](docs/guides/git-workflow.md)** - Branching strategy and commit conventions
- **[CI/CD Documentation](docs/guides/ci-cd.md)** - Automated testing and deployment
- **[Technical Debt Tracker](TECHNICAL_DEBT.md)** - Known issues and improvements

## Getting Started

1. **Fork and Clone**
   ```bash
   git fork https://github.com/rjv717/adai.git
   cd adai
   ```

2. **Build**
   ```bash
   mkdir build && cd build
   cmake ..
   make -j$(nproc)
   ```

3. **Run Tests**
   ```bash
   ctest --output-on-failure
   ```

## Before You Submit

- [ ] Code follows C++ style guidelines (see [docs/guides/contributing.md](docs/guides/contributing.md))
- [ ] All tests pass locally
- [ ] Code is formatted: `./scripts/format_code.sh`
- [ ] No static analysis warnings: `./scripts/analyze_code.sh`
- [ ] New tests added for new features
- [ ] Documentation updated

## Pull Request Process

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Make your changes with clear, atomic commits
3. Push to your fork: `git push origin feature/your-feature`
4. Open a Pull Request against `main` branch
5. Wait for CI checks to pass
6. Address review feedback
7. Squash and merge once approved

## Commit Message Format

Use conventional commits:
```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:** feat, fix, docs, style, refactor, test, chore

**Example:**
```
feat(decoder): Add beam search generation

Implement beam search decoding algorithm for improved
text generation quality.

Closes #123
```

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on the code, not the person
- Help others learn and grow

## Questions?

- Open an issue for bugs or feature requests
- Use GitHub Discussions for questions
- Check existing issues before creating new ones

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.

---

For detailed guidelines, see the **[Full Contributing Guide](docs/guides/contributing.md)**.
