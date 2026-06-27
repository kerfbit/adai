# ADAI Documentation

Welcome to the Advanced Deep Learning AI (ADAI) documentation. This guide will help you navigate through all aspects of the project, from getting started to deep architectural understanding.

## Quick Start

### For New Users

- **[Chatbot User Guide](operations/guides/chatbot-guide.md)** - Using the chatbot
- **[Training Guide](operations/guides/training-guide.md)** - How to train models
- **[Operations Manual](operations/OPERATIONS_MANUAL.md)** - Comprehensive operations reference
- **[Docker Deployment](operations/deployment/docker.md)** - Containerized deployment

### For New Developers

- **[Contributing Guide](development/guides/workflow/contributing.md)** - How to contribute
- **[Building ADAI](development/guides/building/building.md)** - Build instructions
- **[Git Workflow](development/guides/workflow/git-workflow.md)** - Git best practices
- **[API Reference](development/api/)** - Component APIs

## Documentation Structure

### [Development Documentation](development/)

For developers, contributors, and architects:

- **[API Documentation](development/api/)** - Component APIs and interfaces
- **[Architecture](development/architecture/)** - System design and architecture
- **[Guides](development/guides/)** - Organized into subdirectories:
  - [building/](development/guides/building/) - Building, CI/CD, cross-compilation
  - [workflow/](development/guides/workflow/) - Git workflow, contributing, branch protection
  - [training/](development/guides/training/) - Training pipeline, data pipeline, datasets
  - [features/](development/guides/features/) - Feature implementation guides (RAG, OpenMP, augmentation, etc.)
  - [internals/](development/guides/internals/) - System internals (CLI, trainer architecture)
  - [quick-reference/](development/guides/quick-reference/) - Developer cheat sheets
- **[Testing](development/testing/)** - Test specifications and coverage reports
- **[Reference](development/reference/)** - Technical reference materials
- **[Archive](development/archive/)** - Historical documentation and phase summaries

### [Operations Documentation](operations/)

For users, data scientists, and system administrators:

- **[User Guides](operations/guides/)** - Chatbot, training, model management, dataset management
- **[Deployment](operations/deployment/)** - Docker, systemd, and server bundle deployment
- **[Troubleshooting](operations/guides/troubleshooting/)** - Common issues and solutions
- **[Quick References](operations/guides/quick-reference/)** - GUI and Windows build cheat sheets

### [Lessons](lessons/)

Educational deep-dives on ML training and transformer architecture:

- Learning rate selection, batch size, gradient clipping, weight initialization
- Regularization, precision/stability, evaluation strategies, reproducibility
- [Architecting Attention](lessons/architecting-attention.md) - Comprehensive attention mechanism tutorial
- [Core Model Components](lessons/core-model-components.md) - Transformer component reference

### [Proposals](proposals/)

Design proposals for planned features. See [proposals/README.md](proposals/README.md) for the full index with status.

## Directory Layout

```text
docs/
├── README.md                     # This file
├── development/                  # Developer documentation
│   ├── api/                     #   Component API reference
│   ├── architecture/            #   System design
│   ├── guides/                  #   Developer guides
│   │   ├── building/       #     Build & CI/CD
│   │   ├── workflow/           #     Git workflow & contributing
│   │   ├── training/           #     Training & data pipeline
│   │   ├── features/           #     Feature implementations
│   │   ├── internals/          #     System internals
│   │   └── quick-reference/    #     Developer cheat sheets
│   ├── testing/                 #   Test specs & coverage
│   ├── reference/               #   Technical reference
│   └── archive/                 #   Historical docs
├── operations/                   # User & operations documentation
│   ├── guides/                  #   User guides
│   │   ├── quick-reference/    #     Quick reference cards
│   │   └── troubleshooting/    #     Common issues & fixes
│   └── deployment/              #   Docker, systemd, server bundle
├── lessons/                      # ML training lessons
└── proposals/                    # Feature proposals
```

## Documentation Standards

- **File naming:** Use `lower-kebab-case.md` for new files. `UPPER_SNAKE_CASE` only for README.md, CHANGELOG, LICENSE.
- **Audience separation:** Development docs go under `development/`, user-facing docs go under `operations/`.
- **Quick references:** Always go in the `quick-reference/` subdirectory under the appropriate audience directory.
- **Proposals:** All proposals go in the top-level `proposals/` directory.
- **Completed work:** Move status/completion reports to `development/archive/` once the feature is shipped.

## License

See [../LICENSE](../LICENSE) for details.
