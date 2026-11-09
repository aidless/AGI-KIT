# Contributing

Thanks for your interest in improving this repository.

## Getting set up

```bash
git clone https://github.com/aidless/AGI-KIT.git
cd AGI-KIT
pip install -r requirements-test.txt
```

`requirements.txt` is the live experiment stack (Torch and similar). It is not required to run the unit tests.

## Verifying your change

```bash
make test
```

`make test` is the CI gate: `pytest tests/unit`. It does not download models or `data/`. Live scripts under `experiments/` are separate.

## Pull requests

- One logical change per pull request.
- Explain what breaks or is missing without the change.
- State how you verified it, and paste the command output.
- Keep unrelated reformatting out of the diff.

## Security

Do not open a public issue for a security problem. Use the private channel described
in [SECURITY.md](SECURITY.md).

## Code of conduct

Participation is governed by [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
