# Contributing

This project welcomes contributions and suggestions. Most contributions require you to
agree to a Contributor License Agreement (CLA) declaring that you have the right to, and
actually do, grant us the rights to use your contribution. For details, visit
https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to
provide a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow
the instructions provided by the bot. You will only need to do this once across all repos
using our CLA.

This project has adopted the
[Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the
[Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact
[opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or
comments.

## Developing

`faithgap` is pure standard library (Python ≥ 3.9); there are no runtime dependencies to
install. To work on it:

```bash
pip install -e ".[dev]"   # installs pytest
pytest -q                 # all tests are offline and deterministic
python examples/quickstart.py
```

Please keep the core dependency-free, add a test to `tests/test_smoke.py` for any behavior
change, and match the surrounding docstring style.
