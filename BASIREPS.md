# basireps Integration

The `magneupy` package uses the `basireps` program from the FullProf Suite for representation analysis.

## How it works

The package looks for `basireps` in the following order:
1.  Locally in the current directory as `basireps_linux`, `basireps_mac`, or `basireps_win.exe`.
2.  In your system's `PATH` as `basireps`.

## How to install basireps

If you don't have `basireps` installed, you can obtain it as part of the FullProf Suite:

1.  Download the FullProf Suite from the [official website](https://www.ill.eu/sites/fullprof/).
2.  Install the suite on your system.
3.  Ensure the directory containing the `basireps` binary is added to your system's `PATH`.

Alternatively, you can find the `basireps` binary for your platform and place it in your working directory, renaming it to:
- `basireps_linux` (on Linux)
- `basireps_mac` (on macOS)
- `basireps_win.exe` (on Windows)
