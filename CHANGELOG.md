# Changelog

## [1.1.1] - 2026-07-15

### Fixed
- macOS Intel now uses the last officially published Intel-compatible PyTorch pair: 2.2.2 and torchvision 0.17.2.
- macOS windowed self-tests write their result to a file instead of requiring a console stream.
- macOS application bundles are ad-hoc signed and verified before packaging.
- GitHub Actions dependencies were updated to Node.js 24-compatible major versions.

### Changed
- CUDA builds are opt-in per platform and disabled by default because standalone CUDA archives are approximately 3 GB.
- Ordinary tag pushes publish CPU applications only; CUDA can be added by manually dispatching the workflow on the tag.
- Packaged smoke-test failures now print captured stdout and stderr for easier diagnosis.

## [1.1.0] - 2026-07-15

### Added
- GitHub Actions matrix builds for Windows, Linux, macOS Intel and macOS Apple Silicon.
- Automatic GitHub Release publication on `v*` tags.
- Separate CPU and optional CUDA release archives for Windows and Linux, pinned to PyTorch 2.5.1/torchvision 0.20.1; CUDA editions bundle the CUDA 11.8 runtime.
- Hardware menu with CUDA availability probe and build information.
- CUDA acceleration checkbox that is enabled only when the runtime and NVIDIA GPU are usable.
- Tests for CUDA detection and recursive DeeperHistReg device configuration.
- A packaged-executable smoke test in every platform job before archives are uploaded.
- Reproducible PyInstaller and release-packaging scripts.

### Changed
- CPU remains the default, but CUDA is no longer forcibly disabled or monkey-patched.
- DeeperHistReg is installed from its published package during builds rather than relying on untracked local folders.
- GUI updates from registration workers are dispatched safely to the Tk main thread.

## [1.0] - 2026

### Added
- Initial public release.
- CPU-only DeeperHistReg integration.
- Dynamic preset discovery.
- Optional intermediate outputs.
- Automatic displacement field detection.
- Warped TIFF output and error logging.
