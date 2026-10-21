"""Error hierarchy. The CLI maps `ConfigError` to exit code 2 and the rest to 1."""


class AnvilError(Exception):
    """Base class for every error Anvil raises on purpose."""

    exit_code = 1


class ConfigError(AnvilError):
    """The spec, the inputs or the environment are wrong; the user has to fix them."""

    exit_code = 2


class TeacherError(AnvilError):
    """The teacher could not produce a valid label (API failure, refusal, bad output)."""


class StoreError(AnvilError):
    """The label store is missing data a command needs, or is inconsistent."""


class StudentError(AnvilError):
    """A student could not be built, trained, saved or loaded."""


class AlignmentError(ConfigError):
    """Too many teacher spans could not be matched to the input or the student's tokens: the
    prompt or the labels need fixing before a student can be trained on them."""


class ExportError(AnvilError):
    """A student could not be exported to ONNX, failed its parity gate, or an exported bundle
    is missing or incomplete."""


class HubError(AnvilError):
    """Publishing to the Hugging Face Hub failed (network, permissions, a rejected upload)."""
