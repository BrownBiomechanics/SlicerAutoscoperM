import os


class ValueErrorsException(Exception):
    """A custom exception class that accepts a list of errors."""

    def __init__(self, errors):
        if not isinstance(errors, list):
            raise ValueError("The errors input must be a list")
        if len(errors) < 1:
            raise ValueError("The errors list must contain at least one error")
        self.errors = errors
        super().__init__("\n".join(errors))

    def __str__(self):
        err_str = "Invalid value{}".format("" if len(self.errors) == 0 else "s")

        return err_str + ":\n" + "\n".join(self.errors)


def validateInputs(*args: tuple, **kwargs: dict) -> None:
    """
    Validates that the provided inputs are not None or empty.

    :param args: list of inputs to validate
    :param kwargs: dictionary of inputs to validate
    :raises: ValueErrorsException
    """
    errors = []
    for arg in args:
        if arg is None:
            errors.append("Input argument is None")
        if isinstance(arg, str) and arg == "":
            errors.append("Input argument is an empty string")

    for name, arg in kwargs.items():
        if arg is None:
            errors.append(f"Input '{name}' is None")
        if isinstance(arg, str) and arg == "":
            errors.append(f"Input '{name}' is an empty string")

    if len(errors) > 0:
        raise ValueErrorsException(errors)


def validatePaths(*args: tuple, **kwargs: dict) -> None:
    """
    Checks that the provided paths exist.

    :param args: list of paths to validate
    :param kwargs: list of paths to validate
    :raises: ValueErrorsException
    """
    errors = []
    for arg in args:
        if not os.path.exists(arg):
            errors.append(f"Input path '{arg}' does not exist")

    for name, path in kwargs.items():
        if not os.path.exists(path):
            errors.append(f"Input path '{name}' ({path}) does not exist")

    if len(errors) > 0:
        raise ValueErrorsException(errors)
