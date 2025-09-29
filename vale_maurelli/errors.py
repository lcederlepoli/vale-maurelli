"""Fleishman and Vale & Maurelli Exceptions."""

class FitError(Exception):
    """Error to raise when a class instance is called 
    to generate a sample without the generator having 
    been fit yet.
    """
    def __init__(self, message=None, error_code="FitError") -> None:
        super().__init__(message)
        self.error_code = error_code
        return

class LoadError(Exception):
    """Error to raise when loading a serialized object 
    is not possible.
    """
    def __init__(self, message=None, error_code="LoadError") -> None:
        super().__init__(message)
        self.error_code = error_code
        return

class NormalizationError(Exception):
    """Error to raise when a given array-like object 
    is not normalized accordingly to the correct range 
    of values.
    """
    def __init__(self, message=None, error_code="NormalizationError") -> None:
        super().__init__(message)
        self.error_code = error_code
        return
