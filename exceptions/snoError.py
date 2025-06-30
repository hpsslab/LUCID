class ShouldNotOccurError(Exception):
    ''' An error to be used when we reach a code block that should (theoretically) be impossible to reach. '''

    def __init__(self) -> None:
        self.message : str = f"This code block should be unreachable. Look into this"
        super().__init__(self.message)
    
    def __str__(self) -> str:
        return f"SNOError: {self.message}"
