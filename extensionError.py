class ExtensionError(Exception):
    ''' An error to be used when a provided file does not have the expected extension for whatever is being done. '''

    def __init__(self, extension : str) -> None:
        self.extension : str = extension
        self.message : str = f"The file extension {extension} is not currently implemented"
        super().__init__(self.message)
    
    def __str__(self) -> str:
        return f"ExtensionError: {self.message}"
