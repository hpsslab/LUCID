class DuplicateError(Exception):
    ''' An error to be used when we try to add a file into the KG that is already in the KG. '''

    def __init__(self, filename : str) -> None:
        self.message : str = f"The file {filename} has already been added to the KG"
        self.filename = filename
        super().__init__(self.message)
    
    def __str__(self) -> str:
        return f"DuplicateError: {self.message}"
