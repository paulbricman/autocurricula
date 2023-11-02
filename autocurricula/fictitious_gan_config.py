from autocurricula import AutocurriculumConfig


class FictitiousGANConfig(AutocurriculumConfig):
    def __init__(self, epochs: int = 4, rounds: int = 2):
        super().__init__(epochs, rounds)
