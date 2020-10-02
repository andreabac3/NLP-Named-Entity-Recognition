from typing import List


class Model:

    def predict(self, tokens: List[List[str]]) -> List[List[str]]:
        """
        A simple wrapper for your model

        Args:
            tokens: list of list of strings. The outer list represents the sentences, the inner one the tokens contained
            within it. Ex: [ ["This", "is", "the", "first", "homework"], ["Barack", "Obama", "was", "elected"] ]

        Returns:
            list of list of predictions associated to each token in the respective position.
            Ex: Ex: [ ["O", "O", "O", "O", "O"], ["PER", "PER", "O", "O"] ]

        """
        raise NotImplementedError
