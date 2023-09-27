"""Investigating how to remove multiple inheritance structure."""
from __future__ import annotations

from logging import warning

from pydantic import BaseModel, ValidationError


class TopSingle(BaseModel):  # type: ignore[misc, unused-ignore]

    """
    Declare TopSingle.

    Parameters
    ----------
    a: int
        a attribute

    Returns
    -------
    TopSingle
        Object of the class TopSingle
    """

    a: int


class LeftSingle(TopSingle):

    """
    Declare LeftSingle.

    Parameters
    ----------
    a: int
        a attribute
    b: int
        b attribute

    Returns
    -------
    LeftSingle
        Object of the class LeftSingle
    """

    a: int
    b: int


class RightSingle(TopSingle):

    """
    Declare RightSingle.

    Parameters
    ----------
    a: int
        a attribute

    Returns
    -------
    RightSingle
        Object of the class RightSingle
    """

    a: int
    c: tuple[int, int] = (1, 2)

    def __init__(self: RightSingle, a: int) -> None:
        """
        Declare RightSingle.

        Parameters
        ----------
        a: int
            a attribute

        Returns
        -------
        RightSingle
            Object of the class RightSingle
        """
        super().__init__(a=a)

        self.a = a
        self.c = (1, 2)


class TopMulti:

    """
    Declare TopMulti.

    Parameters
    ----------
    a: int
        a attribute

    Returns
    -------
    TopMulti
        Object of the class TopMulti
    """

    a: int


class LeftMulti(BaseModel, TopMulti):  # type: ignore[misc, unused-ignore]

    """
    Declare LeftMulti.

    Parameters
    ----------
    a: int
        a attribute
    b: int
        b attribute

    Returns
    -------
    LeftMulti
        Object of the class LeftMulti
    """

    a: int
    b: int


class RightMulti(BaseModel, TopMulti):  # type: ignore[misc, unused-ignore]

    """
    Declare RightMulti.

    Parameters
    ----------
    a: int
        a attribute

    Returns
    -------
    RightMulti
        Object of the class RightMulti
    """

    a: int
    c: tuple[int, int] = (1, 2)

    def __init__(self: RightMulti, a: int) -> None:
        """
        Declare RightMulti.

        Parameters
        ----------
        a: int
            a attribute

        Returns
        -------
        RightMulti
            Object of the class RightMulti
        """
        super().__init__(a=a)
        self.a = a
        self.c = (1, 2)


if __name__ == "__main__":
    ts1 = TopSingle(a=1)
    ls1 = LeftSingle(a=1, b=2)
    try:
        rs1 = RightSingle(a=1)
    except ValidationError as exc:
        warning(str(exc))

    tm1 = TopMulti()
    lm1 = LeftMulti(a=1, b=2)
    try:
        rm1 = RightMulti(a=1)
    except ValidationError as exc:
        warning(str(exc))
