"""Investigating how to remove multiple inheritance structure."""
from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel

TypeTopSingle = TypeVar("TypeTopSingle", bound="TopSingle")
TypeLeftSingle = TypeVar("TypeLeftSingle", bound="LeftSingle")
TypeRightSingle = TypeVar("TypeRightSingle", bound="RightSingle")
TypeTopMulti = TypeVar("TypeTopMulti", bound="TopMulti")
TypeLeftMulti = TypeVar("TypeLeftMulti", bound="LeftMulti")
TypeRightMulti = TypeVar("TypeRightMulti", bound="RightMulti")


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
    c: tuple[int, int]

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


class RightMulti(BaseModel, TopSingle):  # type: ignore[misc]

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
    c: tuple[int, int]

    def __init__(self: RightMulti, a: int) -> None:
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
        super().__init__()

        self.a = a
        self.c = (1, 2)


if __name__ == "__main__":
    pass
