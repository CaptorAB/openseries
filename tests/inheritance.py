"""Investigating how to remove multiple inheritance structure."""
from __future__ import annotations

from logging import warning
from typing import Optional

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
    c: Optional[tuple[int, int]] = None

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
        self.set_c()

    def set_c(self: RightSingle) -> None:
        """Set the attribute c."""
        self.c = (self.a, 2 * self.a)


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
    c: Optional[tuple[int, int]] = None

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
        self.set_c()

    def set_c(self: RightMulti) -> None:
        """Set the attribute c."""
        self.c = (self.a, 2 * self.a)


if __name__ == "__main__":
    ts1 = TopSingle(a=1)
    print("TopSingle", ts1)  # noqa: T201

    ls1 = LeftSingle(a=1, b=2)
    print("LeftSingle", ls1)  # noqa: T201

    try:
        rs1 = RightSingle(a=1)
        print("RightSingle", rs1)  # noqa: T201
    except ValidationError as exc:
        warning(str(exc))

    tm1 = TopMulti()
    print("TopMulti", tm1)  # noqa: T201

    lm1 = LeftMulti(a=1, b=2)
    print("LeftMulti", lm1)  # noqa: T201

    try:
        rm1 = RightMulti(a=1)
        print("RightMulti", rm1)  # noqa: T201
    except ValidationError as exc:
        warning(str(exc))
